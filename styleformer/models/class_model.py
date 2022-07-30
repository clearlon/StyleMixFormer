import os
import numpy as np
import torch
from collections import OrderedDict
from os import path as osp

from styleformer.archs import build_network
from styleformer.losses import build_loss
from styleformer.utils import get_root_logger
from styleformer.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class ClassModel(BaseModel):
    """Image filter removal model."""

    def __init__(self, opt):
        super().__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True))
        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        if train_opt.get('class_opt'):
            self.margin_loss = build_loss(train_opt['class_opt']).to(self.device)
        else:
            self.margin_loss = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq']
        if isinstance(self.lq, list):
            self.lq = [lq.to(self.device) for lq in self.lq]
        else:
            self.lq = self.lq.to(self.device)

        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if 'label' in data:
            self.label = data['label'].to(self.device)

    def run_network(self):
        self.output = self.net_g(self.lq)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.run_network()
        
        l_total = 0
        loss_dict = OrderedDict()

        # margin loss (softmax, A-softmax, ArcFace...)
        if self.margin_loss:
            margin_weight = self.opt['train']['class_opt']['loss_weight']
            l_margin = self.margin_loss(self.output, self.gt) * margin_weight  # output is embedding, gt is label
            l_total += l_margin
            # loss_dict['l_margin'] = l_margin
            
        loss_dict['l_total'] = l_total
        l_total.backward()

        ############################################################################
        ## gradient clipping
        use_grad_clip = self.opt['train'].get('use_grad_clip', None)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        # calculate psnr for batch
        self.psnr_train = None

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.run_network()

            self.net_g.train()

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.pred.detach().cpu()
        if self.return_HR:
            out_dict['HR'] = self.out_HR.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
