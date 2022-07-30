import imageio
import cv2
import math
import numpy as np
import scipy.ndimage as ndimage
import os
import torch
from torchvision.utils import make_grid


def save_results_yuv(pred, index, test_img_dir):
    test_pred = np.squeeze(pred)
    # print(np.max(test_pred))
    # print(np.min(test_pred))
    # test_pred = np.clip(test_pred, 0, 1)
    # test_pred = np.uint16(test_pred)

    # split image
    pred_y = test_pred[:, :, 0]
    pred_u = test_pred[:, :, 1]
    pred_v = test_pred[:, :, 2]

    # save prediction - must be saved in separate channels due to 16-bit pixel depth
    imageio.imwrite(os.path.join(test_img_dir, "{}-y_pred.png".format(str(int(index) + 1).zfill(2))),
                    pred_y)
    imageio.imwrite(os.path.join(test_img_dir, "{}-u_pred.png".format(str(int(index) + 1).zfill(2))),
                    pred_u)
    imageio.imwrite(os.path.join(test_img_dir, "{}-v_pred.png".format(str(int(index) + 1).zfill(2))),
                    pred_v)

def yuv2rgb_2020(Dy, Du, Dv, mode='420', color_primaries='bt2020', yuv_bit_depth=10, quantification=16, color_range='LimitedRange'):
    """
    only for 420 or 444 yuv mode,default 420
    input:Y,U,V（default 10bit）
    output:Drgb（default uint16)
    """
    # YUV2RGB matrix
    if color_primaries == 'bt2020':
        # BT.2020
        data_type = np.uint16
        assert yuv_bit_depth == 10, ('if color primaries is bt2020, the bit depth have to set to 10')
        rgb2yuv_bt2020_matrix = np.matrix([[0.2627, 0.6780, 0.0593],
                                        [-0.1396, -0.3604, 0.5000],
                                        [0.5000, -0.4598, -0.0402]])
        yuv2rgb_matrix = rgb2yuv_bt2020_matrix.I
    elif color_primaries == 'bt709':
        data_type = np.uint8
        assert yuv_bit_depth == 8 and quantification == 8, ('if color primaries is bt709, the bit depth and quantificaition have to set to 8')
        # BT.709
        yuv2rgb_matrix = np.matrix([[1.0000, 0.0000, 1.5747],
                                [1.0000, -0.1873, -0.4682],
                                [1.0000, 1.8556, 0.0000]])
    else:
        raise ValueError(f'unsupport {color_primaries} type')

    if mode == '420':
        #U,V分量resize成Y的shape，以便进行矩阵运算
        Du = ndimage.zoom(Du, 2, order=0) #使用最近邻插值法
        Dv = ndimage.zoom(Dv, 2, order=0)

    h, w = Dy.shape
    #YUV去量化
    if color_range == 'LimitedRange':
        Ey = ((Dy/2 ** (yuv_bit_depth - 8) - 16)/219).flatten()
        Eu = ((Du/2 ** (yuv_bit_depth - 8) - 128)/224).flatten()
        Ev = ((Dv/2 ** (yuv_bit_depth - 8) - 128)/224).flatten()
    else:
        #Full range
        Ey, Eu, Ev = (Dy / 1023).flatten(), (Du / 1023).flatten(), (Dv / 1023).flatten()

    Eyuv = np.array([Ey,Eu,Ev])

    Ergb = np.dot(yuv2rgb_matrix, Eyuv)
    Er = Ergb[0,:].reshape(h,w).clip(0,1)
    Eg = Ergb[1,:].reshape(h,w).clip(0,1)
    Eb = Ergb[2,:].reshape(h,w).clip(0,1)
    #RGB量化
    if color_range == 'LimitedRange':
        Dr = np.round((219*Er+16)*2**(quantification - 8))
        Dg = np.round((219*Eg+16)*2**(quantification - 8))
        Db = np.round((219*Eb+16)*2**(quantification - 8))
    else:
        Dr = np.round((2 ** quantification - 1) * Er).clip(0, 2 ** quantification - 1)
        Dg = np.round((2 ** quantification - 1) * Eg).clip(0, 2 ** quantification - 1)
        Db = np.round((2 ** quantification - 1) * Eb).clip(0, 2 ** quantification - 1)

    # a = [Dr,Dg,Db]
    Drgb = np.array([Dr,Dg,Db]).astype(data_type)
    Drgb = Drgb.transpose(1,2,0)  #(3,1920,1080)->(1920,1080,3)
    return Drgb

def rgb2yuv(Drgb, color_primaries='bt2020', rgb_bit_depth = 16, quantification=10):
    '''
    input: Drgb 16bit的PNG
    output: Dy, Du, Dv
    '''
    h, w, _ = Drgb.shape
    #RGB去量化  输入为16bit PNG, 若为8bit,则 /1
    Er = ((Drgb[:,:,0] / 2 **(rgb_bit_depth - 8) - 16) / 219).flatten()
    Eg = ((Drgb[:,:,1] / 2 **(rgb_bit_depth - 8) - 16) / 219).flatten()
    Eb = ((Drgb[:,:,2] / 2 **(rgb_bit_depth - 8) - 16) / 219).flatten()
    Ergb = np.array([Er,Eg,Eb])
    #RGB2YUV
    if color_primaries == 'bt2020':
        data_type = np.uint16
        rgb2yuv_matrix = np.matrix([[0.2627, 0.6780, 0.0593],
                                            [-0.1396, -0.3604, 0.5000],
                                            [0.5000, -0.4598, -0.0402]])

    if color_primaries == 'bt709':
        data_type = np.uint8
        yuv2rgb_bt709_matrix = np.matrix([[1.0000, 0.0000, 1.5747],
                                    [1.0000, -0.1873, -0.4682],
                                    [1.0000, 1.8556, 0.0000]])
        rgb2yuv_matrix = yuv2rgb_bt709_matrix.I

    Eyuv = np.dot(rgb2yuv_matrix, Ergb)
    Ey = Eyuv[0,:].reshape(h, w)
    Eu = Eyuv[1,:].reshape(h, w)
    Ev = Eyuv[2,:].reshape(h, w)
    #YUV量化 10bit   若是16bit，则2**8
    Dy = np.round(((219*Ey+16)*2**(quantification-8))).clip(0,2 ** quantification - 1).astype(data_type)
    Du = np.round(((224*Eu+128)*2**(quantification-8))).clip(0,2 ** quantification - 1).astype(data_type)
    Dv = np.round(((224*Ev+128)*2**(quantification-8))).clip(0,2 ** quantification - 1).astype(data_type)

    Dyuv = np.array([Dy,Du,Dv]).astype(data_type)
    Dyuv = Dyuv.transpose(1,2,0)  #(3,1920,1080)->(1920,1080,3)
    return Dyuv


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint16, min_max=(0, 1), data_range=65535.):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError('Only support 4D, 3D or 2D tensor. ' f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        if out_type == np.uint16:
            img_np = (img_np * data_range).round()

        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


def tensor2img_fast(tensor, rgb2bgr=True, min_max=(0, 1), uint16=True):
    """This implementation is slightly faster than tensor2img.
    It now only supports torch tensor with shape (1, c, h, w).

    Args:
        tensor (Tensor): Now only support torch tensor with (1, c, h, w).
        rgb2bgr (bool): Whether to change rgb to bgr. Default: True.
        min_max (tuple[int]): min and max values for clamp.
    """
    output = tensor.squeeze(0).detach().clamp_(*min_max).permute(1, 2, 0)
    if uint16:
        output = (output - min_max[0]) / (min_max[1] - min_max[0]) * 65535
        output = output.type(torch.uint16).cpu().numpy()
    else:
        output = (output - min_max[0]) / (min_max[1] - min_max[0]) * 255
        output = output.type(torch.uint8).cpu().numpy()
    if rgb2bgr:
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output


def imfrombytes(content, flag='unchanged', float32=False, data_range=255.):
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.

    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {'color': cv2.IMREAD_COLOR, 'grayscale': cv2.IMREAD_GRAYSCALE, 'unchanged': cv2.IMREAD_UNCHANGED}
    img = cv2.imdecode(img_np, imread_flags[flag])
    if float32:
            img = img.astype(np.float32) / data_range
    return img


def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image to file.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    return cv2.imwrite(file_path, img, params)


def crop_border(imgs, crop_border):
    """Crop borders of images.

    Args:
        imgs (list[ndarray] | ndarray): Images with shape (h, w, c).
        crop_border (int): Crop border for each end of height and weight.

    Returns:
        list[ndarray]: Cropped images.
    """
    if crop_border == 0:
        return imgs
    else:
        if isinstance(imgs, list):
            return [v[crop_border:-crop_border, crop_border:-crop_border, ...] for v in imgs]
        else:
            return imgs[crop_border:-crop_border, crop_border:-crop_border, ...]


def normalization(input_data, black_level, white_level):
    output_data = (input_data.astype(float) - black_level) / (white_level - black_level)
    return output_data


def inv_normalization(input_data, black_level, white_level):
    output_data = np.clip(input_data, 0., 1.) * (white_level - black_level) + black_level
    output_data = output_data.astype(np.uint16)
    return output_data
