import numpy as np
import torch


def rgb2yuv(rgb, out_range=1, method=1):
    """input: [0,255]
    """
    if isinstance(rgb, np.ndarray): rgb = rgb.copy()
    else: rgb = rgb.clone()

    # convert to [0, 255]
    if rgb.max()<=1: rgb = 255*rgb

    if isinstance(rgb, np.ndarray): yuv = rgb.copy()
    else: yuv = rgb.clone()

    if method==0:
        yuv[:,0] = 0.257*rgb[:,0] + 0.504*rgb[:,1] + 0.098*rgb[:,2] + 16
        yuv[:,1] = -0.148*rgb[:,0] - 0.291*rgb[:,1] + 0.439*rgb[:,2] + 128
        yuv[:,2] = 0.439*rgb[:,0] - 0.368*rgb[:,1] - 0.071*rgb[:,2] + 128
        yuv[:,0] = (yuv[:,0]-16)/(235-16)
        yuv[:,1] = (yuv[:,1]-16)/(240-16)
        yuv[:,2] = (yuv[:,2]-16)/(240-16)

    if method==1:
        yuv[:,0] = 0.212600 * rgb[:,0]  + 0.715200 * rgb[:,1] + 0.072200 * rgb[:,2]
        yuv[:,1] = -0.114572 * rgb[:,0] - 0.385428 * rgb[:,1] + 0.5 * rgb[:,2]      + 128.0
        yuv[:,2] = 0.5 * rgb[:,0]       - 0.454153 * rgb[:,1] - 0.045847 * rgb[:,2] + 128.0
        yuv = yuv / 255.

    yuv = yuv * out_range

    if out_range > 1: yuv = yuv.round()
    
    return yuv


def yuv2rgb(yuv, out_range=1, method=1):
    """input: [0,1];    output: [0,1]
    """
    if isinstance(yuv, np.ndarray): yuv = yuv.copy()
    else: yuv = yuv.clone()

    if yuv.max()>1: yuv = yuv / 255. # convert to [0, 1]

    if method==0:
        yuv[:,0] = (235-16)*yuv[:,0]+16
        yuv[:,1] = (240-16)*yuv[:,1]+16
        yuv[:,2] = (240-16)*yuv[:,2]+16
        
        if isinstance(yuv, np.ndarray): rgb = yuv.copy()
        else: rgb = yuv.clone()
        
        rgb[:,0] = 1.164*(yuv[:,0]-16) + 1.596*(yuv[:,2]-128)
        rgb[:,1] = 1.164*(yuv[:,0]-16) - 0.813*(yuv[:,2]-128) - 0.392*(yuv[:,1]-128)
        rgb[:,2] = 1.164*(yuv[:,0]-16) + 2.017*(yuv[:,1]-128)

    if method==1:
        yuv = yuv * 255.
        
        yuv[:,0] = yuv[:,0]
        yuv[:,1] = yuv[:,1]-128
        yuv[:,2] = yuv[:,2]-128

        if isinstance(yuv, np.ndarray): rgb = yuv.copy()
        else: rgb = yuv.clone()

        rgb[:,0] = yuv[:,0]                         + 1.57480 * yuv[:,2]
        rgb[:,1] = yuv[:,0] - 0.18733 * yuv[:,1]    - 0.46813 * yuv[:,2]
        rgb[:,2] = yuv[:,0] + 1.85563 * yuv[:,1]


    rgb = rgb/255
    rgb = rgb * out_range

    return rgb


def rgb2YCoCg(rgb):
    """input shoud be integer
    """
    if isinstance(rgb, np.ndarray): rgb = rgb.copy()
    else: rgb = rgb.clone()

    if isinstance(rgb, np.ndarray):
        rgb = rgb.astype('float32')
        assert rgb.max()>1
        R, G, B = rgb[:,0], rgb[:,1], rgb[:,2] 
        Co = R - B
        t = B + np.floor(Co/2)
        Cg = G - t
        Y = t + np.floor(Cg/2)
        YCoCg = np.stack([Y,Co,Cg], axis=-1)
    else:
        rgb = rgb.float()
        assert rgb.max()>1
        R, G, B = rgb[:,0], rgb[:,1], rgb[:,2] 
        Co = R - B
        t = B + torch.floor(Co/2)
        Cg = G - t
        Y = t + torch.floor(Cg/2)
        YCoCg = torch.stack([Y,Co,Cg], axis=-1)

    return YCoCg


def YCoCg2rgb(YCoCg):
    if isinstance(YCoCg, np.ndarray): YCoCg = YCoCg.copy()
    else: YCoCg = YCoCg.clone()

    Y, Co, Cg = YCoCg[:,0], YCoCg[:,1], YCoCg[:,2]

    if isinstance(YCoCg, np.ndarray):
        t = Y - np.floor(Cg/2)
        G = Cg + t
        B = t - np.floor(Co/2)
        R = Co + B
        RGB = np.stack([R,G,B], axis=-1)
    else:
        t = Y - torch.floor(Cg/2)
        G = Cg + t
        B = t - torch.floor(Co/2)
        R = Co + B
        RGB = torch.stack([R,G,B], dim=-1)

    return RGB