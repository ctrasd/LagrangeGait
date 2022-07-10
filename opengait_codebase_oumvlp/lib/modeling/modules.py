import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import clones, is_list_or_tuple


class HorizontalPoolingPyramid():
    """
        Horizontal Pyramid Matching for Person Re-identification
        Arxiv: https://arxiv.org/abs/1804.05275
        Github: https://github.com/SHI-Labs/Horizontal-Pyramid-Matching
    """

    def __init__(self, bin_num=None):
        if bin_num is None:
            bin_num = [16, 8, 4, 2, 1]
        self.bin_num = bin_num

    def __call__(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p] 
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = z.mean(-1) + z.max(-1)[0]
            features.append(z)
        return torch.cat(features, -1)


class SetBlockWrapper(nn.Module):
    def __init__(self, forward_block):
        super(SetBlockWrapper, self).__init__()
        self.forward_block = forward_block

    def forward(self, x, *args, **kwargs):
        """
            In  x: [n, s, c, h, w]
            Out x: [n, s, ...]
        """
        n, s, c, h, w = x.size()
        x = self.forward_block(x.view(-1, c, h, w), *args, **kwargs)
        _ = x.size()
        _ = [n, s] + [*_[1:]]
        return x.view(*_)


class PackSequenceWrapper(nn.Module):
    def __init__(self, pooling_func):
        super(PackSequenceWrapper, self).__init__()
        self.pooling_func = pooling_func

    def forward(self, seqs, seqL, seq_dim=1, **kwargs):
        """
            In  seqs: [n, s, ...]
            Out rets: [n, ...]
        """
        if seqL is None:
            return self.pooling_func(seqs, **kwargs)
        seqL = seqL[0].data.cpu().numpy().tolist()
        start = [0] + np.cumsum(seqL).tolist()[:-1]

        rets = []
        for curr_start, curr_seqL in zip(start, seqL):
            narrowed_seq = seqs.narrow(seq_dim, curr_start, curr_seqL)
            # save the memory
            # splited_narrowed_seq = torch.split(narrowed_seq, 256, dim=1)
            # ret = []
            # for seq_to_pooling in splited_narrowed_seq:
            #     ret.append(self.pooling_func(seq_to_pooling, keepdim=True, **kwargs)
            #                [0] if self.is_tuple_result else self.pooling_func(seq_to_pooling, **kwargs))
            rets.append(self.pooling_func(narrowed_seq, **kwargs))
        if len(rets) > 0 and is_list_or_tuple(rets[0]):
            return [torch.cat([ret[j] for ret in rets])
                    for j in range(len(rets[0]))]
        return torch.cat(rets)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return x


class SeparateFCs(nn.Module):
    def __init__(self, parts_num, in_channels, out_channels, norm=False):
        super(SeparateFCs, self).__init__()
        self.p = parts_num
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, out_channels)))
        self.norm = norm

    def forward(self, x):
        """
            x: [p, n, c]
        """
        if self.norm:
            out = x.matmul(F.normalize(self.fc_bin, dim=1))
        else:
            out = x.matmul(self.fc_bin)
        return out


class SeparateBNNecks(nn.Module):
    """
        GaitSet: Bag of Tricks and a Strong Baseline for Deep Person Re-Identification
        CVPR Workshop:  https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf
        Github: https://github.com/michuanhaohao/reid-strong-baseline
    """

    def __init__(self, parts_num, in_channels, class_num, norm=True, parallel_BN1d=True):
        super(SeparateBNNecks, self).__init__()
        self.p = parts_num
        self.class_num = class_num
        self.norm = norm
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, class_num)))
        if parallel_BN1d:
            self.bn1d = nn.BatchNorm1d(in_channels * parts_num)
        else:
            self.bn1d = clones(nn.BatchNorm1d(in_channels), parts_num)
        self.parallel_BN1d = parallel_BN1d

    def forward(self, x):
        """
            x: [p, n, c]
        """
        if self.parallel_BN1d:
            p, n, c = x.size()
            x = x.transpose(0, 1).contiguous().view(n, -1)  # [n, p*c]
            x = self.bn1d(x)
            x = x.view(n, p, c).permute(1, 0, 2).contiguous()
        else:
            x = torch.cat([bn(_.squeeze(0)).unsqueeze(0)
                           for _, bn in zip(x.split(1, 0), self.bn1d)], 0)  # [p, n, c]
        if self.norm:
            feature = F.normalize(x, dim=-1)  # [p, n, c]
            logits = feature.matmul(F.normalize(
                self.fc_bin, dim=1))  # [p, n, c]
        else:
            feature = x
            logits = feature.matmul(self.fc_bin)
        return feature, logits


class FocalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, halving, **kwargs):
        super(FocalConv2d, self).__init__()
        self.halving = halving
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, bias=False, **kwargs)

    def forward(self, x):
        if self.halving == 0:
            z = self.conv(x)
        else:
            h = x.size(2)
            split_size = int(h // 2**self.halving)
            z = x.split(split_size, 2)
            z = torch.cat([self.conv(_) for _ in z], 2)
        return z


class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=bias, **kwargs)

    def forward(self, ipts):
        '''
            ipts: [n, c, s, h, w]
            outs: [n, c, s, h, w]
        '''
        outs = self.conv3d(ipts)
        return outs


def RmBN2dAffine(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.requires_grad = False
            m.bias.requires_grad = False

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1).contiguous()
    img = F.grid_sample(img.contiguous(), grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)
def cal_corr(f0,f1):
    b,c,h,w=f0.shape
    f0=f0.view(b,c,h*w).permute(0,2,1).contiguous()
    f1=f1.view(b,c,h*w).contiguous()
    #print(f0.shape,f1.shape)
    mp=torch.bmm(f0,f1)/c
    mp=mp.view(b,h*w,h,w).contiguous()
    return mp

class CorrBlock(nn.Module):
    def __init__(self,num_levels=1, radius=3,input_dim=64):
        super(CorrBlock, self).__init__()
        self.num_levels = num_levels
        self.radius = radius
        self.fc1=nn.Conv2d(input_dim,input_dim//4,kernel_size=1,bias=False)#nn.Linear(input_dim,input_dim//4,bias=False)
        self.fc0=nn.Conv2d(input_dim,input_dim//4,kernel_size=1,bias=False)
    def forward(self,x):
        self.corr_pyramid = []
        device=x.device
        #print('x:de:',x.device)
        r=self.radius
        x=x.permute(0,2,1,3,4).contiguous()
        # x:b,t,c,h,w
        f0=x[:,:-1].clone()
        f1=x[:,1:].clone()
        b,t,c,h,w=f0.shape
        f0=f0.view(b*t,c,h,w)
        f1=f1.view(b*t,c,h,w)
        f0=self.fc0(f0)
        f1=self.fc1(f1)
        f0=f0.view(b,t,c//4,h,w)
        f1=f1.view(b,t,c//4,h,w)

        b,t,c,h,w=f0.shape # 16 11

        f0=f0.view(b*t,c,h,w)
        f1=f1.view(b*t,c,h,w)
        
        b,c,h,w=f0.shape
        
        
        coords = coords_grid(b, h, w).to(device)
        coords = coords.permute(0, 2, 3, 1).contiguous()
        
        
        corr=cal_corr(f0,f1)
        #print('corr:de:',corr.device)
        #print('coords:de:',coords.device)
        corr=corr.reshape(b*h*w, 1, h, w)
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)
        out_pyramid = []
        for i in range(self.num_levels):
            
            corr=self.corr_pyramid[i]

            #print(coords.shape)
            dx = torch.linspace(-r, r, 2*r+1)
            dy = torch.linspace(-r, r, 2*r+1)
            #print('dx,dy.shape',dx.shape)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(device)
            #print('delata:',delta.device)
            #print(delta)
            #print(coords.shape)
            #print(b,h,w)
            centroid_lvl = coords.reshape(b*h*w, 1, 1, 2) / (2**i)
            #centroid_lvl = coords.view(b*h*w, 1, 1, 2) / (2**i)
            #print(centroid_lvl)
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            #print(delta_lvl)
            coords_lvl = centroid_lvl + delta_lvl
            #print(centroid_lvl.shape,delta_lvl.shape)
            #print(coords_lvl.shape)
            #print('de pair:',corr.device,coords_lvl.device)
            #print(corr.shape,coords_lvl.shape)
            #print(type(corr),type(coords_lvl))
            #print(corr)
            #print(coords_lvl)
            corr = bilinear_sampler(corr, coords_lvl)
            #print('corr:',corr.shape)
            corr = corr.view(b, h, w, -1)
            out_pyramid.append(corr)
            
        out_fea=torch.cat(out_pyramid,3)
        b,h,w,c=out_fea.shape
        out_fea=out_fea.view(b//t,t,h,w,c)
        return out_fea.permute(0, 4, 1, 2, 3).contiguous().float() # b c t h w

