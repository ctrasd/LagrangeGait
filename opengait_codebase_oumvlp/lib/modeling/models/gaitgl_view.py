import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel
from ..modules import SeparateFCs, BasicConv3d, PackSequenceWrapper,CorrBlock


class GLConv(nn.Module):
    def __init__(self, in_channels, out_channels, halving, fm_sign=False, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, **kwargs):
        super(GLConv, self).__init__()
        self.halving = halving
        self.fm_sign = fm_sign
        self.global_conv3d = BasicConv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)
        self.local_conv3d = BasicConv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)

    def forward(self, x):
        '''
            x: [n, c, s, h, w]
        '''
        gob_feat = self.global_conv3d(x)
        if self.halving == 0:
            lcl_feat = self.local_conv3d(x)
        else:
            h = x.size(3)
            split_size = int(h // 2**self.halving)
            lcl_feat = x.split(split_size, 3)
            lcl_feat = torch.cat([self.local_conv3d(_) for _ in lcl_feat], 3)

        if not self.fm_sign:
            feat = F.leaky_relu(gob_feat) + F.leaky_relu(lcl_feat)
        else:
            feat = F.leaky_relu(torch.cat([gob_feat, lcl_feat], dim=3))
        return feat


class GeMHPP(nn.Module):
    def __init__(self, bin_num=[64], p=6.5, eps=1.0e-6):
        super(GeMHPP, self).__init__()
        self.bin_num = bin_num
        self.p = nn.Parameter(
            torch.ones(1)*p)
        self.eps = eps

    def gem(self, ipts):
        return F.avg_pool2d(ipts.clamp(min=self.eps).pow(self.p), (1, ipts.size(-1))).pow(1. / self.p)

    def forward(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p] 
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = self.gem(z).squeeze(-1)
            features.append(z)
        return torch.cat(features, -1)

class GeMHPP_em(nn.Module):
    def __init__(self, bin_num=[64], p=6.5, eps=1.0e-6,view_nums=14,dim=64):
        super(GeMHPP_em, self).__init__()
        self.bin_num = bin_num
        self.p = nn.Parameter(
            torch.ones(1)*p)
        self.eps = eps
        self.view_embedding_64 = nn.Parameter(torch.randn(view_nums,dim,1))
        #self.view_embedding_64_motion = nn.Parameter(torch.randn(self.view_nums,_set_channels[0],1))

    def gem(self, ipts):
        return F.avg_pool2d(ipts.clamp(min=self.eps).pow(self.p), (1, ipts.size(-1))).pow(1. / self.p)

    def forward(self, x,view):
        """
            x  : [n, c, h, w]
            ret: [n, c, p] 
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = self.gem(z).squeeze(-1)
            z = torch.cat((z,self.view_embedding_64[view].expand(-1,-1,b)),1)
            features.append(z)
        return torch.cat(features, -1)

class GaitGL_view(BaseModel):
    """
        GaitGL: Gait Recognition via Effective Global-Local Feature Representation and Local Temporal Aggregation
        Arxiv : https://arxiv.org/pdf/2011.01461.pdf
    """

    def __init__(self, *args, **kargs):
        super(GaitGL_view, self).__init__(*args, **kargs)

    def build_network(self, model_cfg):

        in_c = model_cfg['channels']
        class_num = model_cfg['class_num']
        dataset_name = self.cfgs['data_cfg']['dataset_name']
        self.view_nums= model_cfg['view_num']


        if dataset_name == 'OUMVLP':
            # For OUMVLP
            self.conv3d = nn.Sequential(
                BasicConv3d(1, in_c[0], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True),
                #BasicConv3d(in_c[0], in_c[0], kernel_size=(3, 3, 3),
                #            stride=(1, 1, 1), padding=(1, 1, 1)),
                #nn.LeakyReLU(inplace=True),
            )
            self.LTA = nn.Sequential(
                BasicConv3d(in_c[0], in_c[0], kernel_size=(
                    3, 1, 1), stride=(3, 1, 1), padding=(0, 0, 0)),
                nn.LeakyReLU(inplace=True)
            )

            self.GLConvA0 = nn.Sequential(
                GLConv(in_c[0], in_c[1], halving=1, fm_sign=False, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                GLConv(in_c[1], in_c[1], halving=1, fm_sign=False, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            )
            self.MaxPool0 = nn.MaxPool3d(
                kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.GLConvA1 = nn.Sequential(
                GLConv(in_c[1], in_c[2], halving=1, fm_sign=False, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                GLConv(in_c[2], in_c[2], halving=1, fm_sign=False, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            )
            self.GLConvB2 = nn.Sequential(
                GLConv(in_c[2], in_c[3], halving=1, fm_sign=False,  kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                GLConv(in_c[3], in_c[3], halving=1, fm_sign=True,  kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            )
            self.cls=nn.Linear(in_features=in_c[3], out_features=self.view_nums)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.radius=3
            self.HPP_mo = GeMHPP_em(bin_num=[4],view_nums=self.view_nums,dim=in_c[1])
            #self.HPP_mo=GeMHPP(bin_num=[4])
            
            self.motion_extract=CorrBlock(num_levels=1, radius=self.radius, input_dim=in_c[2])
            self.motion_conv1=BasicConv3d(1*(self.radius*2+1)**2,in_c[3],kernel_size=3)
            #self.motion_conv2=BasicConv3d(_set_channels[2], _set_channels[2],kernel_size=3)
            self.pool_motion=nn.AdaptiveAvgPool2d((1,1))
            
        else:
            # For CASIA-B or other unstated datasets.
            self.conv3d = nn.Sequential(
                BasicConv3d(1, in_c[0], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True)
            )
            self.LTA = nn.Sequential(
                BasicConv3d(in_c[0], in_c[0], kernel_size=(
                    3, 1, 1), stride=(3, 1, 1), padding=(0, 0, 0)),
                nn.LeakyReLU(inplace=True)
            )

            self.GLConvA0 = GLConv(in_c[0], in_c[1], halving=3, fm_sign=False, kernel_size=(
                3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            self.MaxPool0 = nn.MaxPool3d(
                kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.GLConvA1 = GLConv(in_c[1], in_c[2], halving=3, fm_sign=False, kernel_size=(
                3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            self.GLConvB2 = GLConv(in_c[2], in_c[2], halving=3, fm_sign=True,  kernel_size=(
                3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        self.Head0 = SeparateFCs(64+4, in_c[-1]+in_c[0], in_c[-1])
        self.Bn = nn.BatchNorm1d(in_c[-1])
        self.Head1 = SeparateFCs(64+4, in_c[-1], class_num)

        self.TP = PackSequenceWrapper(torch.max)
        #self.HPP = GeMHPP()
        self.HPP = GeMHPP_em(view_nums=self.view_nums,dim=in_c[0])

    def forward(self, inputs):
        ipts, labs, _, view, seqL = inputs
        view=[int(int(i)/15) if int(i)<=90 else int(int(i)-90+15)/15 for i in view]
        view=torch.tensor(view).long().cuda()

        seqL = None if not self.training else seqL
        if not self.training and len(labs) != 1:
            raise ValueError(
                'The input size of each GPU must be 1 in testing mode, but got {}!'.format(len(labs)))
        sils = ipts[0].unsqueeze(1)
        del ipts
        n, _, s, h, w = sils.size()
        if s < 3:
            repeat = 3 if s == 1 else 2
            sils = sils.repeat(1, 1, repeat, 1, 1)

        outs = self.conv3d(sils)
        outs = self.LTA(outs)

        outs = self.GLConvA0(outs)
        outs = self.MaxPool0(outs)

        outs = self.GLConvA1(outs)
        
        x2d_motion=outs.clone()
        b,c,t,h,w=x2d_motion.shape
        x2d_motion=x2d_motion.view(b*c,t,h,w)
        x2d_motion=F.avg_pool2d(x2d_motion, kernel_size=(2,2))
        x2d_motion=x2d_motion.view(b,c,t,h//2,w//2)
        x2d_motion=self.motion_extract(x2d_motion) # b (2*r+1)**2*2 t 16 11
        x2d_motion=self.motion_conv1(x2d_motion) 
        #x2d_motion=self.motion_conv2(x2d_motion) # b 256 t//3 16 11
        b,c,t,h,w=x2d_motion.shape
        x2d_motion=x2d_motion.permute(0,1,3,4,2).contiguous()
        x2d_motion=x2d_motion.view(b,c,h,w*t)
        

        outs = self.GLConvB2(outs)  # [n, c, s, h, w]

        outs = self.TP(outs, dim=2, seq_dim=2, seqL=seqL)[0]  # [n, c, h, w]

        x_feat=self.avgpool(outs)
        n=x_feat.shape[0]
        x_feat=x_feat.view(n,-1)
        angle_probe=self.cls(x_feat) # n 11
        _,angle= torch.max(angle_probe, 1)

        outs_mo=self.HPP_mo(x2d_motion,angle) # n c 4

        outs = self.HPP(outs,angle)  # [n, c, p]
        outs=torch.cat((outs_mo,outs),dim=2) # # n c p+8 

        outs = outs.permute(2, 0, 1).contiguous()  # [p, n, c]

        gait = self.Head0(outs)  # [p, n, c]
        gait = gait.permute(1, 2, 0).contiguous()  # [n, c, p]
        bnft = self.Bn(gait)  # [n, c, p]
        logi = self.Head1(bnft.permute(2, 0, 1).contiguous())  # [p, n, c]

        gait = gait.permute(0, 2, 1).contiguous()  # [n, p, c]
        bnft = bnft.permute(0, 2, 1).contiguous()  # [n, p, c]
        logi = logi.permute(1, 0, 2).contiguous()  # [n, p, c]

        n, _, s, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': bnft, 'labels': labs},
                'softmax': {'logits': logi, 'labels': labs},
                'view_softmax':{'logits':angle_probe.unsqueeze(1),'labels':view}
            },
            'visual_summary': {
                'image/sils': sils.view(n*s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': bnft
            }
        }
        return retval
