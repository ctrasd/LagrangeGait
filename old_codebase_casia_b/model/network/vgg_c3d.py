import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from torch.nn.parameter import Parameter
from .motion_flow import CorrBlock
import torch.nn as nn

def gem(x, p=6.5, eps=1e-6):
    # print('x-',x.shape)
    # print('xpow-',x.clamp(min=eps).pow(p).shape)
    # print(F.avg_pool2d(x.clamp(min=eps).pow(p), (1, x.size(-1))).shape)
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (1, x.size(-1))).pow(1./p)

class GeM(nn.Module):

    def __init__(self, p=6.5, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        # print('p-',self.p)
        return gem(x, p=self.p, eps=self.eps)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

def gem1(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
class GeM_1(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM_1,self).__init__()
        #self.p = Parameter(torch.ones(1)*p)
        self.p=1
        self.eps = eps
    def forward(self, x):
        return gem1(x, p=self.p, eps=self.eps)       
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p) + ', ' + 'eps=' + str(self.eps) + ')'


    
class Temporal(nn.Module):
    def __init__(self, inplanes, planes, bias=False, **kwargs):
        super(Temporal, self).__init__()

    def forward(self, x):
        
        out = torch.max(x, 2)[0]
        return out


class BasicConv3d_p(nn.Module):
    def __init__(self, inplanes, planes, kernel=3, bias=False, p=2, FM=False, **kwargs):
        super(BasicConv3d_p, self).__init__()
        self.p = p
        self.fm = FM
        self.convdl = nn.Conv3d(inplanes, planes, kernel_size=(kernel, kernel, kernel), bias=bias, padding=((kernel-1)//2, (kernel-1)//2, (kernel-1)//2))
        self.convdg = nn.Conv3d(inplanes, planes, kernel_size=(kernel, kernel, kernel), bias=bias, padding=((kernel-1)//2, (kernel-1)//2, (kernel-1)//2))
    def forward(self, x):
        n, c, t, h, w = x.size()
        scale = h//self.p
        # print('p-',x.shape,n, c, t, h, w,'scale-',scale)
        feature = list()
        for i in range(self.p):
            temp = self.convdl(x[:,:,:,i*scale:(i+1)*scale,:])
            # print(temp.shape,i*scale,(i+1)*scale)
            feature.append(temp)

        outl = torch.cat(feature, 3)
        # print('outl-',outl.shape)
        outl = F.leaky_relu(outl, inplace=True)

        outg = self.convdg(x)
        outg = F.leaky_relu(outg, inplace=True)
        # print('outg-',outg.shape)
        if not self.fm:
            # print('1-1')
            out = outg + outl
        else:
            # print('1-2')
            out = torch.cat((outg, outl), dim=3)
        return out


class BasicConv3d(nn.Module):
    def __init__(self, inplanes, planes, dilation=1, bias=False, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 3, 3), bias=bias, dilation=(dilation, 1, 1), padding=(dilation, 1, 1))

    def forward(self, x):
        out = self.conv1(x)
        out = F.leaky_relu(out, inplace=True)
        return out

class LocaltemporalAG(nn.Module):
    def __init__(self, inplanes, planes, dilation=1, bias=False, **kwargs):
        super(LocaltemporalAG, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1), stride=(3,1,1), bias=bias,padding=(0, 0, 0))

    def forward(self, x):
        out1 = self.conv1(x)
        out = F.leaky_relu(out1, inplace=True)
        return out

class LocaltemporalAG33(nn.Module):
    def __init__(self, inplanes, planes, dilation=1, bias=False, **kwargs):
        super(LocaltemporalAG33, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 3, 3), stride=(3,1,1), bias=bias,padding=(0, 1, 1))

    def forward(self, x):
        out1 = self.conv1(x)
        out = F.leaky_relu(out1, inplace=True)
        return out


class C3D_VGG(nn.Module):

    def __init__(self, num_classes=74):
        super(C3D_VGG, self).__init__()
        _set_channels = [32, 64, 128, 256]

        # --------------------------------  2d gei---------------------------------------
        self.conv2dlayer1a = BasicConv3d(1, _set_channels[0], kernel=3)
        # self.conv2dlayer1b = BasicConv3d(_set_channels[0], _set_channels[0])
        self.pool2d1 = LocaltemporalAG(_set_channels[0], _set_channels[0])


        self.conv2dlayer2a = BasicConv3d_p(_set_channels[0], _set_channels[1])
        # self.conv2dlayer2b = BasicConv3d(_set_channels[1], _set_channels[1])
        self.pool2d2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2dlayer25a_3d = BasicConv3d_p(_set_channels[1], _set_channels[2])
        self.conv2dlayer25b_3d = BasicConv3d_p(_set_channels[2], _set_channels[2])

        self.conv2dlayer3a_3d = BasicConv3d_p(_set_channels[2], _set_channels[3])
        self.conv2dlayer3b_3d = BasicConv3d_p(_set_channels[3], _set_channels[3], FM=True)

        self.fpb3d = Temporal(_set_channels[3], _set_channels[3])

        self.Gem = GeM()


        self.bin_numgl = [32*2]
        self.fc_bin = nn.ParameterList([
            nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.zeros(sum(self.bin_numgl), _set_channels[3], _set_channels[3])))
                    ])
                



        self.relu = nn.ReLU()
        for m in self.modules():
            # print('---')
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        # print(x.shape)
        n, c, t, h, w = x.size()
        if t == 1:
            x = x.repeat(1, 1, 3, 1, 1)
        elif t == 2:
            x = x.repeat(1, 1, 2, 1, 1)
        elif t == 3:
            x = torch.cat((x,x[:,:,0:1,:,:]),dim=2)
        # print(x.shape)

        # ----------------2d--------------------
        x2d = self.conv2dlayer1a(x)
        # x2d = self.conv2dlayer1b(x2d)
        x2d = self.pool2d1(x2d)
        # print('pool2d1-',x2d.shape)
        x2d = self.conv2dlayer2a(x2d)
        # x2d = self.conv2dlayer2b(x2d)
        x2d = self.pool2d2(x2d)
        # print('pool2d2-',x2d.shape)

        x2d = self.conv2dlayer25a_3d(x2d)
        x2d = self.conv2dlayer25b_3d(x2d)

        x2da3d = self.conv2dlayer3a_3d(x2d)
        # print('conv2dlayer3a_3d-',x2da3d.shape)
        x2db3d = self.conv2dlayer3b_3d(x2da3d) # b t 256 64 22 
        # print('conv2dlayer3b_3d-',x2db3d.shape)



        x2db3d = self.fpb3d(x2db3d)
        print('x2db-',x2db3d.shape)

        # xgem = self.Gem(x2db3d)
        # print('xgem-',xgem.shape)

        _, c2d, _, _ = x2db3d.size()

        feature = list()
        for num_bin in self.bin_numgl:
            z = x2db3d.view(n, c2d, num_bin, -1).contiguous()
            # z1 = z.mean(3) + z.max(3)[0]
            # print('z1-',z1.shape)
            z2 = self.Gem(z).squeeze(-1)
            # print('z2-',z2.shape)
            feature.append(z2)
        feature = torch.cat(feature, 2).permute(2, 0, 1).contiguous()
        print('feature',feature.shape)
        feature = feature.matmul(self.fc_bin[0])
        feature = feature.permute(1, 2, 0).contiguous()
        print('feature',feature.shape)

        return feature,None

class C3D_VGG_angle(nn.Module):

    def __init__(self, num_classes=74):
        super(C3D_VGG_angle, self).__init__()
        _set_channels = [32, 64, 128, 256]

        # --------------------------------  2d gei---------------------------------------
        self.conv2dlayer1a = BasicConv3d(1, _set_channels[0], kernel=3)
        # self.conv2dlayer1b = BasicConv3d(_set_channels[0], _set_channels[0])
        self.pool2d1 = LocaltemporalAG(_set_channels[0], _set_channels[0])


        self.conv2dlayer2a = BasicConv3d_p(_set_channels[0], _set_channels[1])
        # self.conv2dlayer2b = BasicConv3d(_set_channels[1], _set_channels[1])
        self.pool2d2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2dlayer25a_3d = BasicConv3d_p(_set_channels[1], _set_channels[2])
        self.conv2dlayer25b_3d = BasicConv3d_p(_set_channels[2], _set_channels[2],FM=True)

        self.conv2dlayer3a_3d = BasicConv3d_p(_set_channels[2], _set_channels[3])
        self.conv2dlayer3b_3d = BasicConv3d_p(_set_channels[3], _set_channels[3], FM=True)

        self.fpb3d = Temporal(_set_channels[2], _set_channels[2])

        self.Gem = GeM()
        self.avgpool = GeM_1()
        self.cls=nn.Linear(in_features=_set_channels[2], out_features=11)
        
        self.trans_view=nn.ParameterList([
            nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.zeros(64, 128, 256)))]*11)
        self.bin_numgl = [32*2]
        self.fc_bin = nn.ParameterList([
            nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.zeros(sum(self.bin_numgl), _set_channels[3], _set_channels[3])))
                    ])
                



        self.relu = nn.ReLU()
        for m in self.modules():
            # print('---')
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        # print(x.shape)
        n, c, t, h, w = x.size()
        if t == 1:
            x = x.repeat(1, 1, 3, 1, 1)
        elif t == 2:
            x = x.repeat(1, 1, 2, 1, 1)
        elif t == 3:
            x = torch.cat((x,x[:,:,0:1,:,:]),dim=2)
        # print(x.shape)

        # ----------------2d--------------------
        x2d = self.conv2dlayer1a(x)
        # x2d = self.conv2dlayer1b(x2d)
        x2d = self.pool2d1(x2d)
        print('pool2d1-',x2d.shape)
        x2d = self.conv2dlayer2a(x2d)
        # x2d = self.conv2dlayer2b(x2d)
        x2d = self.pool2d2(x2d)
        print('pool2d2-',x2d.shape)

        x2d = self.conv2dlayer25a_3d(x2d)
        x2d = self.conv2dlayer25b_3d(x2d)
        print('3:',x2d.shape)
        #x2da3d = self.conv2dlayer3a_3d(x2d)
        # print('conv2dlayer3a_3d-',x2da3d.shape)
        #x2db3d = self.conv2dlayer3b_3d(x2da3d)
        # print('conv2dlayer3b_3d-',x2db3d.shape)



        x2db3d = self.fpb3d(x2d)
        print('4:',x2db3d.shape)
        n, c2d, _, _ = x2db3d.size()
        x_feat=self.avgpool(x2db3d)
        x_feat=x_feat.view(n,c2d)
        angle_probe=self.cls(x_feat) # n 11
        _,angle= torch.max(angle_probe, 1)
        
        # print('x2db-',x2db3d.shape)

        # xgem = self.Gem(x2db3d)
        # print('xgem-',xgem.shape)

        _, c2d, _, _ = x2db3d.size()

        feature = list()
        for num_bin in self.bin_numgl:
            z = x2db3d.view(n, c2d, num_bin, -1).contiguous()
            # z1 = z.mean(3) + z.max(3)[0]
            # print('z1-',z1.shape)
            z2 = self.Gem(z).squeeze(-1)
            # print('z2-',z2.shape)
            feature.append(z2)
        feature = torch.cat(feature, 2).permute(2, 0, 1).contiguous() #  64 n 256
        # print('feature',feature.shape)
        #feature = feature.matmul(self.fc_bin[0]) # 64 n 256
        #feature = feature.permute(1, 2, 0).contiguous() #n 256 64
        feature = feature.permute(1, 0, 2).contiguous()#n 64 256 
        
        feature_rt=[]
        for j in range(feature.shape[0]):
            #print(feature[j].shape)
            feature_now=((feature[j].unsqueeze(1)).bmm(self.trans_view[angle[j]])).squeeze(1) # 64*256
            feature_rt.append(feature_now)
        # print('feature',feature.shape)
        feature = torch.cat([x.unsqueeze(0) for x in feature_rt]) # n 64 256
        feature = feature.permute(0, 2, 1).contiguous()
        #print(feature.shape)
        return feature,None,angle_probe

def params_count(net):
    list1 = []
    for p in net.parameters():
        # print('p-',p.shape)
        list1.append(p)
    # print('len(net.parameters)',len(list1))
    n_parameters = sum(p.numel() for p in net.parameters())
    print('-----Model param: {:.5f}M'.format(n_parameters / 1e6))
    # print('-----Model memory: {:.5f}M'.format(n_parameters/1e6))
    return n_parameters


def c3d_vgg_Fusion(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = C3D_VGG(**kwargs)
    return model


class C3D_VGG_angle_emb(nn.Module):

    def __init__(self, num_classes=74):
        super(C3D_VGG_angle_emb, self).__init__()
        _set_channels = [32, 64, 128, 256]
        self.view_nums=11
        # --------------------------------  2d gei---------------------------------------
        self.conv2dlayer1a = BasicConv3d(1, _set_channels[0], kernel=3)
        # self.conv2dlayer1b = BasicConv3d(_set_channels[0], _set_channels[0])
        self.pool2d1 = LocaltemporalAG(_set_channels[0], _set_channels[0])


        self.conv2dlayer2a = BasicConv3d_p(_set_channels[0], _set_channels[1])
        # self.conv2dlayer2b = BasicConv3d(_set_channels[1], _set_channels[1])
        self.pool2d2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2dlayer25a_3d = BasicConv3d_p(_set_channels[1], _set_channels[2])
        self.conv2dlayer25b_3d = BasicConv3d_p(_set_channels[2], _set_channels[2],FM=True)

        self.conv2dlayer3a_3d = BasicConv3d_p(_set_channels[2], _set_channels[3])
        self.conv2dlayer3b_3d = BasicConv3d_p(_set_channels[3], _set_channels[3], FM=True)

        self.fpb3d = Temporal(_set_channels[2], _set_channels[2])

        self.Gem = GeM()
        self.avgpool = GeM_1()
        self.cls=nn.Linear(in_features=_set_channels[2], out_features=11)
        '''
        self.trans_view=nn.ParameterList([
            nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.zeros(64, 128, 256)))]*11)
        '''



        self.view_embedding_64 = nn.Parameter(torch.randn(self.view_nums,_set_channels[1],1))

        self.bin_numgl = [32*2]
        self.fc_bin = nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.zeros(sum(self.bin_numgl), _set_channels[2]+_set_channels[1], _set_channels[3])))
                    
                



        self.relu = nn.ReLU()
        for m in self.modules():
            # print('---')
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        # print(x.shape)
        x=x.permute(0,2,1,3,4).contiguous()
        n, c, t, h, w = x.size()
        if t == 1:
            x = x.repeat(1, 1, 3, 1, 1)
        elif t == 2:
            x = x.repeat(1, 1, 2, 1, 1)
        elif t == 3:
            x = torch.cat((x,x[:,:,0:1,:,:]),dim=2)
        # print(x.shape)

        # ----------------2d--------------------
        x2d = self.conv2dlayer1a(x)
        # x2d = self.conv2dlayer1b(x2d)
        x2d = self.pool2d1(x2d)
        #print('pool2d1-',x2d.shape)
        x2d = self.conv2dlayer2a(x2d)
        # x2d = self.conv2dlayer2b(x2d)
        x2d = self.pool2d2(x2d)
        #print('pool2d2-',x2d.shape)

        x2d = self.conv2dlayer25a_3d(x2d)
        x2d = self.conv2dlayer25b_3d(x2d) # b 256 t 64 22


        #print('3:',x2d.shape)
        #x2da3d = self.conv2dlayer3a_3d(x2d)
        # print('conv2dlayer3a_3d-',x2da3d.shape)
        #x2db3d = self.conv2dlayer3b_3d(x2da3d)
        # print('conv2dlayer3b_3d-',x2db3d.shape)



        x2db3d = self.fpb3d(x2d)
        #print('4:',x2db3d.shape)
        n, c2d, _, _ = x2db3d.size()
        x_feat=self.avgpool(x2db3d)
        x_feat=x_feat.view(n,c2d)
        angle_probe=self.cls(x_feat) # n 11
        _,angle= torch.max(angle_probe, 1)
        
        #print('x2db-',x2db3d.shape)

        # xgem = self.Gem(x2db3d)
        # print('xgem-',xgem.shape)

        _, c2d, _, _ = x2db3d.size()

        feature = list()
        for num_bin in self.bin_numgl:
            z = x2db3d.view(n, c2d, num_bin, -1).contiguous()
            # z1 = z.mean(3) + z.max(3)[0]
            # print('z1-',z1.shape)
            #z2 = self.Gem(z).squeeze(-1)
            z2=torch.cat((self.Gem(z).squeeze(-1),self.view_embedding_64[angle].expand(-1,-1,num_bin)),1)
            # print('z2-',z2.shape)
            feature.append(z2)
        feature = torch.cat(feature, 2).permute(2, 0, 1).contiguous() #  64 n 256
        #print('feature',feature.shape)
        feature = feature.matmul(self.fc_bin) # 64 n 256
        feature = feature.permute(1, 2, 0).contiguous()
        #feature = feature.permute(1, 2, 0).contiguous() #n 256 64
        #feature = feature.permute(1, 0, 2).contiguous()#n 64 256 
        '''
        feature_rt=[]
        for j in range(feature.shape[0]):
            #print(feature[j].shape)
            feature_now=((feature[j].unsqueeze(1)).bmm(self.trans_view[angle[j]])).squeeze(1) # 64*256
            feature_rt.append(feature_now)
        # print('feature',feature.shape)
        feature = torch.cat([x.unsqueeze(0) for x in feature_rt]) # n 64 256
        feature = feature.permute(0, 2, 1).contiguous()
        #print(feature.shape)
        '''
        return feature,angle_probe


class C3D_VGG_angle_emb_motion(nn.Module):

    def __init__(self, num_classes=74):
        super(C3D_VGG_angle_emb_motion, self).__init__()
        _set_channels = [32, 64, 128, 256]
        self.view_nums=11
        self.radius=3
        # --------------------------------  2d gei---------------------------------------
        self.conv2dlayer1a = BasicConv3d(1, _set_channels[0], kernel=3)
        # self.conv2dlayer1b = BasicConv3d(_set_channels[0], _set_channels[0])
        self.pool2d1 = LocaltemporalAG(_set_channels[0], _set_channels[0])


        self.conv2dlayer2a = BasicConv3d_p(_set_channels[0], _set_channels[1])
        # self.conv2dlayer2b = BasicConv3d(_set_channels[1], _set_channels[1])
        self.pool2d2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2dlayer25a_3d = BasicConv3d_p(_set_channels[1], _set_channels[2])
        self.conv2dlayer25b_3d = BasicConv3d_p(_set_channels[2], _set_channels[2],FM=True)

        #self.conv2dlayer3a_3d = BasicConv3d_p(_set_channels[2], _set_channels[3])
        #self.conv2dlayer3b_3d = BasicConv3d_p(_set_channels[3], _set_channels[3], FM=True)

        self.fpb3d = Temporal(_set_channels[2], _set_channels[2])

        self.Gem = GeM()
        self.avgpool = GeM_1()
        self.cls=nn.Linear(in_features=_set_channels[2], out_features=11)
        '''
        self.trans_view=nn.ParameterList([
            nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.zeros(64, 128, 256)))]*11)
        '''
        self.view_embedding_64 = nn.Parameter(torch.randn(self.view_nums,_set_channels[0],1))

        self.bin_numgl = [32*2]
        
        self.fc_bin = nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.zeros(sum(self.bin_numgl), _set_channels[2]+_set_channels[0], _set_channels[3])))
        '''
        self.fc_bin = nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.zeros(sum(self.bin_numgl), _set_channels[2], _set_channels[3])))
        '''
        
        self.view_embedding_64_motion = nn.Parameter(torch.randn(self.view_nums,_set_channels[0],1))
        self.view_embedding_64_motion2 = nn.Parameter(torch.randn(self.view_nums,_set_channels[0],1))

        self.bin_numgl_motion = [4]
        
        self.fc_bin_motion = nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.zeros(sum(self.bin_numgl_motion), _set_channels[2]+_set_channels[0], _set_channels[3])))
        '''
        self.fc_bin_motion2 = nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.zeros(sum(self.bin_numgl_motion), _set_channels[2]+_set_channels[0], _set_channels[3])))
        '''
        '''         
        self.fc_bin_motion = nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.zeros(sum(self.bin_numgl_motion), _set_channels[2], _set_channels[3])))
        '''

        self.motion_extract=CorrBlock(num_levels=1, radius=self.radius, input_dim=_set_channels[0])
        self.motion_conv1=BasicConv3d(1*(self.radius*2+1)**2,_set_channels[1],kernel_size=3)
        self.motion_conv2=BasicConv3d(_set_channels[1], _set_channels[2],kernel_size=3)
        self.pool_motion=nn.AdaptiveAvgPool2d((1,1))
        '''
        self.motion_extract2=CorrBlock(num_levels=1, radius=self.radius, input_dim=_set_channels[2])
        self.motion_conv12=BasicConv3d(1*(self.radius*2+1)**2,_set_channels[2],kernel_size=3)
        #self.motion_conv2=BasicConv3d(_set_channels[2], _set_channels[2],kernel_size=3)
        self.pool_motion2=nn.AdaptiveAvgPool2d((1,1))
        '''

        self.relu = nn.ReLU()
        for m in self.modules():
            # print('---')
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        # print(x.shape)
        x=x.permute(0,2,1,3,4).contiguous()
        n, c, t, h, w = x.size()
        if t == 1:
            x = x.repeat(1, 1, 6, 1, 1)
        elif t == 2:
            x = x.repeat(1, 1, 3, 1, 1)
        elif t <= 5:
            x = torch.cat((x,x[:,:,0:3,:,:]),dim=2)
        #print(x.shape)

        # ----------------2d--------------------
        x2d = self.conv2dlayer1a(x)
        # x2d = self.conv2dlayer1b(x2d)
        x2d = self.pool2d1(x2d)

        x2d_motion=x2d.clone()
        x2d_motion=self.pool2d2(x2d_motion)

        #print('pool2d1-',x2d.shape)
        x2d = self.conv2dlayer2a(x2d)
        # x2d = self.conv2dlayer2b(x2d)
        x2d = self.pool2d2(x2d)
        #print('pool2d2-',x2d.shape)


        

        b,c,t,h,w=x2d_motion.shape
        x2d_motion=x2d_motion.view(b*c,t,h,w)
        x2d_motion=F.avg_pool2d(x2d_motion, kernel_size=(2,2))
        x2d_motion=x2d_motion.view(b,c,t,h//2,w//2)
        x2d_motion=self.motion_extract(x2d_motion) # b (2*r+1)**2*2 t 16 11
        x2d_motion=self.motion_conv1(x2d_motion) 
        x2d_motion=self.motion_conv2(x2d_motion) # b 256 t//3 16 11
        b,c,t,h,w=x2d_motion.shape
        x2d_motion=x2d_motion.mean(2)
        #x2d_motion = torch.max(x2d_motion, 2)[0]
        #x2d_motion=x2d_motion.permute(0,1,3,4,2).contiguous()
        #x2d_motion=x2d_motion.view(b,c,h,w*t)

        x2d = self.conv2dlayer25a_3d(x2d)
        x2d = self.conv2dlayer25b_3d(x2d) # b t 256 64 22
        '''
        x2d_motion2=x2d.clone()

        b,c,t,h,w=x2d_motion2.shape
        x2d_motion2=x2d_motion2.view(b*c,t,h,w)
        x2d_motion2=F.avg_pool2d(x2d_motion2, kernel_size=(2,2))
        x2d_motion2=x2d_motion2.view(b,c,t,h//2,w//2)
        x2d_motion2=self.motion_extract2(x2d_motion2) # b (2*r+1)**2*2 t 16 11
        x2d_motion2=self.motion_conv12(x2d_motion2) 
        #x2d_motion=self.motion_conv2(x2d_motion) # b 256 t//3 16 11
        b,c,t,h,w=x2d_motion2.shape
        x2d_motion2=x2d_motion2.permute(0,1,3,4,2).contiguous()
        x2d_motion2=x2d_motion2.view(b,c,h,w*t)
        '''


        x2db3d = self.fpb3d(x2d)
        #print('4:',x2db3d.shape)
        n, c2d, _, _ = x2db3d.size()
        x_feat=self.avgpool(x2db3d)
        x_feat=x_feat.view(n,c2d)
        angle_probe=self.cls(x_feat) # n 11
        _,angle= torch.max(angle_probe, 1)
        
        
        
        n,c2d,hh,ww=x2d_motion.shape
        feature_motion=[]
        for num_bin in self.bin_numgl_motion:
            z = x2d_motion.view(n, c2d, num_bin, hh//num_bin,ww).contiguous()
            #z = x2d_motion.view(n, c2d, num_bin, -1).contiguous()
            # z1 = z.mean(3) + z.max(3)[0]
            # print('z1-',z1.shape)
            #z2 = self.Gem(z).squeeze(-1)
            z=self.pool_motion(z)
            #print(z.shape)
            #z2=z.view(n, c2d, num_bin)
            z2=torch.cat([z.view(n, c2d, num_bin),self.view_embedding_64_motion[angle].expand(-1,-1,num_bin)],1)
            # print('z2-',z2.shape)
            feature_motion.append(z2)
        feature_motion = torch.cat(feature_motion, 2).permute(2, 0, 1).contiguous() #  8 n 256
        #print('feature',feature.shape)
        feature_motion = feature_motion.matmul(self.fc_bin_motion) # 8 n 256
        feature_motion = feature_motion.permute(1, 2, 0).contiguous()
        '''
        n,c2d,hh,ww=x2d_motion2.shape
        feature_motion2=[]
        for num_bin in self.bin_numgl_motion:
            z = x2d_motion2.view(n, c2d, num_bin, hh//num_bin,ww).contiguous()
            #z = x2d_motion.view(n, c2d, num_bin, -1).contiguous()
            # z1 = z.mean(3) + z.max(3)[0]
            # print('z1-',z1.shape)
            #z2 = self.Gem(z).squeeze(-1)
            z=self.pool_motion(z)
            #print(z.shape)
            #z2=z.view(n, c2d, num_bin)
            z2=torch.cat([z.view(n, c2d, num_bin),self.view_embedding_64_motion2[angle].expand(-1,-1,num_bin)],1)
            # print('z2-',z2.shape)
            feature_motion2.append(z2)
        feature_motion2 = torch.cat(feature_motion2, 2).permute(2, 0, 1).contiguous() #  8 n 256
        #print('feature',feature.shape)
        feature_motion2 = feature_motion2.matmul(self.fc_bin_motion2) # 8 n 256
        feature_motion2 = feature_motion2.permute(1, 2, 0).contiguous()

        '''
        #print('3:',x2d.shape)
        #x2da3d = self.conv2dlayer3a_3d(x2d)
        # print('conv2dlayer3a_3d-',x2da3d.shape)
        #x2db3d = self.conv2dlayer3b_3d(x2da3d)
        # print('conv2dlayer3b_3d-',x2db3d.shape)
        
        #print('x2db-',x2db3d.shape)

        # xgem = self.Gem(x2db3d)
        # print('xgem-',xgem.shape)

        _, c2d, _, _ = x2db3d.size()

        feature = list()
        for num_bin in self.bin_numgl:
            z = x2db3d.view(n, c2d, num_bin, -1).contiguous()
            # z1 = z.mean(3) + z.max(3)[0]
            # print('z1-',z1.shape)
            #z2 = self.Gem(z).squeeze(-1)
            #z2=self.Gem(z).squeeze(-1)
            z2=torch.cat((self.Gem(z).squeeze(-1),self.view_embedding_64[angle].expand(-1,-1,num_bin)),1)
            # print('z2-',z2.shape)
            feature.append(z2)
        feature = torch.cat(feature, 2).permute(2, 0, 1).contiguous() #  64 n 256
        #print('feature',feature.shape)
        feature = feature.matmul(self.fc_bin) # 64 n 256
        feature = feature.permute(1, 2, 0).contiguous()
        feature=torch.cat([feature_motion,feature],2)
        #feature = feature.permute(1, 2, 0).contiguous() #n 256 64
        #feature = feature.permute(1, 0, 2).contiguous()#n 64 256 
        '''
        feature_rt=[]
        for j in range(feature.shape[0]):
            #print(feature[j].shape)
            feature_now=((feature[j].unsqueeze(1)).bmm(self.trans_view[angle[j]])).squeeze(1) # 64*256
            feature_rt.append(feature_now)
        # print('feature',feature.shape)
        feature = torch.cat([x.unsqueeze(0) for x in feature_rt]) # n 64 256
        feature = feature.permute(0, 2, 1).contiguous()
        #print(feature.shape)
        '''
        return feature,angle_probe






if __name__ == "__main__":
    net = c3d_vgg_Fusion(num_classes=74)
    print(params_count(net))
    with torch.no_grad():
        # x = torch.ones(4*3*16*64*44).reshape(4,3,16,64,44)
        x = torch.ones(4 * 1 * 32 * 64 * 44).reshape(4, 1, 32, 64, 44)
        # a = Variable(a.cuda)
        print('x=', x.shape)
        # a,b = net(x)
        # print('a,b=',a.shape,b.shape)
        a,_ = net(x)
        print('a,b=', a.shape)