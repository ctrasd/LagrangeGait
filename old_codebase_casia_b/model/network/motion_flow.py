import torch
import torch.nn.functional as F
import torch.nn as nn
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
        self.fc1=nn.Conv2d(input_dim,input_dim//2,kernel_size=1,bias=False)#nn.Linear(input_dim,input_dim//4,bias=False)
        self.fc0=nn.Conv2d(input_dim,input_dim//2,kernel_size=1,bias=False)
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
        f0=f0.view(b,t,c//2,h,w)
        f1=f1.view(b,t,c//2,h,w)

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
            corr = bilinear_sampler(corr, coords_lvl)
            #print('corr:',corr.shape)
            corr = corr.view(b, h, w, -1)
            out_pyramid.append(corr)
            
        out_fea=torch.cat(out_pyramid,3)
        b,h,w,c=out_fea.shape
        out_fea=out_fea.view(b//t,t,h,w,c)
        return out_fea.permute(0, 4, 1, 2, 3).contiguous().float() # b c t h w