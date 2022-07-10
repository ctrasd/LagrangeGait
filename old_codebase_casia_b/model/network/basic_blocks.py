import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return F.leaky_relu(x, inplace=True)

class BasicConv3d(nn.Module):
    def __init__(self, inplanes, planes, dilation=1, bias=False, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 3, 3), bias=bias, dilation=(dilation, 1, 1), padding=(dilation, 1, 1))

    def forward(self, x):
        out = self.conv1(x)
        out = F.leaky_relu(out, inplace=True)
        return out



class Feed_Forward(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,dropout=0.5, **kwargs):
        super(Feed_Forward, self).__init__()
        self.conv = nn.Conv2d(in_channels, hidden_channels, (1,1), bias=False, **kwargs)
        self.relu=nn.LeakyReLU(inplace=True)
        self.drop=nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, (1,1), bias=False, **kwargs)
        self.upsample=nn.Conv2d(in_channels, out_channels, (1,1), bias=False, **kwargs)
        self.norm = nn.LayerNorm(out_channels)
    def forward(self, x):
        x_ori=x.clone()
        x=self.conv(x)
        x=self.relu(x)
        x=self.drop(x)
        x=self.conv2(x) # n c h w
        x=x.permute(0,2,3,1).contiguous()
        x=self.norm(x).permute(0,3,1,2).contiguous()

        return x+self.upsample(x_ori)

class Feed_Forward_ori(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Feed_Forward_cdim(nn.Module):
    def __init__(self, dim, hidden_dim,out_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)



class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=4):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = in_channels // reduc_ratio

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
                :param x: (b, c, h, w)
                :return x: (b, c, h, w)
        '''
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        #f_div_C = f / N
        f_div_C=F.softmax(f,-1)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z





class Attention_ori(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        # get q,k,v from a single weight matrix
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        #self.to_q=nn.Linear(dim, inner_dim , bias = False)
        #self.to_k=nn.Linear(dim, inner_dim , bias = False)
        #self.to_v=nn.Linear(dim, inner_dim , bias = False)
        #self.to_qkv.apply(weights_init_kaiming)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim)
        #self.to_out.apply(weights_init_kaiming)
    def forward(self, x, mask = None):
        #print(x.shape)
        x_ori=x.clone()
        #ipdb.set_trace()
        # x:[batch_size, patch_num, pathch_embedding_dim]
        b, n, _, h = *x.shape, self.heads
        #print(b, n, _, h)
        # get qkv tuple:([batch, patch_num, head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        
        # split q,k,v from [batch, patch_num, head_num*head_dim] -> [batch, head_num, patch_num, head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        #print(q.shape,v.shape)
        #transpose(k) * q / sqrt(head_dim) -> [batch, head_num, patch_num, patch_num]
        #dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        q=q.contiguous().view(b*h,n,-1)
        k=k.contiguous().view(b*h,-1,n).contiguous()
        dots=torch.bmm(q,k)
        dots=dots.view(b,h,n,n)
        
        #print(dots.shape)
        
        
        # mask value: -inf
        '''
        mask_value = -10000000000000

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask
        '''
        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)

        # value * attention matrix -> output
        
        attn=attn.view(b*h,n,-1)
        v=v.contiguous().view(b*h,n,-1).contiguous()
        out=torch.bmm(attn,v)
        out=out.view(b,h,n,-1)
        
        #out = torch.einsum('bhij,bhjd->bhid', attn, v)

        # cat all output -> [batch, patch_num, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        #out = out.view(b,h,n,d).permute(0,2,1,3).contiguous()
        #out=out.view(b,n,h*d)
        # Linear + Dropout
        out =  self.to_out(out)
        out =  self.norm(out)
        # out: [batch, patch_num, embedding_dim]
        return out+x_ori



class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        # get q,k,v from a single weight matrix
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        #self.to_q=nn.Linear(dim, inner_dim , bias = False)
        #self.to_k=nn.Linear(dim, inner_dim , bias = False)
        #self.to_v=nn.Linear(dim, inner_dim , bias = False)
        #self.to_qkv.apply(weights_init_kaiming)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim)
        #self.to_out.apply(weights_init_kaiming)
    def forward(self, x, mask = None):
        # bt c h w
        bt,c,he,wi=x.shape
        #print(x.shape)
        x_ori=x.clone()
        x=x.permute(0,2,3,1).contiguous()
        x=x.view(bt,he*wi,c)
        #ipdb.set_trace()
        # x:[batch_size, patch_num, pathch_embedding_dim]
        b, n, _, h = *x.shape, self.heads
        #print(b, n, _, h)
        # get qkv tuple:([batch, patch_num, head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        
        # split q,k,v from [batch, patch_num, head_num*head_dim] -> [batch, head_num, patch_num, head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        #print(q.shape,v.shape)
        #transpose(k) * q / sqrt(head_dim) -> [batch, head_num, patch_num, patch_num]
        #dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        q=q.contiguous().view(b*h,n,-1)
        k=k.contiguous().view(b*h,-1,n).contiguous()
        dots=torch.bmm(q,k)
        dots=dots.view(b,h,n,n)
        
        #print(dots.shape)
        
        
        # mask value: -inf
        '''
        mask_value = -10000000000000

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask
        '''
        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)

        # value * attention matrix -> output
        
        attn=attn.view(b*h,n,-1)
        v=v.contiguous().view(b*h,n,-1).contiguous()
        out=torch.bmm(attn,v)
        out=out.view(b,h,n,-1)
        
        #out = torch.einsum('bhij,bhjd->bhid', attn, v)

        # cat all output -> [batch, patch_num, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        #out = out.view(b,h,n,d).permute(0,2,1,3).contiguous()
        #out=out.view(b,n,h*d)
        # Linear + Dropout
        out =  self.to_out(out)
        out =  self.norm(out)
        out = out.view(bt,he,wi,-1).permute(0,3,1,2).contiguous()
        # out: [batch, patch_num, embedding_dim]
        return out+x_ori



class SetBlock(nn.Module):
    def __init__(self, forward_block, pooling=False):
        super(SetBlock, self).__init__()
        self.forward_block = forward_block
        self.pooling = pooling
        if pooling:
            self.pool2d = nn.MaxPool2d(2)
    def forward(self, x):
        n, s, c, h, w = x.size()
        x = self.forward_block(x.view(-1,c,h,w))
        if self.pooling:
            x = self.pool2d(x)
        _, c, h, w = x.size()
        return x.view(n, s, c, h ,w)



class SetBlock_feature(nn.Module):
    def __init__(self, forward_block, pooling=False):
        super(SetBlock_feature, self).__init__()
        self.forward_block = forward_block
        self.pooling = pooling

    def forward(self, x):
        n, s,h, c = x.size()
        x = self.forward_block(x.view(-1,h,c))

        _, h,c = x.size()
        return x.view(n, s,h, c)