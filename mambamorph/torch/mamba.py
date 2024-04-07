import torch
from torch import nn
from mamba_ssm import Mamba
import torch.nn.functional as nnf
import pdb

from einops import rearrange, repeat
from vmamba import SS2D
from functools import partial
from typing import Optional, Callable
from vmamba import VSSBlock



class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, reduce_factor=2):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, (8 // reduce_factor) * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x, H, W, T):
        """
        x: B, H*W*T, C
        """
        B, L, C = x.shape
        assert L == H * W * T, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0 and T % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, T, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1) or (T % 2 == 1)
        if pad_input:
            x = nnf.pad(x, (0, 0, 0, T % 2, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, 0::2, :]  # B H/2 W/2 T/2 C
        x1 = x[:, 1::2, 0::2, 0::2, :]  # B H/2 W/2 T/2 C
        x2 = x[:, 0::2, 1::2, 0::2, :]  # B H/2 W/2 T/2 C
        x3 = x[:, 0::2, 0::2, 1::2, :]  # B H/2 W/2 T/2 C
        x4 = x[:, 1::2, 1::2, 0::2, :]  # B H/2 W/2 T/2 C
        x5 = x[:, 0::2, 1::2, 1::2, :]  # B H/2 W/2 T/2 C
        x6 = x[:, 1::2, 0::2, 1::2, :]  # B H/2 W/2 T/2 C
        x7 = x[:, 1::2, 1::2, 1::2, :]  # B H/2 W/2 T/2 C
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B H/2 W/2 T/2 8*C
        x = x.view(B, -1, 8 * C)  # B H/2*W/2*T/2 8*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpanding(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(
            dim, 4*dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        # import math
        x = self.expand(x)
        # print(x.shape)
        # B, L, C = x.shape
        # root = math.pow(L,1/3)
   
        # root = int(math.ceil(root))

        # x = x.view(B, root,root,root,C)

        B, H, W, D, C = x.shape
        x = rearrange(x, 'b h w d (p1 p2 p3 c)-> b (h p1) (w p2) (d p3)c', p1=2, p2=2, p3=2, c=C//8)
        x= self.norm(x)

        return x



class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, downsample=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=nn.LayerNorm, reduce_factor=4)
        else:
            self.downsample = None

    def forward(self, x, H, W, T):
        B, C = x.shape[0], x.shape[-1]
        assert C == self.dim
        x_norm = self.norm(x)
        if x_norm.dtype == torch.float16:
            x_norm = x_norm.type(torch.float32)
        x_mamba = self.mamba(x_norm)
        x = x_mamba.type(x.dtype)

        if self.downsample is not None:
            x_down = self.downsample(x, H, W, T)
            Wh, Ww, Wt = (H + 1) // 2, (W + 1) // 2, (T + 1) // 2
            return x, H, W, T, x_down, Wh, Ww, Wt
        else:
            return x, H, W, T, x, H, W, T


class VMambaLayer(nn.Module):
    def __init__(self, dim, depths, d_state=16, d_conv=4, expand=2, downsample=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        # self.mamba = Mamba(
        #     d_model=dim,  # Model dimension d_model
        #     d_state=d_state,  # SSM state expansion factor
        #     d_conv=d_conv,  # Local convolution width
        #     expand=expand,  # Block expansion factor
        # )
        self.mamba = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=0.,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0.,
                d_state=d_state,
            )
            for i in range(depths)])
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=nn.LayerNorm, reduce_factor=4)
        else:
            self.downsample = None

    def forward(self, x, H, W, T):
        B, C = x.shape[0], x.shape[-1]
        assert C == self.dim
        x_norm = self.norm(x)
        if x_norm.dtype == torch.float16:
            x_norm = x_norm.type(torch.float32)
        for blk in self.mamba:
            x_mamba = blk(x_norm)
        x = x_mamba.type(x.dtype)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W, T)
            Wh, Ww, Wt = (H + 1) // 2, (W + 1) // 2, (T + 1) // 2
            return x, H, W, T, x_down, Wh, Ww, Wt
        else:
            return x, H, W, T, x, H, W, T

class VMambaLayer_up(nn.Module):
    def __init__(self, dim, depths, d_state=16, d_conv=4, expand=2, upsample=PatchExpanding):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        # self.mamba = Mamba(
        #     d_model=dim,  # Model dimension d_model
        #     d_state=d_state,  # SSM state expansion factor
        #     d_conv=d_conv,  # Local convolution width
        #     expand=expand,  # Block expansion factor
        # )
        self.mamba = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=0.,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0.,
                d_state=d_state,
            )
            for i in range(depths)])
        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpanding(dim, norm_layer=nn.LayerNorm)
        else:
            self.upsample = None
        self.concat_back_dim = nn.Linear(int(dim),int(dim//2)) 
    def forward(self, x, skip):
        x = x.permute(0, 2, 3, 4, 1)
        skip = skip.permute(0, 2, 3, 4, 1)
        if self.upsample is not None:
            x = self.upsample(x)

        x = torch.cat([x,skip],-1)
        B,D,H,W,C = x.size()

        x= x.view(B, D*H*W, C)
    
        for blk in self.mamba:
            x = blk(x)
        x = self.concat_back_dim(x)
        x = x.view(B,-1,D,H,W)
        return x
        

if __name__ == '__main__':
    model = VMambaLayer_up(96*4,2).to('cuda')
    # model = PatchExpanding(192).to('cuda')
    input1 = torch.randn(1, 384, 3, 3, 3).cuda()
    input2 = torch.randn(1, 192, 6, 6, 6).cuda()
    out = model(input1, input2)
    print(out.shape)