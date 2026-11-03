from efficientViM_utils import LayerNorm1D, LayerNorm2D, ConvLayer1D, ConvLayer2D, FFN, Stem, PatchMerging
from timm.models.vision_transformer import trunc_normal_
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn as nn
import torch


class HSMSSD(nn.Module):
    def __init__(self, d_model, ssd_expand=1, A_init_range=(1, 16), state_dim = 64):
        super().__init__()
        self.ssd_expand = ssd_expand
        self.d_inner = int(self.ssd_expand * d_model)
        self.state_dim = state_dim

        self.BCdt_proj = ConvLayer1D(d_model, 3*state_dim, 1, norm=None, act_layer=None)
        conv_dim = self.state_dim*3
        self.dw = ConvLayer2D(conv_dim, conv_dim, 3,1,1, groups=conv_dim, norm=None, act_layer=None, bn_weight_init=0) 
        self.hz_proj = ConvLayer1D(d_model, 2*self.d_inner, 1, norm=None, act_layer=None)
        self.out_proj = ConvLayer1D(self.d_inner, d_model, 1, norm=None, act_layer=None, bn_weight_init=0)

        A = torch.empty(self.state_dim, dtype=torch.float32).uniform_(*A_init_range)
        self.A = torch.nn.Parameter(A)
        self.act = nn.SiLU()
        self.D = nn.Parameter(torch.ones(1))
        self.D._no_weight_decay = True

    def forward(self, x, H, W):
        batch, _, L= x.shape
        BCdt = self.dw(self.BCdt_proj(x).view(batch,-1, H, W)).flatten(2)
        B,C,dt = torch.split(BCdt, [self.state_dim, self.state_dim,  self.state_dim], dim=1) 
        A = (dt + self.A.view(1,-1,1)).softmax(-1) 
        
        AB = (A * B) 
        h = x @ AB.transpose(-2,-1) 
        
        h, z = torch.split(self.hz_proj(h), [self.d_inner, self.d_inner], dim=1) 
        h = self.out_proj(h * self.act(z)+ h * self.D)
        y = h @ C # B C N, B C L -> B C L
        
        y = y.view(batch, -1, H, W).contiguous()# + x * self.D  # B C H W
        return y, h


class EfficientViMBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., ssd_expand=1, state_dim=64):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        
        self.mixer = HSMSSD(d_model=dim, ssd_expand=ssd_expand,state_dim=state_dim)  
        self.norm = LayerNorm1D(dim)
        
        self.dwconv1 = ConvLayer2D(dim, dim, 3, padding=1, groups=dim, bn_weight_init=0, act_layer = None)
        self.dwconv2 = ConvLayer2D(dim, dim, 3, padding=1, groups=dim, bn_weight_init=0, act_layer = None)
        
        self.ffn = FFN(in_dim=dim, dim=int(dim * mlp_ratio))
        
        #LayerScale
        self.alpha = nn.Parameter(1e-4 * torch.ones(4,dim), requires_grad=True)
        
    def forward(self, x):
        alpha = torch.sigmoid(self.alpha).view(4,-1,1,1)
        
        # DWconv1
        x = (1-alpha[0]) * x + alpha[0] * self.dwconv1(x)
        
        # HSM-SSD
        x_prev = x
        H, W = x.shape[2:]
        x, h = self.mixer(self.norm(x.flatten(2)), H, W)
        x = (1-alpha[1]) * x_prev + alpha[1] * x
        
        # DWConv2
        x = (1-alpha[2]) * x + alpha[2] * self.dwconv2(x)
        
        # FFN
        x = (1-alpha[3]) * x + alpha[3] * self.ffn(x)
        return x, h


class EfficientViMStage(nn.Module):
    def __init__(self, in_dim, out_dim, depth,  mlp_ratio=4.,downsample=None, ssd_expand=1, state_dim=64):
        super().__init__()
        self.depth = depth
        self.blocks = nn.ModuleList([
            EfficientViMBlock(dim=in_dim, mlp_ratio=mlp_ratio, ssd_expand=ssd_expand, state_dim=state_dim) for _ in range(depth)])
        
        self.downsample = downsample(in_dim=in_dim, out_dim =out_dim) if downsample is not None else None

    def forward(self, x):
        for blk in self.blocks:
            x, h = blk(x)
            
        x_out = x
        if self.downsample is not None:
            x = self.downsample(x)
        return x, x_out, h


class EfficientViM(nn.Module):
    def __init__(self, in_dim=3, frozen_stages=0, embed_dim=[128,256,512], depths=[2, 2, 2], mlp_ratio=4., ssd_expand=1, state_dim=[49,25,9], distillation=False, **kwargs):
        super().__init__()
        self.num_layers = len(depths)
        self.distillation =distillation
        self.patch_embed = Stem(in_dim=in_dim, dim=embed_dim[0])
        PatchMergingBlock = PatchMerging

        # build stages
        self.frozen_stages = frozen_stages
        self.stages = nn.ModuleList()
        for i_layer in range(self.num_layers):
            stage = EfficientViMStage(in_dim=int(embed_dim[i_layer]),
                               out_dim=int(embed_dim[i_layer+1]) if (i_layer < self.num_layers - 1) else None,
                               depth=depths[i_layer],
                               mlp_ratio=mlp_ratio,
                               downsample=PatchMergingBlock if (i_layer < self.num_layers - 1) else None,
                               ssd_expand=ssd_expand,
                               state_dim = state_dim[i_layer])
            self.stages.append(stage)

        self.apply(self._init_weights)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm2D):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, LayerNorm1D):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(EfficientViM, self).train(mode)
        if mode:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x):
        x = self.patch_embed(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x, x_out, h = stage(x)
            outs.append(x_out)

        return tuple(outs)

from mmdet.models.builder import BACKBONES
@BACKBONES.register_module()
def EfficientViM_M4(pretrained=False, frozen_stages=0, **kwargs):
    model = EfficientViM(
        frozen_stages=frozen_stages,
        in_dim=3,
        embed_dim=[224,320,512],
        depths=[3,4,2],
        mlp_ratio=4.,
        ssd_expand=1.,
        state_dim=[64,32,16],
        **kwargs)
    if pretrained is not None:
        checkpoint = torch.load(pretrained, map_location ='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
    return model

