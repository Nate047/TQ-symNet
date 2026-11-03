"""Experimental adapters around PINNED upstream detection implementations.

Upstream math/functions are not rewritten. AST edits only isolate registries
and resolve utility imports under private namespaces. No substitute CUDA ops.
For coarse-stem models, real stride4/8 stem taps are exposed and the final
stride64 semantic map is concatenated into stride32 (bilinear align_corners=False).
That adaptation is NOT a native architecture/parameter-count-only comparison.
"""
import ast
import hashlib
import importlib
import json
import logging
import math
from pathlib import Path
import sys
import types

import torch
from torch import nn
from torch.nn import functional as F
from mmdet.models.builder import BACKBONES
from TQ_SymNet.capacity_pretrained import load_official_checkpoint, strict_load_backbone

ROOT = Path(__file__).resolve().parents[2] / 'third_party' / 'capacity_backbones'


class _PrivateRegistry:
    def register_module(self, *args, **kwargs):
        return lambda item: item


def _package(name, path):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)]
        sys.modules[name] = mod
    return name


def _source(path, module_name, injected=None):
    """Namespace/import shim only; every function body remains upstream."""
    if module_name in sys.modules:
        return sys.modules[module_name]
    text = path.read_text(encoding='utf-8-sig')
    lock = json.loads((ROOT/'SOURCES.json').read_text(encoding='utf-8'))
    relative = path.relative_to(ROOT).as_posix()
    actual_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
    if lock['sha256'].get(relative) != actual_hash:
        raise RuntimeError('Upstream source missing/changed: '+relative)
    tree = ast.parse(text, filename=str(path))
    skip = {'mmcv_custom', 'efficientViM_utils', 'mmdet.registry',
            'mmdet.models.builder', 'backbones.lib_mamba.vmambanew'}
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and node.module in skip)]
    for node in ast.walk(tree):
        # timm 0.6 uses this older import path; operator implementation unchanged.
        if isinstance(node, ast.ImportFrom) and node.module == 'timm.layers':
            node.module = 'timm.models.layers'
    mod = types.ModuleType(module_name)
    mod.__file__ = str(path)
    mod.__package__ = module_name.rpartition('.')[0]
    mod.__dict__.update(BACKBONES=_PrivateRegistry(), MODELS=_PrivateRegistry())
    mod.__dict__.update(injected or {})
    sys.modules[module_name] = mod
    try:
        exec(compile(tree, str(path), 'exec'), mod.__dict__)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    mod.source_sha256 = hashlib.sha256(text.encode('utf-8')).hexdigest()
    return mod


def _load_pretrained(net, path, family):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Download official weights first: ' + str(path))
    obj = load_official_checkpoint(path)
    for _ in range(3):
        key = next((k for k in ('state_dict','model','model_ema')
                    if isinstance(obj,dict) and isinstance(obj.get(k),dict)), None)
        if key is None:
            break
        obj = obj[key]
    if not isinstance(obj,dict):
        raise TypeError('Expected a checkpoint state_dict')
    target = net.state_dict()
    state, resized = {}, []
    regenerated = [k for k in target if family.startswith('lsnet') and
                   k.endswith('attention_bias_idxs')]
    for key, value in obj.items():
        if not isinstance(value,torch.Tensor):
            continue
        for prefix in ('module.','backbone.','net.'):
            if key.startswith(prefix): key = key[len(prefix):]
        if key not in target:
            continue
        if key in regenerated:
            continue
        if (family.startswith('lsnet') and key.endswith('attention_biases') and
                tuple(value.shape) != tuple(target[key].shape)):
            # Same interpolation as official LSNet detection init_weights:
            # classification uses a smaller bias grid; detector uses14x14.
            if value.dim()!=2 or value.shape[0]!=target[key].shape[0]:
                raise RuntimeError('Incompatible LSNet attention heads: '+key)
            src,dst = math.isqrt(value.shape[1]),math.isqrt(target[key].shape[1])
            if src*src!=value.shape[1] or dst*dst!=target[key].shape[1]:
                raise RuntimeError('Non-square LSNet bias grid: '+key)
            value = F.interpolate(value.float().reshape(1,value.shape[0],src,src),
                size=(dst,dst),mode='bicubic',align_corners=False).reshape_as(target[key])
            resized.append(key)
        state[key] = value
    # Official OverLoCK detector introduces these output adapters after pretraining.
    allowed = ('extra_norm.','h_proj.') if family == 'overlock_xt' else ()
    audit = strict_load_backbone(net,state,allowed_new=allowed,regenerated=regenerated)
    audit.update(checkpoint=str(path),resized_attention_biases=resized)
    logging.getLogger('mmdet').info('[CAPACITY-PRETRAIN] %s',audit)
    return audit


@BACKBONES.register_module()
class TQEfficientBackbone(nn.Module):
    CHANNELS = {'lsnet_s':(48,96,192,768),'lsnet_b':(64,128,256,896),
                'overlock_xt':(56,112,340,420),
                'mobilemamba_b1':(50,100,200,824),
                'efficientvim_m3':(56,112,224,832),
                'efficientvim_m4':(56,112,224,832)}

    def __init__(self, family, pretrained, norm_eval=True):
        super().__init__()
        if family not in self.CHANNELS:
            raise ValueError('Unknown official backbone '+str(family))
        self.family, self.pretrained = family, pretrained
        self.norm_eval = bool(norm_eval)
        self._loaded = False
        self.pretrained_audit = None
        self.out_channels = self.CHANNELS[family]
        try:
            self.net = self._build(family)
        except (ImportError, ModuleNotFoundError, AttributeError) as exc:
            raise ImportError('Official %s dependencies unavailable. See '
                'configs/backbone_capacity/EXTRA_BACKBONES.md; no approximate '
                'operator is substituted. Original: %s' % (family,exc)) from exc
        self.init_weights()

    @staticmethod
    def _build(family):
        if family.startswith('lsnet'):
            folder = ROOT/'lsnet/detection/model'
            pkg = _package('_tq_capacity_lsnet',folder)
            mod = _source(folder/'lsnet.py',pkg+'.lsnet')
            return getattr(mod,family)(pretrained=None,frozen_stages=-1)
        if family == 'overlock_xt':
            folder = ROOT/'overlock/detection/models'
            pkg = _package('_tq_capacity_overlock',folder)
            mod = _source(folder/'overlock.py',pkg+'.overlock')
            # Standard torch convolution is an upstream-supported path, not a
            # custom replacement for natten (which is still required).
            return mod.OverLoCK(depth=[2,2,3,2],sub_depth=[6,2],
                embed_dim=[56,112,256,336],kernel_size=[17,15,13,7],
                mlp_ratio=[4,4,4,4],sub_num_heads=[4,6],sub_mlp_ratio=[3,3],
                use_gemm=False, use_checkpoint=[0,0,0,0])
        if family == 'mobilemamba_b1':
            folder = ROOT/'mobilemamba/downstream/det/backbones'
            pkg = _package('_tq_capacity_mobilemamba',folder)
            _package(pkg+'.lib_mamba',folder/'lib_mamba')
            scan = importlib.import_module(pkg+'.lib_mamba.csms6s')
            if not any(getattr(scan,key,False) for key in (
                    'WITH_SELECTIVESCAN_OFLEX','WITH_SELECTIVESCAN_CORE',
                    'WITH_SELECTIVESCAN_MAMBA')):
                raise ImportError('Install official selective_scan CUDA extension; '
                                  'refusing silent Python-loop fallback')
            ss2d = importlib.import_module(pkg+'.lib_mamba.vmambanew').SS2D
            mod = _source(folder/'mobilemamba.py',pkg+'.mobilemamba',dict(SS2D=ss2d))
            return mod.MobileMamba(img_size=256,embed_dim=[200,376,448],
                depth=[2,3,2],global_ratio=[.8,.7,.6],local_ratio=[.2,.2,.3],
                ssm_ratio=2,drop_path=.03,
                pretrained=None,sync_bn=False,frozen_stages=-1)
        folder = ROOT/'efficientvim/detection'
        pkg = _package('_tq_capacity_efficientvim',folder)
        util = _source(folder/'efficientViM_utils.py',pkg+'.utils')
        injected = {name:getattr(util,name) for name in (
            'LayerNorm1D','LayerNorm2D','ConvLayer1D','ConvLayer2D',
            'FFN','Stem','PatchMerging')}
        mod = _source(folder/'efficientViM.py',pkg+'.model',injected)
        m4 = family.endswith('m4')
        return mod.EfficientViM(frozen_stages=-1,embed_dim=[224,320,512],
            depths=[3,4,2] if m4 else [2,2,2],
            state_dim=[64,32,16] if m4 else [49,25,9],ssd_expand=1.)

    def init_weights(self):
        if not self._loaded:
            self.pretrained_audit = _load_pretrained(self.net,self.pretrained,self.family)
            self._loaded = True
            logging.getLogger('mmdet').info(
                '[CAPACITY-ADAPTER] family=%s channels=%s strides=4/8/16/32 '
                'deep64=concat_into32 normEval=%s',
                self.family,self.out_channels,self.norm_eval)

    def train(self, mode=True):
        super().train(mode)
        if self.norm_eval:
            for module in self.modules():
                if isinstance(module,nn.modules.batchnorm._BatchNorm): module.eval()
        return self

    @staticmethod
    def _deep_fusion(middle, deep):
        return torch.cat((middle,F.interpolate(deep,size=middle.shape[-2:],
                          mode='bilinear',align_corners=False)),dim=1)

    def forward(self, x):
        shape = x.shape[-2:]
        if self.family == 'overlock_xt':
            outs = self.net(x)
        elif self.family.startswith('lsnet'):
            for index,layer in enumerate(self.net.patch_embed):
                x = layer(x)
                if index == 3: p2 = x
            p3 = self.net.blocks1(x)
            p4 = self.net.blocks2(p3)
            p5 = self.net.blocks3(p4)
            deep = self.net.blocks4(p5)
            outs = (p2,p3,p4,self._deep_fusion(p5,deep))
        else:
            efficient = self.family.startswith('efficientvim')
            stem = self.net.patch_embed.conv if efficient else self.net.patch_embed
            for index,layer in enumerate(stem):
                x = layer(x)
                if index == (1 if efficient else 3): p2 = x
                if index == (2 if efficient else 5): p3 = x
            if efficient:
                levels = []
                for stage in self.net.stages:
                    x,level,_ = stage(x)
                    levels.append(level)
                p4,p5,deep = levels
            else:
                p4 = self.net.blocks1(x)
                p5 = self.net.blocks2(p4)
                deep = self.net.blocks3(p5)
            outs = (p2,p3,p4,self._deep_fusion(p5,deep))
        for level,(out,channels) in enumerate(zip(outs,self.out_channels)):
            stride = 2**(level+2)
            expected = tuple((s+stride-1)//stride for s in shape)
            if out.shape[1] != channels or tuple(out.shape[-2:]) != expected:
                raise RuntimeError('Backbone pyramid mismatch at level %d: %s expected C%s/%s'
                    % (level,tuple(out.shape),channels,expected))
        return tuple(outs)
