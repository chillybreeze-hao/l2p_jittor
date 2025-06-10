import logging
import math
from typing import Callable
from typing import Union, Optional
import jittor as jt
import jittor.nn as nn
from functools import partial
from timm.models.helpers import checkpoint_seq
import numpy as np

from prompt_jittor import Prompt
from itertools import repeat
import collections.abc

_logger = logging.getLogger(__name__)

# -------------辅助模块----------------
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

def _assert(condition, message=""):
    """断言函数"""
    if not condition:
        raise AssertionError(message)
    
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding - Jittor版本
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        # Jittor的Conv2d
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    def execute(self, x):
        """Jittor使用execute而不是forward"""
        
        B, C, H, W = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        
        x = self.proj(x)
        
        if self.flatten:
            # Jittor版本的flatten和transpose
            x = x.flatten(start_dim=2)  # BCHW -> BC(HW)
            x = x.transpose(1, 2)       # BC(HW) -> B(HW)C, 即BNC
        x = self.norm(x)
        return x
    
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks - Jittor版本
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])
    def execute(self, x):
        """Jittor使用execute而不是forward"""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    
    keep_prob = 1 - drop_prob
    # 适用于不同维度的张量，不仅仅是2D卷积网络
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    
    # 生成伯努利分布的随机掩码
    random_tensor = jt.random(shape)  # 生成0-1之间的随机数
    random_tensor = (random_tensor < keep_prob).float()  # 转换为伯努利分布
    
    # 如果需要按保持概率缩放
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor = random_tensor / keep_prob
    
    return x * random_tensor

class DropPath(nn.Module):  
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):

        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    def execute(self, x):

        return drop_path(x, self.drop_prob, self.is_training(), self.scale_by_keep)
    
    def forward(self, x):
        return self.execute(x)
    
    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}, scale_by_keep={self.scale_by_keep}'
    
def named_apply(fn: Callable, module: nn.Module, name='', depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    
    # Jittor的模块遍历方式
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    
    if depth_first and include_root:
        fn(module=module, name=name)
    
    return module

def adapt_input_conv(in_chans, conv_weight):
    conv_type = conv_weight.dtype
    conv_weight = conv_weight.float()  # 确保是float类型以便在CPU上进行求和
    O, I, J, K = conv_weight.shape
    
    if in_chans == 1:
        if I > 3:
            # 检查权重形状是否可以被3整除
            assert conv_weight.shape[1] % 3 == 0, f"输入通道数{I}不能被3整除"
            # 对于使用space2depth的stem
            conv_weight = conv_weight.view(O, I // 3, 3, J, K)
            conv_weight = conv_weight.sum(dim=2, keepdim=False)
        else:
            conv_weight = conv_weight.sum(dim=1, keepdim=True)
    elif in_chans != 3:
        if I != 3:
            raise NotImplementedError(f'不支持的权重格式转换: 输入通道从{I}到{in_chans}')
        else:
            # 注意：这种策略应该比随机初始化更好，但可能存在其他组合
            # 原始RGB输入层权重的组合对特定情况效果更好
            repeat = int(math.ceil(in_chans / 3))
            conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv_weight *= (3 / float(in_chans))
    conv_weight = conv_weight.to(conv_type)
    return conv_weight



class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def execute(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 替换unbind操作

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = nn.softmax(attn, dim=-1)  # 使用nn.softmax
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = jt.ones(dim) * init_values
    def execute(self, x):
        if self.inplace:
            # Jittor中直接修改原张量的操作
            x *= self.gamma
            return x
        else:
            return x * self.gamma
        
class Block(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def execute(self, x):
        # 第一个残差连接：Multi-Head Attention
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        # 第二个残差连接：MLP
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
 
    
class ResPostBlock(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.init_values = init_values
        # 注意力子层
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # MLP子层
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 初始化权重
        self.init_weights()
    def init_weights(self):
        """
        初始化LayerNorm权重，用于控制残差连接的强度
        类似于LayerScale的作用，但直接作用在norm的权重上
        """
        if self.init_values is not None:
            # 将norm层的权重初始化为较小的值，用于稳定训练
            jt.init.constant_(self.norm1.weight, self.init_values)
            jt.init.constant_(self.norm2.weight, self.init_values)
    def execute(self, x):
        # Post-Norm架构：先计算子层，再进行归一化
        x = x + self.drop_path1(self.norm1(self.attn(x)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x
    
    

class VisionTransformer_jittor(nn.Module):
    """ Vision Transformer
    A Jittor impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, init_values=None,
            class_token=True, no_embed_class=False, fc_norm=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            weight_init='', embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block,
            prompt_length=None, embedding_key='cls', prompt_init='uniform', prompt_pool=False, prompt_key=False, pool_size=None,
            top_k=None, batchwise_prompt=False, prompt_key_init='uniform', head_type='token', use_prompt_mask=False,):
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.img_size = img_size
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.class_token = class_token
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False
        # Patch embedding layer
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        # Class token
        self.cls_token = jt.zeros(1, 1, embed_dim) if class_token else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        if prompt_length is not None and pool_size is not None and prompt_pool:
            embed_len += prompt_length * top_k
        
        # Position embedding
        self.pos_embed = jt.randn(1, embed_len, embed_dim) * .02
        self.pos_drop = nn.Dropout(p=drop_rate)
        # Prompt相关参数
        self.prompt_pool = prompt_pool
        self.head_type = head_type
        self.use_prompt_mask = use_prompt_mask
        
        if prompt_length is not None and pool_size is not None and prompt_pool: 
            self.prompt = Prompt(length=prompt_length, embed_dim=embed_dim, embedding_key=embedding_key, prompt_init=prompt_init,
                    prompt_pool=prompt_pool, prompt_key=prompt_key, pool_size=pool_size, top_k=top_k, batchwise_prompt=batchwise_prompt,
                    prompt_key_init=prompt_key_init,)
        # Stochastic depth decay rule
        dpr = [x.item() for x in jt.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        
        # Normalization layers
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        
        # Classifier Head
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        if weight_init != 'skip':
            self.init_weights(weight_init)
            
    def init_weights(self, mode=''):
        """初始化模型权重"""
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        # 使用Jittor的初始化方法
        jt.init.trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            jt.init.gauss_(self.cls_token, mean=0, std=1e-6)
        # 应用权重初始化到所有子模块
        named_apply(get_init_weights_vit(mode, head_bias), self)
        
    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)
        
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)
        
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}
    
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )    
        
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable
    
    def get_classifier(self):
        return self.head
    
    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
    def forward_features(self, x, task_id=-1, cls_features=None, train=False):
        # 图像分块嵌入
        x = self.patch_embed(x)
        # Prompt机制处理
        if hasattr(self, 'prompt'):
            if self.use_prompt_mask and train:
                # 计算当前任务的prompt范围
                start = task_id * self.prompt.top_k
                end = (task_id + 1) * self.prompt.top_k
                
                # 创建prompt mask
                single_prompt_mask = jt.arange(start, end)
                prompt_mask = single_prompt_mask.unsqueeze(0).expand(x.shape[0], -1)
                
                # 检查prompt pool边界
                if end > self.prompt.pool_size:
                    prompt_mask = None
            else:
                prompt_mask = None
            # 应用prompt
            res = self.prompt(x, prompt_mask=prompt_mask, cls_features=cls_features)
            self.total_prompt_len = res['total_prompt_len']
            x = res['prompted_embedding']
        else:
            res=dict() 
        # 添加CLS token
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = jt.cat((cls_tokens, x), dim=1)
        # 位置编码
        x = self.pos_drop(x + self.pos_embed)
        
        if self.grad_checkpointing:
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        # 最终归一化
        x = self.norm(x)
        res['x'] = x
        return res
    
    def forward_head(self, res, pre_logits: bool = False):
        x = res['x']
        if self.class_token and self.head_type == 'token':
            x = x[:, 0]
        elif self.head_type == 'gap' and self.global_pool == 'avg':
            x = x.mean(dim=1)
        elif self.head_type == 'prompt' and self.prompt_pool:
            x = x[:, 1:(1 + self.total_prompt_len)] if self.class_token else x[:, 0:self.total_prompt_len]
            x = x.mean(dim=1)
        elif self.head_type == 'token+prompt' and self.prompt_pool and self.class_token:
            x = x[:, 0:self.total_prompt_len + 1]
            x = x.mean(dim=1)
        else:
            raise ValueError(f'Invalid classifier={self.classifier}')
        
        res['pre_logits'] = x

        x = self.fc_norm(x)
        
        res['logits'] = self.head(x)
        
        return res
    
    def execute(self, x, task_id=-1, cls_features=None, train=False):
        res = self.forward_features(x, task_id=task_id, cls_features=cls_features, train=train)
        res = self.forward_head(res)
        return res
        
def init_weights_vit_timm(module: nn.Module, name: str = ''):
    """ ViT weight initialization adapted for Jittor (original timm impl) """
    # 处理线性层 (Jittor 的 nn.Linear 对应 PyTorch)
    if isinstance(module, nn.Linear):
        # Jittor 的截断正态分布初始化 (参数与 PyTorch 的 trunc_normal_ 对齐)
        jt.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
        
        # 初始化偏置为零
        if module.bias is not None:
            jt.init.constant_(module.bias, 0.0)
    
    # 处理带有自定义初始化方法的模块 (如 LayerNorm)
    elif hasattr(module, 'init_weights') and callable(module.init_weights):
        module.init_weights()
        
def init_weights_vit_jax(module: nn.Module, name: str = '', head_bias: float = 0.):
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            # 分类头：权重初始化为0，偏置为指定值
            nn.init.constant_(module.weight, 0.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, head_bias)
        else:
            # 其他线性层：Xavier均匀初始化
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                if 'mlp' in name:
                    # MLP层的偏置使用小的正态分布
                    nn.init.normal_(module.bias, mean=0.0, std=1e-6)
                else:
                    # 其他层的偏置初始化为0
                    nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Conv2d):
        # 卷积层使用LeCun正态初始化
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif hasattr(module, 'init_weights'):
        # 如果模块有自定义的初始化方法，调用它
        module.init_weights()
        
def init_weights_vit_moco(module: nn.Module, name: str = ''):
    """ ViT weight initialization, matching moco-v3 impl minus fixed PatchEmbed """
    if isinstance(module, nn.Linear):
        if 'qkv' in name:
            # treat the weights of Q, K, V separately
            val = math.sqrt(6. / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
            nn.init.uniform_(module.weight, -val, val)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()
        
def get_init_weights_vit(mode='jax', head_bias: float = 0.):
    if 'jax' in mode:
        return partial(init_weights_vit_jax, head_bias=head_bias)
    elif 'moco' in mode:
        return init_weights_vit_moco
    else:
        return init_weights_vit_timm
    
def _load_weights(model, checkpoint_path: str, prefix: str = ''):
    
    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                # HWIO -> OIHW (Flax到PyTorch/Jittor的卷积格式)
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                # HWI -> IHW 
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                # IJ -> JI (转置线性层权重)
                w = w.transpose([1, 0])
        return jt.array(w)
    
    # 加载.npz文件
    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'
        
    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        
        stem.conv.weight.assign(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.assign(
            jt.array(_n2p(w[f'{prefix}gn_root/scale'])).reshape(stem.norm.weight.shape)
        )
        stem.norm.bias.assign(
            jt.array(_n2p(w[f'{prefix}gn_root/bias'])).reshape(stem.norm.bias.shape)
        )
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.assign(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.assign(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.assign(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.assign(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.assign(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.assign(_n2p(w[f'{bp}gn_proj/bias']))
                        
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
        
    model.patch_embed.proj.weight.assign(embed_conv_w)
    model.patch_embed.proj.bias.assign(_n2p(w[f'{prefix}embedding/bias']))
    
    # 加载CLS token
    model.cls_token.assign(_n2p(w[f'{prefix}cls'], t=False))
    
    # 加载和调整位置编码
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(
            pos_embed_w,
            model.pos_embed,
            getattr(model, 'num_prefix_tokens', 1),
            model.patch_embed.grid_size
        )
    model.pos_embed.assign(pos_embed_w)
    
    # 加载最终归一化层
    model.norm.weight.assign(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.assign(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    
    if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
        model.head.weight.assign(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.assign(_n2p(w[f'{prefix}head/bias']))
    
    for i, block in enumerate(model.blocks):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        
        # 加载第一个LayerNorm
        block.norm1.weight.assign(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.assign(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        
        # 加载QKV权重 (需要将分离的Q、K、V权重合并)
        qkv_weight = jt.concat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).transpose(1, 0) 
            for n in ('query', 'key', 'value')
        ], dim=0)
        block.attn.qkv.weight.assign(qkv_weight)
        
        # 加载QKV偏置
        qkv_bias = jt.concat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) 
            for n in ('query', 'key', 'value')
        ], dim=0)
        block.attn.qkv.bias.assign(qkv_bias)
        
        # 加载注意力输出投影
        block.attn.proj.weight.assign(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.assign(_n2p(w[f'{mha_prefix}out/bias']))
        
        # 加载MLP层权重
        for r in range(2):
            fc_layer = getattr(block.mlp, f'fc{r + 1}')
            fc_layer.weight.assign(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            fc_layer.bias.assign(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        
        # 加载第二个LayerNorm
        block.norm2.weight.assign(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.assign(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))
                
def resize_pos_embed(posemb, posemb_new, num_prefix_tokens=1, gs_new=()):
    """调整位置编码的尺寸以适配不同的输入图像大小
    
    Args:
        posemb: 原始位置编码 [1, N, D]
        posemb_new: 目标位置编码（用于获取目标形状）[1, M, D]
        num_prefix_tokens: 前缀token数量（通常是CLS token）
        gs_new: 新的网格尺寸 [H, W]
    
    Returns:
        调整后的位置编码 [1, M, D]
    """
    
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    
    ntok_new = posemb_new.shape[1]
    
    if num_prefix_tokens:
        # 分离前缀token（如CLS token）和网格位置编码
        posemb_prefix = posemb[:, :num_prefix_tokens]
        posemb_grid = posemb[0, num_prefix_tokens:]  # 移除batch维度
    else:
        posemb_prefix = posemb[:, :0]  # 空张量
        posemb_grid = posemb[0]
    
    # 计算原始网格尺寸
    gs_old = int(math.sqrt(len(posemb_grid)))
    
    # 处理prompt tokens的情况
    if ntok_new > gs_old ** 2:
        # 当新的token数量大于原始网格大小时，扩展前缀位置编码用于prompt tokens
        ntok_new -= gs_old ** 2
        # 扩展CLS token的位置编码给prompt tokens使用
        posemb_prefix = posemb_prefix.expand(-1, ntok_new, -1)
    
    # 向后兼容性：如果没有指定新的网格尺寸，计算正方形网格
    if not len(gs_new):
        gs_new = [int(math.sqrt(ntok_new))] * 2
    
    assert len(gs_new) >= 2, "gs_new must have at least 2 dimensions"
    
    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    
    # 将位置编码重塑为2D网格形式进行插值
    # posemb_grid: [H*W, D] -> [1, H, W, D] -> [1, D, H, W]
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    
    # 使用双三次插值调整网格尺寸
    posemb_grid = jt.nn.interpolate(
        posemb_grid, 
        size=gs_new, 
        mode='bicubic', 
        align_corners=False
    )
    
    # 转换回原始格式: [1, D, H, W] -> [1, H, W, D] -> [1, H*W, D]
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    
    # 合并前缀和网格位置编码
    posemb = jt.concat([posemb_prefix, posemb_grid], dim=1)
    
    return posemb

def checkpoint_filter_fn(state_dict, model, adapt_layer_scale=False):
    import re
    
    out_dict = {}
    
    # 处理嵌套的模型状态字典（如DeiT模型）
    if 'model' in state_dict:
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        # 确保v是Jittor张量
        if not isinstance(v, jt.Var):
            if hasattr(v, 'detach') and hasattr(v, 'cpu'):  # PyTorch张量
                v = jt.array(v.detach().cpu().numpy())
            elif hasattr(v, 'shape'):  # NumPy数组
                v = jt.array(v)
            else:
                v = jt.array(v)
        
        # 1. 处理patch embedding权重转换
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # 将旧模型的线性投影权重转换为卷积权重格式
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        
        # 2. 处理位置编码尺寸调整
        elif k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
            # 当使用的模型尺寸与预训练权重不同时，调整位置编码
            v = resize_pos_embed(
                v,
                model.pos_embed,
                0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1),
                model.patch_embed.grid_size
            )
        
        # 3. 处理layer-scale参数重命名（DeiT3模型）
        elif adapt_layer_scale and 'gamma_' in k:
            # 将gamma_X重新映射为lsX.gamma格式
            k = re.sub(r'gamma_([0-9])', r'ls\1.gamma', k)
        
        # 4. 跳过pre_logits层
        elif 'pre_logits' in k:
            continue
        
        out_dict[k] = v
    
    return out_dict

                        