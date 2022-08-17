import numpy as np
import snntorch as snn
from functools import partial
import torch
import torch.nn as nn
from snntorch import surrogate
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from spikingjelly.clock_driven import neuron


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class SpikeOperator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mem):
        spk = (mem > 0).float()  # Heaviside on the forward pass: Eq(2)
        ctx.save_for_backward(spk)  # store the spike for use in the backward pass
        return spk

    @staticmethod
    def backward(ctx, grad_output):
        (spk,) = ctx.saved_tensors  # retrieve the spike
        grad = grad_output * spk  # scale the gradient by the spike: 1/0
        return grad


spike_fc = SpikeOperator.apply

spike_grad = surrogate.fast_sigmoid(slope=25)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        # patch_dim = in_c * patch_size[0] * patch_size[1]
        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size[0], p2=patch_size[1]),
        #     nn.Linear(patch_dim, embed_dim),
        # )
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        # x = self.to_patch_embedding(x)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,  # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 threshold=0.3):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.fc1 = nn.Linear(head_dim, head_dim)
        self.fc2 = nn.Linear(head_dim, head_dim)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()




    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        # B, N, C = x.shape
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]
        # Q_spk = self.relu1(q)
        # K_spk = self.relu2(k)
        # Q_spk = self.fc1(Q_spk)
        # K_spk = self.fc2(K_spk)
        # att_spk = self.relu3((Q_spk @ K_spk.transpose(-2, -1)) * self.scale)
        # V_spk = self.relu4(v)
        # att = att_spk.softmax(dim=-1)
        # x = (att @ V_spk).transpose(1, 2).reshape(B, N, C)
        # x = self.proj(x)
        # x = self.proj_drop(x)
        # return x


        # B, N, C = x.shape
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]
        # Q_spk = self.relu1(q)
        # K_spk = self.relu1(k)
        # Q_spk = self.fc1(Q_spk)
        # K_spk = self.fc2(K_spk)
        # att_spk = self.relu2((Q_spk @ K_spk.transpose(-2, -1)) * self.scale)
        # V_spk = self.relu1(v)
        # att = att_spk.softmax(dim=-1)
        # x = (att @ V_spk).transpose(1, 2).reshape(B, N, C)
        # x = self.proj(x)
        # x = self.proj_drop(x)


        B, N, C = x.shape
        # Q_spk_rec = []
        # K_spk_rec = []
        V_spk_rec = []
        att_spk_rec = []

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]
        T = 4
        n = T / 2

        V_spk_rec = torch.zeros((B, self.num_heads, N, C // self.num_heads)).to(x.device)
        att_spk_rec = torch.zeros((B, self.num_heads, N, N)).to(x.device)
        for t in range(T):
            if t < int(n):
                Q_spk = self.relu1(q)
                K_spk = self.relu2(k)
                Q_spk = self.fc1(Q_spk)
                K_spk = self.fc2(K_spk)
                att_spk = self.relu3((Q_spk @ K_spk.transpose(-2, -1)) * self.scale)
                att_spk_rec += att_spk
            V_spk = self.relu4(v)
            V_spk_rec += V_spk
        att = att_spk_rec.softmax(dim=-1)
        x = (att @ V_spk_rec).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 threshold=0.3):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio, threshold=threshold)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=6, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):

        super(VisionTransformer, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        patch_dim = in_c * patch_size * patch_size
        # thresh_list = np.load('./train_multi_GPU/best_max_threshold.npy')
        thresh_list = [0.5] * depth

        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('t b c (h p1) (w p2) -> t b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
        #     nn.Linear(patch_dim, embed_dim),
        # )

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        # self.pos_embed = nn.Parameter(torch.zeros(1, 1, num_patches + self.num_tokens, embed_dim))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer, threshold=thresh_list[i])
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        # self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward(self, x):
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        # x = self.avgpool(x.transpose(1, 2))  # [B, C, 1]
        # x = torch.flatten(x, 1)
        x = self.head(x[:,0])
        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def Cifarmodel():
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                             drop_ratio=0.1,
                             drop_path_ratio=0.1,
                              num_classes=10)
    return model


# def Cifarmodel():
#     return VisionTransformer(img_size=32,
#                              patch_size=4,
#                              embed_dim=512,
#                              depth=8,
#                              num_heads=8,
#                              drop_ratio=0.1,
#                              drop_path_ratio=0.1,
#                              representation_size=None,
#                              num_classes=10)



def Mnistmodel():
    return VisionTransformer(img_size=28,
                             in_c=1,
                             patch_size=14,
                             embed_dim=392,
                             depth=6,
                             num_heads=8,
                             representation_size=None,
                             # drop_ratio=0.1,
                             # drop_path_ratio=0.1,
                             num_classes=10)
