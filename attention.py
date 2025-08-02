import torch
import torch.nn as nn
import torch.nn.functional as F


def LightweightConv(in_channels, kernel_size, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
    )
    
def resize_tensors_to_target(tensor_list, target_size=(100, 100)):
    resized_tensors = []
    for t in tensor_list:
        # If it's a 3D tensor [B, H, W], first add a channel dimension to make it [B, 1, H, W]
        if t.dim() == 3:
            t = t.unsqueeze(1)
            t_resized = F.interpolate(t, size=target_size, mode='bilinear', align_corners=False)
            t_resized = t_resized.squeeze(1)  # Restore to [B, H, W]
        # If it's a 4D tensor [B, C, H, W], directly interpolate
        elif t.dim() == 4:
            t_resized = F.interpolate(t, size=target_size, mode='bilinear', align_corners=False)
        else:
            raise ValueError(f"Unsupported tensor dimension {t.dim()}")
        resized_tensors.append(t_resized)
    return resized_tensors


class MixtureOfPoolingExperts(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.avgpool = nn.AdaptiveAvgPool2d(output_size)
        self.maxpool = nn.AdaptiveMaxPool2d(output_size)

    def forward(self, x):
        avg = self.avgpool(x)
        max_ = self.maxpool(x)

        # The greater the difference, the more it favors max pooling; the smaller the difference, the more it favors avg pooling
        diff = (avg - max_) ** 2
        gate = torch.sigmoid(-diff.mean(dim=(1, 2, 3), keepdim=True))  # shape: [B, 1, 1, 1]

        return gate * max_ + (1 - gate) * avg


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class MultiScaleSelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=4, head_dim=32):  # Changed to 4 heads
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.embed_dim = num_heads * head_dim
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Conv2d(in_channels, self.embed_dim, kernel_size=1)
        self.k_proj = nn.Conv2d(in_channels, self.embed_dim, kernel_size=1)
        self.v_proj = nn.Conv2d(in_channels, self.embed_dim, kernel_size=1)
        self.out_proj = nn.Conv2d(self.embed_dim, in_channels, kernel_size=1)

        self.pool_configs = ['none', 'none', '7x7', '7x7']

        # Lightweight 3x3 convolution applied to q, k, and v
        self.q_conv = LightweightConv(self.embed_dim, kernel_size=3, padding=1)  # ⌊(3-1)/2⌋ = 1
        self.k_conv = LightweightConv(self.embed_dim, kernel_size=3, padding=1)  # ⌊(5-1)/2⌋ = 2
        self.v_conv = LightweightConv(self.embed_dim, kernel_size=3, padding=1)  # ⌊(7-1)/2⌋ = ?

        self.pooling = nn.ModuleDict({
            'none': nn.Identity(),
            '7x7': MixtureOfPoolingExperts((7, 7)),
            '5x5': MixtureOfPoolingExperts((5, 5))
        })

        self.v_adjust = nn.ModuleList([
            DepthwiseSeparableConv2d(head_dim) for _ in range(num_heads)
        ])

        self.norm = nn.LayerNorm(in_channels)
        self.attn_map = None

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x

        x_ln = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        q = self.q_conv(self.q_proj(x_ln)).reshape(B, self.num_heads, self.head_dim, H, W)
        k = self.k_conv(self.k_proj(x_ln)).reshape(B, self.num_heads, self.head_dim, H, W)
        v = self.v_conv(self.v_proj(x_ln))
        v_res = v
        v = v.reshape(B, self.num_heads, self.head_dim, H, W)

        attn_maps = []
        outputs = []

        for i in range(self.num_heads):
            pool_type = self.pool_configs[i]

            qi = q[:, i]  # [B, d, H, W], q is not pooled

            ki = self.pooling[pool_type](k[:, i])  # [B, d, hk, wk]
            vi = self.pooling[pool_type](v[:, i])  # [B, d, hk, wk]

            B_, d, hq, wq = qi.shape
            _, _, hk, wk = ki.shape

            qi_flat = qi.reshape(B_, d, -1).permute(0, 2, 1)  # [B, Lq, d]
            ki_flat = ki.reshape(B_, d, -1)                    # [B, d, Lk]

            attn = torch.bmm(qi_flat, ki_flat) * self.scale
            attn = F.softmax(attn, dim=-1)

            # Save the current head's attention weights
            attn_maps.append(attn.detach())

            vi_flat = vi.reshape(B_, d, -1).permute(0, 2, 1)   # [B, Lk, d]

            out_i = torch.bmm(attn, vi_flat)                   # [B, Lq, d]
            out_i = out_i.permute(0, 2, 1).reshape(B_, d, hq, wq)  

            outputs.append(out_i)

        # Concatenate all head outputs
        out = torch.cat(outputs, dim=1)  # [B, embed_dim, H, W]
        out = out + v_res
        out = self.out_proj(out)

        # Save all heads' attention weights, shape: [num_heads, B, Lq, Lk]
        attn_maps = resize_tensors_to_target(attn_maps, target_size=(100, 100))

        self.attn_map = torch.stack(attn_maps, dim=0)  

        return out, self.attn_map  # Residual connection


class MultiScaleSelfAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads=4, initial_alpha=0.5):
        super().__init__()
        self.pre_norm = nn.LayerNorm(dim)
        self.conv_1x1_pre = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.attention = MultiScaleSelfAttention(dim, num_heads)
        self.post_norm = nn.LayerNorm(dim)
        self.conv_1x1_post = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.attn_map = None

    def forward(self, x):
        x_norm = self.pre_norm(x.flatten(2).transpose(1, 2)).transpose(1, 2).view(x.size())
        attn_out, self.attn_map = self.attention(x_norm)
        x = x + attn_out
        x_norm = self.post_norm(x.flatten(2).transpose(1, 2)).transpose(1, 2).view(x.size())
        x_out = self.conv_1x1_post(x_norm)
        return x + x_out
