from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from timm.models.layers import DropPath

from point2vec.modules.transformer import Block, TransformerEncoderOutput


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale=None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self, x_vis: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, num_vis, dim = x_vis.shape
        if x_mask is None:
            qkv = (
                self.qkv(x_vis)
                .reshape(bsz, num_vis, 3, self.num_heads, dim // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]
            vis_attn = (q @ k.transpose(-2, -1)) * self.scale
            vis_attn = self.attn_drop(vis_attn.softmax(dim=-1))
            x_vis = (vis_attn @ v).transpose(1, 2).reshape(bsz, num_vis, dim)
            x_vis = self.proj_drop(self.proj(x_vis))
            return x_vis, None

        num_mask = x_mask.shape[1]

        x = torch.cat([x_vis, x_mask], dim=1)
        qkv = (
            self.qkv(x)
            .reshape(bsz, num_vis + num_mask, 3, self.num_heads, dim // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        vis_attn = (q[:, :, :num_vis] @ k[:, :, :num_vis].transpose(-2, -1)) * self.scale
        vis_attn = self.attn_drop(vis_attn.softmax(dim=-1))
        x_vis = (vis_attn @ v[:, :, :num_vis]).transpose(1, 2).reshape(bsz, num_vis, dim)
        x_vis = self.proj_drop(self.proj(x_vis))

        mask_attn = (q[:, :, num_vis:] @ k[:, :, :num_vis].transpose(-2, -1)) * self.scale
        mask_attn = self.attn_drop(mask_attn.softmax(dim=-1))
        x_mask = (mask_attn @ v[:, :, :num_vis]).transpose(1, 2).reshape(bsz, num_mask, dim)
        x_mask = self.proj_drop(self.proj(x_mask))

        return x_vis, x_mask


class CrossBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale=None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(
        self, x_vis: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if x_mask is None:
            new_vis, _ = self.attn(self.norm1(x_vis), None)
            x_vis = x_vis + self.drop_path(new_vis)
            x_vis = x_vis + self.drop_path(self.mlp(self.norm2(x_vis)))
            return x_vis, None

        new_vis, new_mask = self.attn(self.norm1(x_vis), self.norm1(x_mask))
        assert new_mask is not None
        x_vis = x_vis + self.drop_path(new_vis)
        x_mask = x_mask + self.drop_path(new_mask)
        x_vis = x_vis + self.drop_path(self.mlp(self.norm2(x_vis)))
        x_mask = x_mask + self.drop_path(self.mlp(self.norm2(x_mask)))
        return x_vis, x_mask


class CrossTransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 4,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale=None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float | List[float] = 0.0,
        add_pos_at_every_layer: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                CrossBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_rate[i]
                    if isinstance(drop_path_rate, list)
                    else drop_path_rate,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.add_pos_at_every_layer = add_pos_at_every_layer

    def forward(
        self,
        x_vis: torch.Tensor,
        pos_vis: torch.Tensor,
        x_mask: torch.Tensor,
        pos_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.add_pos_at_every_layer:
            x_vis = x_vis + pos_vis
            x_mask = x_mask + pos_mask
        for block in self.blocks:
            if self.add_pos_at_every_layer:
                x_vis = x_vis + pos_vis
                x_mask = x_mask + pos_mask
            x_vis, x_mask = block(x_vis, x_mask)
        return self.norm(x_vis), self.norm(x_mask)


class HybridCrossTransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale=None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float | List[float] = 0.0,
        add_pos_at_every_layer: bool = False,
        cross_depth: int = 4,
    ) -> None:
        super().__init__()
        if cross_depth <= 0 or cross_depth >= depth:
            raise ValueError("cross_depth must be in [1, depth - 1].")

        self.depth = depth
        self.cross_depth = cross_depth
        self.self_depth = depth - cross_depth
        self.add_pos_at_every_layer = add_pos_at_every_layer

        if isinstance(drop_path_rate, list):
            dpr = drop_path_rate
        else:
            dpr = [drop_path_rate for _ in range(depth)]

        self.self_blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                )
                for i in range(self.self_depth)
            ]
        )
        self.cross_blocks = nn.ModuleList(
            [
                CrossBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[self.self_depth + i],
                )
                for i in range(self.cross_depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        pos_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
        return_attentions: bool = False,
        return_ffns: bool = False,
    ) -> TransformerEncoderOutput:
        hidden_states = [] if return_hidden_states else None
        attentions = [] if return_attentions else None
        ffns = [] if return_ffns else None

        if x_mask is None:
            if not self.add_pos_at_every_layer:
                x = x + pos
            for block in self.self_blocks:
                if self.add_pos_at_every_layer:
                    x = x + pos
                x, attn, ffn = block(x)
                if return_hidden_states:
                    assert hidden_states is not None
                    hidden_states.append(x)
                if return_attentions:
                    assert attentions is not None
                    attentions.append(attn)
                if return_ffns:
                    assert ffns is not None
                    ffns.append(ffn)

            for block in self.cross_blocks:
                if self.add_pos_at_every_layer:
                    x = x + pos
                elif len(self.self_blocks) == 0:
                    x = x + pos
                x, _ = block(x, None)
                if return_hidden_states:
                    assert hidden_states is not None
                    hidden_states.append(x)
            x = self.norm(x)
            cls_hidden_state = x[:, 0] if x.shape[1] > 0 else None
            return TransformerEncoderOutput(x, cls_hidden_state, hidden_states, attentions, ffns)

        if pos_mask is None:
            raise ValueError("pos_mask is required when x_mask is provided.")

        if not self.add_pos_at_every_layer:
            x = x + pos
            x_mask = x_mask + pos_mask

        for block in self.self_blocks:
            if self.add_pos_at_every_layer:
                x = x + pos
            x, attn, ffn = block(x)
            if return_hidden_states:
                assert hidden_states is not None
                hidden_states.append(torch.cat([x, x_mask], dim=1))
            if return_attentions:
                assert attentions is not None
                attentions.append(attn)
            if return_ffns:
                assert ffns is not None
                ffns.append(ffn)

        for block in self.cross_blocks:
            if self.add_pos_at_every_layer:
                x = x + pos
                x_mask = x_mask + pos_mask
            x, x_mask = block(x, x_mask)
            assert x_mask is not None
            if return_hidden_states:
                assert hidden_states is not None
                hidden_states.append(torch.cat([x, x_mask], dim=1))

        x = self.norm(x)
        x_mask = self.norm(x_mask)
        cls_hidden_state = x[:, 0] if x.shape[1] > 0 else None
        return TransformerEncoderOutput(
            torch.cat([x, x_mask], dim=1),
            cls_hidden_state,
            hidden_states,
            attentions,
            ffns,
        )
