import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

from .shared import BackboneRegistry
from .visual_encoder import VisualEncoder

class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim * 2, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def _sinusoidal(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )                                                  
        args  = t[:, None] * freqs[None, :]                
        return torch.cat([args.sin(), args.cos()], dim=-1) 

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self._sinusoidal(t))               

class LearnedPosEmbed2D(nn.Module):
    def __init__(self, n_freq: int, n_time: int, dim: int):
        super().__init__()
        self.freq_emb = nn.Embedding(n_freq, dim)
        self.time_emb = nn.Embedding(n_time, dim)
        self._n_freq  = n_freq
        self._n_time  = n_time

    def forward(self) -> torch.Tensor:
        device = self.freq_emb.weight.device
        freq_idx = torch.arange(self._n_freq, device=device)       
        time_idx = torch.arange(self._n_time, device=device)       
        fe = self.freq_emb(freq_idx)                               
        te = self.time_emb(time_idx)                               
        pos = (fe[:, None, :] + te[None, :, :]).reshape(-1, fe.shape[-1])
        return pos                                                

class VisualProjector(nn.Module):
    def __init__(self, h_visual_dim: int, dim: int, n_frames: int = 51):
        super().__init__()
        self.proj    = nn.Linear(h_visual_dim, dim)
        self.pos_emb = nn.Embedding(n_frames, dim)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # video: (B, frames, h_visual_dim) → (B, frames, hidden_dim)
        B, T, _ = video.shape
        idx = torch.arange(T, device=video.device)
        return self.proj(video) + self.pos_emb(idx)[None] # (B, 51, hidden_dim)

class AdaLN(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.proj = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, N, dim)   cond: (B, cond_dim)
        gamma, beta = self.proj(cond).chunk(2, dim=-1) # each (B, dim)
        return (1 + gamma[:, None]) * self.norm(x) + beta[:, None]

class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
        
class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2) 
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))          

        attn = (q @ k.transpose(-2, -1)) * self.scale  
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class CrossAttention(nn.Module):
    def __init__(self, dim: int, context_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(context_dim, 2 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        q  = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(context)
        k, v = kv.chunk(2, dim=-1)
        Nv = context.shape[1]
        k  = k.reshape(B, Nv, self.num_heads, self.head_dim).transpose(1, 2)
        v  = v.reshape(B, Nv, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale             
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)           
        return self.proj(x)

class DiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        cond_dim: int,
        h_visual_dim: int,
        ffn_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        # adaln -> mh self att
        self.norm1 = AdaLN(dim, cond_dim)
        self.self_attn = SelfAttention(dim, num_heads, dropout)

        # adaln -> mh cross att
        self.norm2 = AdaLN(dim, cond_dim)
        self.cross_attn = CrossAttention(dim, h_visual_dim, num_heads, dropout)

        # adaln -> feed forward
        self.norm3 = AdaLN(dim, cond_dim)
        self.ffn = FeedForward(dim, ffn_mult, dropout)

    def forward(
        self,
        x: torch.Tensor,       
        t_emb: torch.Tensor,   
        visual: torch.Tensor,   
    ) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x, t_emb))
        x = x + self.cross_attn(self.norm2(x, t_emb), visual)
        x = x + self.ffn(self.norm3(x, t_emb))
        return x
        
@BackboneRegistry.register("avss_dit")
class AVSSDiT(nn.Module):
    """
    Audio-Visual Speech Separation DiT with Flow Matching

    Input:
      x: clean speech mel spec (B, 1, 100, 192)
      y: mixture speech mel spec (B, 1, 100, 192)
      t: flow-matching timestep (B,)
      vid: visual features from TCN (B, 51, 128)
    """
    def __init__(
        self,
        in_channels: int = 2, # concat([x, y])
        out_channels: int = 1, # score for x only
        patch_size: int = 4, # patch along each axis → tokens: (100 / 4) x (192 / 4) = 25×48 = 1200
        mel_freq: int = 100, # mel frequency bins
        mel_time: int = 192, # mel time frames
        hidden_dim: int = 384, # transformer hidden dim
        depth: int = 12, # number of DiT blocks
        num_heads: int = 6, # number of attention heads
        visual_dim: int = 128, # visual feature dim from TCN
        visual_frames: int = 51, # visual frames for 2.04 s
        ffn_mult: int = 4,
        dropout: float = 0.0,
        pretrained_talknet: str = None,
        **unused_kwargs
    ):
        super().__init__()

        assert mel_freq % patch_size == 0, \
            f"mel_freq ({mel_freq}) must be divisible by patch_size ({patch_size})"
        assert mel_time % patch_size == 0, \
            f"mel_time ({mel_time}) must be divisible by patch_size ({patch_size})"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.mel_freq = mel_freq
        self.mel_time = mel_time
        n_freq = mel_freq // patch_size # 100 // 4 = 25
        n_time = mel_time // patch_size # 192 // 4 = 48
        self.n_freq = n_freq
        self.n_time = n_time
        n_tokens = n_freq * n_time           # 25 × 48 = 1200
        
        patch_dim = in_channels * patch_size * patch_size
        self.patch_embed = nn.Sequential(
            nn.Linear(patch_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        self.pos_embed = LearnedPosEmbed2D(n_freq, n_time, hidden_dim)
        self.time_embed = TimestepEmbedding(hidden_dim=hidden_dim)
        self.visual_proj = VisualProjector(visual_dim, hidden_dim, visual_frames)
        
        self.blocks = nn.ModuleList([
            DiTBlock(
                dim = hidden_dim,
                num_heads = num_heads,
                cond_dim = hidden_dim,
                h_visual_dim = hidden_dim,          
                ffn_mult = ffn_mult,
                dropout = dropout,
            )
            for _ in range(depth)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        out_patch_dim = out_channels * patch_size * patch_size
        self.final_proj = nn.Linear(hidden_dim, out_patch_dim, bias=True)
        
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

        self.visual_encoder = VisualEncoder()
        if pretrained_talknet:
            print("load pretrained visual encoder")
            state_dict = torch.load(pretrained_talknet, map_location="cpu")
            new_state_dict = {
                k.replace("model.", "", 1): v
                for k, v in state_dict.items()
                if k.startswith("model.")
            }
            missing, unexpected = self.visual_encoder.load_state_dict(
                new_state_dict,
                strict=False
            )

    def patchify(self, z: torch.Tensor) -> torch.Tensor:
        """
        (B, C, H, W) → (B, N, C × p × p)
        where N = (H/p) × (W/p)
        """
        p = self.patch_size
        # rearrange: split H and W into (n_freq, p) and (n_time, p)
        return rearrange(
            z,
            "b c (nf pf) (nt pt) -> b (nf nt) (c pf pt)",
            pf=p, pt=p,
        )

    def unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        (B, N, out_channels × p × p) → (B, out_channels, H, W)
        """
        p  = self.patch_size
        nf = self.n_freq
        nt = self.n_time
        return rearrange(
            tokens,
            "b (nf nt) (c pf pt) -> b c (nf pf) (nt pt)",
            nf=nf, nt=nt, pf=p, pt=p, c=self.out_channels,
        )
        
    def forward(self, x, t, context):

        context = self.visual_encoder(context) # visual features: (B, 51, 128) 
        visual_tokens = self.visual_proj(context) # (B, 51, dim)
        
        B = x.shape[0]

        tokens = self.patchify(x) # (B, N, patch_dim): N=12000
        tokens = self.patch_embed(tokens) # (B, N, dim)
        
        pos = self.pos_embed().to(tokens.device) # (N, dim)
        tokens = tokens + pos[None] # (B, N, dim)

        t_emb = self.time_embed(t) # (B, cond_dim)
        
        for block in self.blocks:
             tokens = block(tokens, t_emb, visual_tokens) # (B, N, dim)

        tokens = self.final_norm(tokens) # (B, N, dim)
        tokens = self.final_proj(tokens) # (B, N, out_patch_dim)
        score = self.unpatchify(tokens) # (B, 1, 100, 192)

        return score

def AVSSDiT_S(**kwargs):
    return AVSSDiT(depth=12, hidden_dim=384, num_heads=6, **kwargs)

def AVSSDiT_B(**kwargs):
    return AVSSDiT(depth=12, hidden_dim=512, num_heads=8, **kwargs)

def AVSSDiT_L(**kwargs):
    return AVSSDiT(depth=12, hidden_dim=768, num_heads=12, **kwargs)
