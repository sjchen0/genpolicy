import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from einops import rearrange
from torch.nn.functional import scaled_dot_product_attention
from huggingface_hub import PyTorchModelHubMixin
from omegaconf import OmegaConf

from . import rotary


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNormWot(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.bias = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :] + self.bias[None, None, :]


#################################################################################
#                                 Core Model                                    #
#################################################################################


class DDiTBlockWot(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4, dropout=0.1, use_checkpoint=False):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNormWot(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNormWot(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True), nn.GELU(approximate="tanh"), nn.Linear(mlp_ratio * dim, dim, bias=True)
        )
        self.dropout2 = nn.Dropout(dropout)

        self.dropout = dropout

        self.use_checkpoint = use_checkpoint

    def forward(self, x, rotary_cos_sin, seqlens=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, rotary_cos_sin, seqlens)
        else:
            return self._forward(x, rotary_cos_sin, seqlens)

    def _forward(self, x, rotary_cos_sin, seqlens=None):
        batch_size, seq_len = x.shape[0], x.shape[1]

        # attention operation
        x_skip = x
        x = self.norm1(x)

        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin
            qkv = rotary.apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
        qkv = rearrange(qkv, 'b s three h d -> three b h s d')
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]
        x = scaled_dot_product_attention(q, k, v)
        x = rearrange(x, 'b h s d-> b s (h d)', b=batch_size)

        x = x_skip + F.dropout(self.attn_out(x), p=self.dropout, training=self.training)

        # mlp operation
        x = torch.add(x, F.dropout(self.mlp(self.norm2(x)), p=self.dropout, training=self.training))
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        """
        Mode arg: 0 -> use a learned layer, 1 -> use eigenvectors,
        2-> add in eigenvectors, 3 -> use pretrained embedding matrix
        """
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]


class DDitFinalLayerWot(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = LayerNormWot(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x


class RADD(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__()

        # hack to make loading in configs easier
        if type(config) == dict:
            config = OmegaConf.create(config)

        self.config = config

        vocab_size = config.tokens + 1

        self.vocab_embed = EmbeddingLayer(config.model.hidden_size, vocab_size)
        self.rotary_emb = rotary.Rotary(config.model.hidden_size // config.model.n_heads)

        self.blocks = nn.ModuleList(
            [
                DDiTBlockWot(
                    config.model.hidden_size, config.model.n_heads, dropout=config.model.dropout, use_checkpoint=config.model.use_checkpoint
                )
                for _ in range(config.model.n_blocks)
            ]
        )

        self.output_layer = DDitFinalLayerWot(config.model.hidden_size, vocab_size)
        if config.model.dtype == 'float32':
            self.dtype = torch.float32
        elif config.model.dtype == 'float16': # force recasting
            self.dtype = torch.float32
        elif config.model.dtype == 'bfloat16':
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.bfloat16

    def forward(self, indices):

        x = self.vocab_embed(indices)

        rotary_cos_sin = self.rotary_emb(x)

        with torch.cuda.amp.autocast(dtype=self.dtype):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, seqlens=None)

            x = self.output_layer(x)

            x[:, :, :-1] = x[:, :, :-1].log_softmax(dim=-1)

        return x
    
    def forward_with_hidden(self, indices):
        x = self.vocab_embed(indices)

        rotary_cos_sin = self.rotary_emb(x)

        with torch.cuda.amp.autocast(dtype=self.dtype):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, seqlens=None)

            x_out = self.output_layer(x)

            x_out[:, :, :-1] = x_out[:, :, :-1].log_softmax(dim=-1)

        return x_out, x

    def logits(self, indices):

        x = self.vocab_embed(indices)

        rotary_cos_sin = self.rotary_emb(x)

        with torch.cuda.amp.autocast(dtype=self.dtype):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, seqlens=None)

            x = self.output_layer(x)

        return x

class PolicyNet(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__()

        # hack to make loading in configs easier
        if type(config) == dict:
            config = OmegaConf.create(config)

        self.config = config

        input_size = config.model.hidden_size + 1
        mlp_hidden = config.model.hidden_size // 2
        self.mlp = nn.Sequential(
            nn.Linear(input_size, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 2)
        )
        if config.model.dtype == 'float32':
            self.dtype = torch.float32
        elif config.model.dtype == 'float16':
            self.dtype = torch.float16
        elif config.model.dtype == 'bfloat16':
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.bfloat16
        
    def forward(self, hidden_states, log_score, time):
        # hidden_states: (B, L, h), log_score: (B, L, V+1), time: (B,)
        with torch.cuda.amp.autocast(dtype=self.dtype):
            x = torch.cat([hidden_states, time[:,None,None].repeat(1, hidden_states.shape[1], 1)], dim=-1)
            x = self.mlp(x)
            x = F.softmax(x, dim=1)
        return x
    
def rotate_half(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

class RoPEMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = dropout

        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        if attn_mask is not None:
            attn_mask = attn_mask.to(torch.bool)
        B, T, D = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(T, x.device)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]

        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attn_mask is not None:
            attn = attn.masked_fill(~attn_mask, -float("inf"))

        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask[:, None, None, :],
                -float("inf")
            )

        attn = torch.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = RoPEMultiheadAttention(d_model, n_heads, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        h = self.ln1(x)
        x = x + self.attn(h, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x + self.ff(self.ln2(x))
        return x



class PolicyTransformer(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__()
        if type(config) == dict:
            config = OmegaConf.create(config)
        self.config = config
        self.input_size = config.model.hidden_size + 2 # two additional masking embeddings
        self.n_heads = 4
        self.attn_hidden_size = self.input_size // self.n_heads * self.n_heads
        self.ffn_hidden_size = self.input_size * 2

        self.first_layer = nn.Linear(self.input_size, self.attn_hidden_size)

        self.tf = TransformerBlock(
            d_model=self.attn_hidden_size,
            n_heads=self.n_heads,
            d_ff=self.ffn_hidden_size
        )

        self.final_layer = nn.Linear(self.attn_hidden_size, 2)

        if config.model.dtype == 'float32':
            self.dtype = torch.float32
        elif config.model.dtype == 'float16':
            self.dtype = torch.float16
        elif config.model.dtype == 'bfloat16':
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.bfloat16
        
    def forward(self, hidden_states, log_score, time, **kwargs):
        '''
        hidden_states: (B, L, h), 
        log_score: (B, L, V+1), 
        time: (B,), 
        mask_index: (B, L), bool
        prompt_index: (B, L), bool
        '''
        # with torch.cuda.amp.autocast(dtype=self.dtype):
        forward_suppress = kwargs["prompt_index"] + kwargs["mask_index"]
        backward_suppress = kwargs["prompt_index"] + ~kwargs["mask_index"]
        x = torch.cat([
            hidden_states,
            forward_suppress.unsqueeze(-1).to(hidden_states.dtype),
            backward_suppress.unsqueeze(-1).to(hidden_states.dtype),
        ], dim=-1)
        x = self.first_layer(x)
        if "attn_mask" in kwargs:
            x = self.tf(x, attn_mask=kwargs["attn_mask"].to(x.dtype))
        else:
            x = self.tf(x)
        x = self.final_layer(x)
        suppress_mask = torch.cat([forward_suppress.unsqueeze(-1), backward_suppress.unsqueeze(-1)], dim=-1)
        # x = x.masked_fill(suppress_mask, -float("inf"))
        x = F.softmax(x, dim=1)
        return x