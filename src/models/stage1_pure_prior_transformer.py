from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.attention_processor import (
    AttentionProcessor,
    AttnProcessor,
)
from diffusers.models.modeling_utils import ModelMixin


@dataclass
class PriorTransformerOutput(BaseOutput):
    """Return type for the pure prior."""
    predicted_image_embedding: torch.FloatTensor


class Stage1_PriorTransformerPure(ModelMixin, ConfigMixin):
    """Pure‑Transformer prior: directly regresses CLIP image embeddings.

    ‑ **No diffusion / timestep / noise tokens**
    ‑ Input sequence = [cond_tokens, proj_token, <prd>] where
        • cond_tokens  – e.g. agnostic person embedding (N×E)
        • proj_token   – cloth image embedding (1×E)
        • <prd>        – learnable output token (1×E)
    The last token after Transformer layers is mapped back to CLIP dim.
    """

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 32,
        attention_head_dim: int = 64,
        num_layers: int = 20,
        embedding_dim: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        # projections for input tokens
        self.embedding_proj = nn.Linear(embedding_dim, inner_dim)
        self.encoder_hidden_states_proj = nn.Linear(embedding_dim, inner_dim)
        self.encoder_hidden_states_proj1 = nn.Linear(embedding_dim, inner_dim)

        # learnable output token
        self.prd_embedding = nn.Parameter(torch.zeros(1, 1, inner_dim))
        # positional embedding (proj_token + prd_token = 2 extra positions)
        self.positional_embedding = nn.Parameter(torch.zeros(1, 2, inner_dim))

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    activation_fn="gelu",
                    attention_bias=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = nn.LayerNorm(inner_dim)
        self.proj_to_clip_embeddings = nn.Linear(inner_dim, embedding_dim)

        self.clip_mean = torch.tensor(-0.016)
        self.clip_std = torch.tensor(0.415)

    # ---------------------------------------------------------------------
    # optional helpers (LoRA / attention processor)
    # ---------------------------------------------------------------------
    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        processors = {}

        def walk(name: str, module: nn.Module):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor
            for sub_name, child in module.named_children():
                walk(f"{name}.{sub_name}", child)

        for n, m in self.named_children():
            walk(n, m)
        return processors

    def set_default_attn_processor(self):
        self.set_attn_processor(AttnProcessor())

    # ---------------------------------------------------------------------
    # forward
    # ---------------------------------------------------------------------
    def forward(
        self,
        proj_embedding: torch.FloatTensor,                # (bs, E)
        encoder_hidden_states: torch.FloatTensor,         # (bs, N, E)
        encoder_hidden_states1: Optional[torch.FloatTensor] = None,  # optional extra conds
        attention_mask: Optional[torch.BoolTensor] = None,
        return_dict: bool = True,
    ):
        bs = proj_embedding.shape[0]
        inner_proj = self.embedding_proj(proj_embedding).unsqueeze(1)  # (bs,1,D)
        cond1 = self.encoder_hidden_states_proj(encoder_hidden_states)  # (bs,N,D)

        tokens = [cond1]
        if encoder_hidden_states1 is not None:
            cond2 = self.encoder_hidden_states_proj1(encoder_hidden_states1)
            tokens.append(cond2)
        tokens.append(inner_proj)

        prd_tok = self.prd_embedding.to(cond1.dtype).expand(bs, -1, -1)
        tokens.append(prd_tok)

        hidden_states = torch.cat(tokens, dim=1)  # (bs, seq, D)

        # add(positional_embedding) — broadcast / pad as needed
        pe = F.pad(
            self.positional_embedding,
            (0, 0, 0, hidden_states.size(1) - self.positional_embedding.size(1)),
            value=0.0,
        )
        hidden_states = hidden_states + pe.to(hidden_states.dtype)

        # optional mask → (-10000) for tokens to ignore
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask[:, None, None, :]
        else:
            attention_mask = None

        for blk in self.transformer_blocks:
            hidden_states = blk(hidden_states, attention_mask=attention_mask)

        hidden_states = self.norm_out(hidden_states)
        out_vec = hidden_states[:, -1]            # prd token
        pred_embed = self.proj_to_clip_embeddings(out_vec)

        if not return_dict:
            return (pred_embed,)
        return PriorTransformerOutput(predicted_image_embedding=pred_embed)

    # (optional) de‑norm helper ------------------------------------------------
    def post_process_latents(self, prior_latents: torch.FloatTensor):
        return (prior_latents * self.clip_std) + self.clip_mean
