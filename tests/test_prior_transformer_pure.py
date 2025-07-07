import os
import sys
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.models.stage1_pure_prior_transformer import Stage1_PriorTransformerPure


def _expected_mask(length: int) -> torch.Tensor:
    mask = torch.full((length, length), -10000.0)
    mask.triu_(1)
    return mask[None, ...]


def test_optional_token_mask_shape():
    model = Stage1_PriorTransformerPure(
        num_layers=0,
        embedding_dim=4,
        num_attention_heads=1,
        attention_head_dim=4,
    )
    proj = torch.zeros(1, 1, 4)
    cond1 = torch.zeros(1, 1, 4)
    model(proj_embedding=proj, encoder_hidden_states=cond1)

    seq_len = 3  # cond1 + proj + prd
    mask = model.causal_attention_mask[:, :seq_len, :seq_len]
    assert mask.shape == (1, seq_len, seq_len)
    assert torch.allclose(mask, _expected_mask(seq_len))


def test_mask_rebuild_larger_seq():
    model = Stage1_PriorTransformerPure(
        num_layers=0,
        embedding_dim=4,
        num_attention_heads=1,
        attention_head_dim=4,
    )
    proj = torch.zeros(1, 1, 4)
    cond1 = torch.zeros(1, 4, 4)
    cond2 = torch.zeros(1, 2, 4)
    model(
        proj_embedding=proj,
        encoder_hidden_states=cond1,
        encoder_hidden_states1=cond2,
    )
    seq_len = cond1.size(1) + cond2.size(1) + 2
    assert model.causal_attention_mask.shape[-1] == seq_len
    mask = model.causal_attention_mask[0]
    expected = _expected_mask(seq_len)[0]
    assert torch.allclose(mask, expected)
