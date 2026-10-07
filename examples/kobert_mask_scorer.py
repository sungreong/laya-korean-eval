"""Proof-of-concept dynamic option scorer using KoBERT [MASK] hidden states.

The scorer is randomly initialized. It must be trained on option-ranking examples before its
scores have meaning. This file demonstrates shapes and packing, not a pretrained classifier.
"""
from dataclasses import dataclass
from typing import Dict
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


@dataclass
class Packed:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor
    marker_positions: torch.Tensor
    option_keys: list[str]


class KoBertMaskScorer(nn.Module):
    def __init__(self, model_name: str = 'skt/kobert-base-v1'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        width = self.encoder.config.hidden_size
        self.scorer = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def pack(self, state: str, instruction: str, criteria: Dict[str, str], max_length: int = 512) -> Packed:
        tok = self.tokenizer
        ids = [tok.cls_token_id] + tok.encode(instruction, add_special_tokens=False) + [tok.sep_token_id]
        types = [0] * len(ids)
        state_ids = tok.encode(state, add_special_tokens=False)
        ids += state_ids + [tok.sep_token_id]
        types += [0] * (len(state_ids) + 1)
        markers, keys = [], []
        for key, description in criteria.items():
            markers.append(len(ids)); keys.append(key)
            option = [tok.mask_token_id] + tok.encode(description, add_special_tokens=False) + [tok.sep_token_id]
            ids += option; types += [1] * len(option)
        if len(ids) > max_length:
            raise ValueError(f'Packed sequence has {len(ids)} tokens; KoBERT limit is {max_length}. Shorten state/options.')
        return Packed(torch.tensor([ids]), torch.ones(1, len(ids), dtype=torch.long),
                      torch.tensor([types]), torch.tensor([markers]), keys)

    def forward(self, packed: Packed):
        hidden = self.encoder(input_ids=packed.input_ids, attention_mask=packed.attention_mask,
                              token_type_ids=packed.token_type_ids).last_hidden_state
        index = packed.marker_positions[:, :, None].expand(-1, -1, hidden.size(-1))
        marker_hidden = torch.gather(hidden, 1, index)       # [batch, number_of_options, 768]
        logits = self.scorer(marker_hidden).squeeze(-1)     # [batch, number_of_options]
        return logits


if __name__ == '__main__':
    model = KoBertMaskScorer()
    packed = model.pack('비가 오면 도로 가장자리에 물이 차오릅니다.', '민원의 담당 유형을 고르세요.', {
        'road_damage': '노면이나 보도 파손을 보수하는 요청',
        'road_drainage': '빗물받이 막힘과 도로 배수를 정비하는 요청',
        'street_light': '꺼진 가로등과 도로 조명을 수리하는 요청',
    })
    logits = model(packed)
    print({'shape': list(logits.shape), 'warning': 'Random scorer: train before interpreting scores.'})
