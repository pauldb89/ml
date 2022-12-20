import math
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import Embedding
from torch.nn.functional import F
from torch.nn.utils.rnn import pad_sequence

from wordle.consts import MAX_GUESSES
from wordle.consts import WORD_LENGTH
from wordle.environment import State
from wordle.vocabulary import Vocabulary


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()

        self.to_qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.num_heads = num_heads
        self.scale = 1 / math.sqrt(embed_dim / num_heads)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()
        queries, keys, values = torch.chunk(self.to_qkv(x), chunks=3)

        queries = queries.view(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 1, 3) * self.scale
        keys = keys.view(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        values = values.view(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)

        attn = queries @ keys
        if mask is not None:
            attn_mask = torch.unsqueeze(mask, dim=2) @ torch.unsqueeze(mask, dim=1)
            attn = torch.masked_fill(attn, attn_mask == 0, -np.inf)

        attn = F.softmax(attn, dim=-1)
        return (attn @ values).permute(0, 2, 1, 3).view(batch_size, seq_len, embed_dim)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, mlp_size: int, num_heads: int):
        super().__init__()

        self.embed_dim = embed_dim
        self.mlp_size = mlp_size
        self.num_heads = num_heads

        self.attn_ln = nn.LayerNorm(embed_dim)
        self.attn_layer = MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads)

        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, mlp_size),
            nn.GELU(),
            nn.Linear(mlp_size, embed_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn_layer(x=self.attn_ln(x), mask=mask)
        return x + self.mlp(x)


class TransformerPolicy(nn.Module):
    def __init__(
        self,
        vocabulary: Vocabulary,
        embed_dim: int = 384,
        num_layers: int = 6,
        num_heads: int = 4,
        mlp_size: int = 1536,
    ):
        super().__init__()
        self.vocabulary = vocabulary
        self.embed_dim = embed_dim

        # We need 29 embeddings (26 for lowercase letters and 3 for 0-2 for match states), but we allocate more to
        # keep the code as simple as possible.
        self.embedding = Embedding(num_embeddings=256, embedding_dim=embed_dim)
        self.positional_encodings = nn.Parameter(
            torch.randn(MAX_GUESSES * WORD_LENGTH * 2, embed_dim),
            requires_grad=True,
        )
        self.global_token = nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad=True)

        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(TransformerBlock(embed_dim=embed_dim, mlp_size=mlp_size, num_heads=num_heads))

        self.mlp = nn.Linear(in_features=embed_dim, out_features=len(vocabulary))

    def preprocess(self, input_states: List[State]) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = []
        for input_state in input_states:
            state = "".join(word_guess + match_state for word_guess, match_state in input_state)
            tokens.append(torch.tensor([ord(char) for char in state], device="cuda"))

        tokens = pad_sequence(tokens, batch_first=False)
        mask = tokens != 0
        return tokens, mask

    def logits(self, input_states: List[State]) -> torch.Tensor:
        tokens, mask = self.preprocess(input_states)
        batch_size, seq_length = tokens.size()

        sequence = self.embedding(tokens) + self.position_embeddings[:seq_length, :]

        sequence = torch.cat([self.global_token.expand(batch_size, -1, -1), sequence], dim=1)
        mask = F.pad(mask, pad=(1, 0), mode="constant", value=1)

        for block in self.blocks:
            sequence = block(x=sequence, mask=mask)

        return self.mlp(sequence[:, 0, :])

    def forward(
        self,
        input_states: List[State],
        actions: List[str],
        rewards: List[float],
    ) -> torch.Tensor:
        logits = self.logits(input_states)
        word_ids = torch.tensor([self.vocabulary.get_word_id(word) for word in actions], device="cuda")
        loss = F.cross_entropy(logits, word_ids, reduction="none")
        return torch.mean(torch.tensor(rewards, device="cuda") * loss)

    def predict(self, guesses: State) -> List[str]:
        input_states = [guesses]
        distribution = torch.distributions.Categorical(self.logits(input_states))
        word_ids = distribution.sample().cpu().numpy()
        return [self.vocabulary.get_word(word_id) for word_id in word_ids]
