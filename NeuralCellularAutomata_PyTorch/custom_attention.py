from __future__ import annotations
from typing import Iterable
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import sys

EPSILON = sys.float_info.epsilon
"""A number that is very small, but not 0"""

CLOSE_ENOUGH = 1e-4
"""Small but not vanishingly so value, to determine if two values are almost equal"""

NOT_EPSILON = 1
"""A number that isn't very small, and definitely not 0"""


def swishmax(input: torch.Tensor, dim=0, keepdim=False):
    xexp = input * torch.exp(input - torch.max(input, dim=dim, keepdim=True).values)
    # print(xexp.shape, input.shape, "swishmax")
    out = torch.div(
        xexp,
        (
            torch.sum(
                torch.abs(xexp),
                dim=dim,
                keepdim=True,
            )
            + NOT_EPSILON
        ),
    )
    return out
    # return torch.div(xexp, (torch.sum(torch.abs(xexp), dim=dim, keepdim=True)))


def near_zero(input: torch.Tensor):
    return CLOSE_ENOUGH > torch.max(torch.abs(input))


def near_equal(a: torch.Tensor, b: torch.Tensor):
    return near_zero(a - b)


def make_weight(*dims: int):
    # return Parameter(torch.rand(dims) * ((1 / dims[-2]) ** 0.5))
    return Parameter(F.normalize(torch.randn(dims), dim=-2))


class CustomAttention(nn.Module):
    """One query vector per operation"""

    def __init__(self, heads: int, token_size: int, attention_size: int, compress_size: int) -> None:
        super().__init__()

        # should preserve 0s
        self.key_down = make_weight(heads, token_size, attention_size)

        # should NOT preserve 0s
        self.query_down = make_weight(heads, token_size, attention_size)
        self.query_bias = make_weight(heads, 1, attention_size)

        # should preserve 0s
        self.compress = token_size > compress_size * 2
        if self.compress:
            self.value_down = make_weight(heads, token_size, compress_size)
            self.value_up = make_weight(heads, compress_size, token_size)
        else:
            self.value_over = make_weight(heads,token_size,token_size)

    def forward(
        self,
        key_tokens: torch.Tensor | Iterable[torch.Tensor],
        query_token: torch.Tensor,
    ):
        """query is a [1,d] shape vector\n
        keys is a torch.stack() of 1d vectors (shape of [n,d] for n vectors of d length)\n
        OR an iterable of 1d vectors
        """
        # assert query_token.shape[-2] == 1, "query is not single vector"
        if type(key_tokens) is Iterable:
            key_tokens = torch.stack(list(key_tokens))
        assert type(key_tokens) is torch.Tensor, "iterable not tensor"

        # print("A",key_tokens.shape, self.key_transform.shape)
        # print("B",query_token.shape, self.query_transform.shape)
        # print("C",keys.shape, query.transpose(-2, -1).shape)

        key_tokens = key_tokens.unsqueeze(-3)
        # print(key_tokens.shape, "KT")
        keys = key_tokens @ self.key_down
        # print(keys.shape, "KA")

        query_token = query_token.unsqueeze(-3)

        # print(query_token.shape, "QT")
        query = query_token @ self.query_down
        # print(query.shape, "QA)

        query = query + self.query_bias
        # print(query.shape, "QA+B")

        attention = swishmax(keys @ query.transpose(-2, -1), dim=-2)

        # print(key_tokens.unsqueeze(-2).shape, attention.unsqueeze(-1).shape, "KA")

        values_scaled = key_tokens.unsqueeze(-2) * attention.unsqueeze(-1)
        # print(values_scaled.shape, "VS")

        value_shift = values_scaled.sum(-3)
        # print(value_shift.shape, "VSH")

        if self.compress:
            value_shift @= self.value_down
            value_shift @= self.value_up
        else:
            value_shift @= self.value_over
        # print(value_shift.shape, "VSH")

        value_shift = value_shift.sum(-3)
        # print(value_shift.shape, "VSH")

        # print(query_token.shape,"QT")

        # update = query_token.squeeze(-2).squeeze(-2) + (
        #     value_shift.squeeze(-2).squeeze(-2)
        # )
        update = value_shift.squeeze(-2).squeeze(-2)

        # print(update.shape, "U")

        # update = update / (torch.linalg.vector_norm(update, dim=-1, keepdim=True) + 1)

        return update


class Conv2DAttention(nn.Module):
    # def __init__(self, *args, ) -> None:
    #     super().__init__(*args, )
    def __init__(self, attention_block: CustomAttention) -> None:
        super().__init__()
        self.attention = attention_block
        self.pad = nn.CircularPad2d(3 // 2)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        width = 3

        image = self.pad(image)

        sections = image.unfold(-2, width, 1).unfold(-2, width, 1)
        sections = sections.reshape(list(sections.shape[:-2]) + [-1])
        middle = sections.shape[-1] // 2

        # print(middle)

        # print(sections.shape)

        keys = torch.cat((sections[..., :middle], sections[..., middle + 1 :]), dim=-1)
        # print(keys.shape, "K")

        query = sections[..., middle : middle + 1]
        # print(query.shape, "Q")

        keys = keys.movedim(1, -1)
        query = query.movedim(1, -1)

        # print(keys,query)

        outs = self.attention.forward(keys, query)

        return outs.movedim(-1, 1)
