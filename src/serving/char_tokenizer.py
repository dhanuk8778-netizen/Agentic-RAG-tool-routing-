"""
Character-level tokenizer for TinyGPT (src/serving/tiny_transformer.py).
Word-level tokenization (src/data/tokenizer.py) is used for the
retrieval/router models where fixed-length padding is convenient; the
language model instead needs arbitrary-length autoregressive sequences, so
a small, fixed vocabulary (~90 printable characters) sidesteps needing a
variable-length word vocab or an external BPE tokenizer.
"""
from __future__ import annotations

import json

_CHARS = list(" \n\tabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:!?'\"-()/%")
BOS, EOS = "\x02", "\x03"


class CharTokenizer:
    def __init__(self):
        chars = [BOS, EOS] + _CHARS
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.unk_id = self.stoi[" "]

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        ids = [self.stoi.get(c, self.unk_id) for c in text]
        return ([self.bos_id] + ids) if add_bos else ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos.get(i, "") for i in ids if i not in (self.bos_id, self.eos_id))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.stoi, f)
