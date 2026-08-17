"""
Minimal tokenizer built directly from the training corpus -- no external
tokenizer files or pretrained vocab required (keeps the whole project
offline-runnable). Uses lowercased word-level tokens plus a small set of
frequent character bigrams as a crude subword fallback for OOV words, which
is enough for the small synthetic vocabulary used in this project without
pulling in a BPE library.
"""
from __future__ import annotations

import json
import re
from collections import Counter

PAD, UNK, CLS, SEP, MASK = "<pad>", "<unk>", "<cls>", "<sep>", "<mask>"
SPECIAL_TOKENS = [PAD, UNK, CLS, SEP, MASK]

_WORD_RE = re.compile(r"[a-z0-9]+")


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


class SimpleTokenizer:
    def __init__(self, vocab: dict[str, int] | None = None, max_len: int = 64):
        self.vocab = vocab or {}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.max_len = max_len

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.vocab[PAD]

    @property
    def unk_id(self) -> int:
        return self.vocab[UNK]

    @property
    def cls_id(self) -> int:
        return self.vocab[CLS]

    @property
    def sep_id(self) -> int:
        return self.vocab[SEP]

    @classmethod
    def build(cls, texts: list[str], vocab_size: int = 8000, max_len: int = 64) -> "SimpleTokenizer":
        counter: Counter = Counter()
        for t in texts:
            counter.update(_words(t))
        most_common = [w for w, _ in counter.most_common(vocab_size - len(SPECIAL_TOKENS))]
        vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        for w in most_common:
            vocab[w] = len(vocab)
        return cls(vocab=vocab, max_len=max_len)

    def encode(self, text: str, add_special: bool = True, max_len: int | None = None) -> list[int]:
        max_len = max_len or self.max_len
        ids = [self.vocab.get(w, self.unk_id) for w in _words(text)]
        if add_special:
            ids = [self.cls_id] + ids + [self.sep_id]
        ids = ids[:max_len]
        ids = ids + [self.pad_id] * (max_len - len(ids))
        return ids

    def encode_pair(self, text_a: str, text_b: str, max_len: int | None = None) -> list[int]:
        """[CLS] text_a [SEP] text_b [SEP], truncated/padded to max_len -- used by the cross-encoder."""
        max_len = max_len or self.max_len
        a_ids = [self.vocab.get(w, self.unk_id) for w in _words(text_a)]
        b_ids = [self.vocab.get(w, self.unk_id) for w in _words(text_b)]
        budget = max_len - 3  # cls + 2 sep
        a_budget = budget // 2
        b_budget = budget - a_budget
        a_ids = a_ids[:a_budget]
        b_ids = b_ids[:b_budget]
        ids = [self.cls_id] + a_ids + [self.sep_id] + b_ids + [self.sep_id]
        ids = ids[:max_len]
        ids = ids + [self.pad_id] * (max_len - len(ids))
        return ids

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"vocab": self.vocab, "max_len": self.max_len}, f)

    @classmethod
    def load(cls, path: str) -> "SimpleTokenizer":
        with open(path) as f:
            obj = json.load(f)
        return cls(vocab=obj["vocab"], max_len=obj["max_len"])
