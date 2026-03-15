"""
transformer_qa.py
─────────────────────────────────────────────────────────────────────────────
A compact GPT-style encoder-decoder transformer (~4.1 M parameters) for
question-answering, built with PyTorch, pandas, and numpy only.

Architecture
  • Custom BPE tokenizer  – ASCII-only (codepoints 32-126), no external libs
  • Encoder-decoder transformer – 3 + 3 layers, d_model = 192
  • Weight tying between input embedding and output projection
  • Teacher forcing during training; autoregressive greedy / nucleus sampling
    at inference

Dependencies
  torch, pandas, numpy  (stdlib otherwise)
─────────────────────────────────────────────────────────────────────────────
"""

import json
import math
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ══════════════════════════════════════════════════════════════════════════════
# 1.  BPE TOKENIZER
# ══════════════════════════════════════════════════════════════════════════════

# Printable US ASCII: space (0x20) through tilde (0x7E)
_ASCII_SAFE = set(chr(i) for i in range(32, 127))

# End-of-word marker appended to the last character of every word token
_EOW = "</w>"


class BPETokenizer:
    """
    Byte-pair encoding tokenizer restricted to printable US ASCII.

    Special token IDs
        PAD = 0   (padding)
        UNK = 1   (unknown)
        BOS = 2   (begin-of-sequence)
        EOS = 3   (end-of-sequence)
    """

    SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3

    def __init__(self) -> None:
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self._merge_map: Dict[Tuple[str, str], str] = {}

    # ── public API ────────────────────────────────────────────────────────────

    def train(self, texts: List[str], vocab_size: int = 5000, verbose: bool = True) -> None:
        """Learn BPE merge rules from *texts*. Safe to call multiple times."""
        texts = [self._to_ascii(t) for t in texts]

        # --- count word frequencies -------------------------------------------
        word_freq: Counter = Counter()
        for text in texts:
            for word in text.lower().split():
                word_freq[word] += 1

        # --- initial character-level segmentation of each word ----------------
        # Every word is stored as a tuple of chars with _EOW appended to the last
        seg: Dict[Tuple[str, ...], int] = {}
        for word, freq in word_freq.items():
            key = tuple(list(word) + [_EOW])
            seg[key] = freq

        # --- seed the vocabulary with all observed characters -----------------
        self.vocab = {}
        for tok in self.SPECIAL_TOKENS:
            self.vocab[tok] = len(self.vocab)
        for chars in seg:
            for ch in chars:
                if ch not in self.vocab:
                    self.vocab[ch] = len(self.vocab)

        self.merges = []
        self._merge_map = {}

        n_merges = vocab_size - len(self.vocab)
        if n_merges <= 0:
            self.id_to_token = {v: k for k, v in self.vocab.items()}
            return

        if verbose:
            print(
                f"[BPE] base vocab = {len(self.vocab)} | "
                f"target = {vocab_size} | merges needed = {n_merges}"
            )

        for i in range(n_merges):
            pair_freq = self._count_pairs(seg)
            if not pair_freq:
                break
            best = max(pair_freq, key=pair_freq.__getitem__)
            merged = best[0] + best[1]
            self.vocab[merged] = len(self.vocab)
            self.merges.append(best)
            self._merge_map[best] = merged
            seg = self._apply_merge(best, merged, seg)

            if verbose and (i + 1) % 500 == 0:
                print(f"  merge {i+1:>5}/{n_merges}  {best!r:30s} → {merged!r}  vocab={len(self.vocab)}")

        self.id_to_token = {v: k for k, v in self.vocab.items()}
        if verbose:
            print(f"[BPE] final vocab size = {len(self.vocab)}")

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Return a list of integer token IDs for *text*."""
        text = self._to_ascii(text).lower()
        ids: List[int] = []
        for word in text.split():
            chars = list(word) + [_EOW]
            chars = self._apply_merges_to_word(chars)
            ids.extend(self.vocab.get(tok, self.UNK_ID) for tok in chars)
        if add_special_tokens:
            ids = [self.BOS_ID] + ids + [self.EOS_ID]
        return ids

    def decode(self, ids: List[int]) -> str:
        """Reconstruct a string from a list of token IDs."""
        skip = {self.PAD_ID, self.BOS_ID, self.EOS_ID, self.UNK_ID}
        tokens = [self.id_to_token.get(i, "") for i in ids if i not in skip]
        return "".join(tokens).replace(_EOW, " ").strip()

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"vocab": self.vocab, "merges": list(self.merges)}, fh)

    def load(self, path: str) -> None:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        self.vocab = data["vocab"]
        self.merges = [tuple(m) for m in data["merges"]]
        self._merge_map = {tuple(m): m[0] + m[1] for m in self.merges}
        self.id_to_token = {v: k for k, v in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _to_ascii(text: str) -> str:
        return "".join(ch for ch in text if ch in _ASCII_SAFE)

    @staticmethod
    def _count_pairs(seg: Dict[Tuple[str, ...], int]) -> Counter:
        pairs: Counter = Counter()
        for word, freq in seg.items():
            for a, b in zip(word, word[1:]):
                pairs[(a, b)] += freq
        return pairs

    @staticmethod
    def _apply_merge(
        pair: Tuple[str, str],
        merged: str,
        seg: Dict[Tuple[str, ...], int],
    ) -> Dict[Tuple[str, ...], int]:
        a, b = pair
        new_seg: Dict[Tuple[str, ...], int] = {}
        for word, freq in seg.items():
            new_word: List[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_seg[tuple(new_word)] = freq
        return new_seg

    def _apply_merges_to_word(self, chars: List[str]) -> List[str]:
        """Apply all learned merge rules to a single word (list of chars)."""
        for pair, merged in self._merge_map.items():
            i = 0
            out: List[str] = []
            while i < len(chars):
                if i < len(chars) - 1 and chars[i] == pair[0] and chars[i + 1] == pair[1]:
                    out.append(merged)
                    i += 2
                else:
                    out.append(chars[i])
                    i += 1
            chars = out
        return chars


# ══════════════════════════════════════════════════════════════════════════════
# 2.  MODEL COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════


class MultiHeadAttention(nn.Module):
    """Scaled dot-product multi-head attention (encoder or decoder variant)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = math.sqrt(self.d_k)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,                    # (B, T_q, D)
        k: torch.Tensor,                    # (B, T_k, D)
        v: torch.Tensor,                    # (B, T_k, D)
        mask: Optional[torch.Tensor] = None,  # broadcastable bool mask
    ) -> torch.Tensor:
        B, T_q = q.shape[:2]
        T_k = k.shape[1]
        H, dk = self.n_heads, self.d_k

        Q = self.W_q(q).view(B, T_q, H, dk).transpose(1, 2)  # (B,H,T_q,dk)
        K = self.W_k(k).view(B, T_k, H, dk).transpose(1, 2)
        V = self.W_v(v).view(B, T_k, H, dk).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B,H,T_q,T_k)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = self.dropout(torch.softmax(scores, dim=-1))

        out = torch.matmul(attn, V)                              # (B,H,T_q,dk)
        out = out.transpose(1, 2).contiguous().view(B, T_q, H * dk)
        return self.W_o(out)


class FeedForward(nn.Module):
    """Position-wise two-layer FFN with GELU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderBlock(nn.Module):
    """Single transformer encoder block (pre-norm)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), self.norm1(x), self.norm1(x), mask))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class DecoderBlock(nn.Module):
    """Single transformer decoder block with causal self-attn + cross-attn (pre-norm)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        mem_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n1 = self.norm1(x)
        x = x + self.drop(self.self_attn(n1, n1, n1, tgt_mask))
        n2 = self.norm2(x)
        x = x + self.drop(self.cross_attn(n2, enc_out, enc_out, mem_mask))
        x = x + self.drop(self.ff(self.norm3(x)))
        return x


class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[:, : x.size(1)])


# ══════════════════════════════════════════════════════════════════════════════
# 3.  FULL MODEL
# ══════════════════════════════════════════════════════════════════════════════


class QATransformer(nn.Module):
    """
    Encoder-decoder transformer for question answering.

    Default hyperparameters produce ≈ 4.1 M trainable parameters:
        vocab_size  = 5 000
        d_model     = 192
        n_heads     = 8  (head size = 24)
        n_enc / dec = 3 + 3 layers
        d_ff        = 768
        max_seq_len = 128

    Weight tying: output projection shares weights with the input embedding,
    saving ~1 M parameters compared to a separate projection matrix.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 192,
        n_heads: int = 8,
        n_enc_layers: int = 3,
        n_dec_layers: int = 3,
        d_ff: int = 768,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = SinusoidalPE(d_model, max_seq_len, dropout)

        self.encoder = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_enc_layers)]
        )
        self.decoder = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_dec_layers)]
        )
        self.enc_norm = nn.LayerNorm(d_model)
        self.dec_norm = nn.LayerNorm(d_model)

        # Output projection – weight-tied to embedding table (no extra params)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=True)
        self.output_proj.weight = self.embed.weight

        self._init_weights()

    # ── mask builders ─────────────────────────────────────────────────────────

    def _src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Padding mask for encoder.  Shape: (B, 1, 1, T_src)  dtype: bool"""
        return (src != self.pad_id).unsqueeze(1).unsqueeze(2)

    def _tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """Causal + padding mask for decoder.  Shape: (B, 1, T_tgt, T_tgt)"""
        T = tgt.size(1)
        pad_mask = (tgt != self.pad_id).unsqueeze(1).unsqueeze(2)       # (B,1,1,T)
        causal = torch.tril(torch.ones(T, T, device=tgt.device, dtype=torch.bool))  # (T,T)
        return pad_mask & causal.unsqueeze(0)

    # ── forward passes ────────────────────────────────────────────────────────

    def encode(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run encoder.  Returns (enc_out, src_mask)."""
        mask = self._src_mask(src)
        x = self.pos_enc(self.embed(src) * math.sqrt(self.d_model))
        for layer in self.encoder:
            x = layer(x, mask)
        return self.enc_norm(x), mask

    def decode(
        self,
        tgt: torch.Tensor,
        enc_out: torch.Tensor,
        enc_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run one decoder pass. Returns decoder hidden states."""
        tgt_mask = self._tgt_mask(tgt)
        x = self.pos_enc(self.embed(tgt) * math.sqrt(self.d_model))
        for layer in self.decoder:
            x = layer(x, enc_out, tgt_mask, enc_mask)
        return self.dec_norm(x)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: (B, T_src) – question token ids
            tgt: (B, T_tgt) – shifted answer token ids (BOS … without EOS)
        Returns:
            logits: (B, T_tgt, vocab_size)
        """
        enc_out, enc_mask = self.encode(src)
        dec_out = self.decode(tgt, enc_out, enc_mask)
        return self.output_proj(dec_out)

    # ── utilities ─────────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_breakdown(self) -> None:
        """Print a per-component parameter count."""
        groups = {
            "Embedding": self.embed,
            "Encoder": self.encoder,
            "Decoder": self.decoder,
            "Layer norms": nn.ModuleList([self.enc_norm, self.dec_norm]),
            "Output proj (tied)": None,   # weight is shared – not double-counted
        }
        total = 0
        print(f"{'Component':<22} {'Parameters':>12}")
        print("─" * 36)
        for name, mod in groups.items():
            if mod is None:
                n = self.output_proj.bias.numel()
            else:
                n = sum(p.numel() for p in mod.parameters())
            print(f"{name:<22} {n:>12,}")
            total += n
        print("─" * 36)
        print(f"{'TOTAL':<22} {total:>12,}")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  DATASET
# ══════════════════════════════════════════════════════════════════════════════


class QADataset(Dataset):
    """
    Wraps a pandas DataFrame with 'Question' and 'Answer' columns.

    Each sample is returned as three tensors of equal length:
        src     – question token ids (padded / truncated to max_src_len)
        tgt_in  – answer token ids starting with BOS (teacher-forcing input)
        tgt_out – answer token ids ending with EOS  (loss target, shifted +1)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: BPETokenizer,
        max_src_len: int = 64,
        max_tgt_len: int = 64,
    ) -> None:
        self.records = self._build(df, tokenizer, max_src_len, max_tgt_len)

    @staticmethod
    def _pad(ids: List[int], length: int) -> List[int]:
        ids = ids[:length]
        return ids + [BPETokenizer.PAD_ID] * (length - len(ids))

    def _build(
        self,
        df: pd.DataFrame,
        tok: BPETokenizer,
        max_src: int,
        max_tgt: int,
    ) -> List[Tuple[List[int], List[int], List[int]]]:
        records = []
        for _, row in df.iterrows():
            src = self._pad(tok.encode(str(row["Question"])), max_src)
            ans = tok.encode(str(row["Answer"]), add_special_tokens=True)
            tgt_in  = self._pad(ans[:-1], max_tgt)   # BOS … (EOS dropped)
            tgt_out = self._pad(ans[1:],  max_tgt)   #    … EOS (BOS dropped)
            records.append((src, tgt_in, tgt_out))
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src, ti, to = self.records[idx]
        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(ti,  dtype=torch.long),
            torch.tensor(to,  dtype=torch.long),
        )


# ══════════════════════════════════════════════════════════════════════════════
# 5.  TRAINER
# ══════════════════════════════════════════════════════════════════════════════


class Trainer:
    """
    Training loop with:
        • AdamW optimiser
        • Linear warmup → cosine decay LR schedule
        • Gradient clipping (norm = 1.0)
        • Optional best-checkpoint saving (by validation loss)
    """

    def __init__(
        self,
        model: QATransformer,
        tokenizer: BPETokenizer,
        lr: float = 3e-4,
        warmup_steps: int = 400,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=1e-2
        )
        self.warmup_steps = warmup_steps
        self._step = 0
        self._total_steps = 1  # updated at train() time

    # ── learning-rate schedule ────────────────────────────────────────────────

    def _lr_lambda(self, step: int) -> float:
        """Linear warmup then cosine decay to 10 % of peak LR."""
        s = step + 1
        if s <= self.warmup_steps:
            return s / max(1, self.warmup_steps)
        prog = (s - self.warmup_steps) / max(1, self._total_steps - self.warmup_steps)
        prog = min(prog, 1.0)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog))

    # ── public API ────────────────────────────────────────────────────────────

    def train(
        self,
        df: pd.DataFrame,
        epochs: int = 20,
        batch_size: int = 32,
        max_src_len: int = 64,
        max_tgt_len: int = 64,
        val_split: float = 0.1,
        checkpoint_path: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        """
        Train the model on *df* (a DataFrame with 'Question' / 'Answer' columns).

        Args:
            df:               Training data.
            epochs:           Number of full passes over the training set.
            batch_size:       Samples per gradient step.
            max_src_len:      Maximum question length in tokens (longer = truncated).
            max_tgt_len:      Maximum answer length in tokens  (longer = truncated).
            val_split:        Fraction of *df* held out for validation (first rows).
            checkpoint_path:  If given, save the best model here.
            verbose:          Print per-epoch metrics.
        """
        val_n = max(1, int(len(df) * val_split))
        df_val = df.iloc[:val_n].reset_index(drop=True)
        df_trn = df.iloc[val_n:].reset_index(drop=True)

        trn_ds = QADataset(df_trn, self.tokenizer, max_src_len, max_tgt_len)
        val_ds = QADataset(df_val, self.tokenizer, max_src_len, max_tgt_len)
        trn_loader = DataLoader(trn_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

        self._total_steps = len(trn_loader) * epochs
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self._lr_lambda)
        pad = BPETokenizer.PAD_ID
        best_val = float("inf")

        if verbose:
            print(
                f"\n[Trainer] device={self.device}  "
                f"train={len(trn_ds)}  val={len(val_ds)}  "
                f"params={self.model.count_parameters():,}\n"
                f"{'Epoch':>6}  {'trn_loss':>9}  {'val_loss':>9}  "
                f"{'lr':>9}  {'time':>7}"
            )
            print("─" * 52)

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # ─ training pass ─
            self.model.train()
            trn_loss = 0.0
            for src, tgt_in, tgt_out in trn_loader:
                src    = src.to(self.device)
                tgt_in = tgt_in.to(self.device)
                tgt_out = tgt_out.to(self.device)

                logits = self.model(src, tgt_in)           # (B, T, V)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_out.reshape(-1),
                    ignore_index=pad,
                )
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                scheduler.step()
                self._step += 1
                trn_loss += loss.item()
            trn_loss /= len(trn_loader)

            # ─ validation pass ─
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for src, tgt_in, tgt_out in val_loader:
                    src    = src.to(self.device)
                    tgt_in = tgt_in.to(self.device)
                    tgt_out = tgt_out.to(self.device)
                    logits = self.model(src, tgt_in)
                    val_loss += F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        tgt_out.reshape(-1),
                        ignore_index=pad,
                    ).item()
            val_loss /= max(len(val_loader), 1)

            if verbose:
                lr_now = self.optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t0
                flag = " ✓" if (checkpoint_path and val_loss < best_val) else ""
                print(
                    f"{epoch:>6}  {trn_loss:>9.4f}  {val_loss:>9.4f}  "
                    f"{lr_now:>9.2e}  {elapsed:>6.1f}s{flag}"
                )

            if checkpoint_path and val_loss < best_val:
                best_val = val_loss
                self.save(checkpoint_path)

        if verbose:
            print("─" * 52)
            print("Training complete.")
            if checkpoint_path:
                print(f"Best checkpoint → {checkpoint_path}  (val_loss={best_val:.4f})")

    def save(self, path: str) -> None:
        """Save model weights and a minimal config dict."""
        torch.save({"model_state": self.model.state_dict()}, path)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  INFERENCE
# ══════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def generate_answer(
    question: str,
    model: QATransformer,
    tokenizer: BPETokenizer,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    device: str = "cpu",
) -> str:
    """
    Generate an answer for *question* using greedy / nucleus sampling.

    Args:
        question:       Raw question string.
        model:          Trained QATransformer.
        tokenizer:      Fitted BPETokenizer (must match model's vocab).
        max_new_tokens: Hard cap on generated tokens.
        temperature:    Softmax temperature (lower = more peaked / deterministic).
        top_k:          Keep only the top-k logits (0 = disabled).
        top_p:          Nucleus sampling – keep smallest set whose cumulative
                        probability ≥ top_p (1.0 = disabled).
        device:         'cpu' or 'cuda'.
    Returns:
        Decoded answer string.
    """
    dev = torch.device(device)
    model.to(dev)
    model.eval()

    # --- encode source question -----------------------------------------------
    src_ids = tokenizer.encode(question)
    if not src_ids:
        return ""
    src = torch.tensor([src_ids], dtype=torch.long, device=dev)

    # --- run encoder once -  reuse for every decoding step --------------------
    enc_out, enc_mask = model.encode(src)

    # --- autoregressive decoding ----------------------------------------------
    dec_ids: List[int] = [BPETokenizer.BOS_ID]

    for _ in range(max_new_tokens):
        tgt = torch.tensor([dec_ids], dtype=torch.long, device=dev)
        dec_out = model.decode(tgt, enc_out, enc_mask)
        logits = model.output_proj(dec_out[:, -1, :])   # (1, V)

        # temperature scaling
        logits = logits / max(float(temperature), 1e-8)

        # top-k filtering
        if top_k > 0:
            k = min(top_k, logits.size(-1))
            threshold = torch.topk(logits, k).values[:, -1, None]
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        # nucleus (top-p) filtering
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # remove tokens where cumulative prob already exceeds top_p
            remove_mask = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[remove_mask] = float("-inf")
            logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        next_id = int(torch.multinomial(probs, num_samples=1).item())

        if next_id == BPETokenizer.EOS_ID:
            break
        dec_ids.append(next_id)

    return tokenizer.decode(dec_ids[1:])   # strip leading BOS


def load_model(
    checkpoint_path: str,
    tokenizer: BPETokenizer,
    d_model: int = 192,
    n_heads: int = 8,
    n_enc_layers: int = 3,
    n_dec_layers: int = 3,
    d_ff: int = 768,
    max_seq_len: int = 128,
    dropout: float = 0.0,       # disable dropout at inference
) -> QATransformer:
    """Reconstruct a QATransformer from a checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = QATransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_enc_layers=n_enc_layers,
        n_dec_layers=n_dec_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 7.  END-TO-END EXAMPLE  (runs when executed directly)
# ══════════════════════════════════════════════════════════════════════════════


def _make_sample_dataframe() -> pd.DataFrame:
    """Small synthetic Q&A corpus for a quick smoke test."""
    pairs = [
        ("What is the capital of France?",          "The capital of France is Paris."),
        ("What is the capital of Germany?",         "The capital of Germany is Berlin."),
        ("What is the capital of Japan?",           "The capital of Japan is Tokyo."),
        ("What is the capital of Brazil?",          "The capital of Brazil is Brasilia."),
        ("What is the capital of Australia?",       "The capital of Australia is Canberra."),
        ("What is 2 plus 2?",                       "2 plus 2 equals 4."),
        ("What is 3 multiplied by 5?",              "3 multiplied by 5 equals 15."),
        ("What is the square root of 9?",           "The square root of 9 is 3."),
        ("Who wrote Hamlet?",                       "Hamlet was written by William Shakespeare."),
        ("Who wrote Romeo and Juliet?",             "Romeo and Juliet was written by William Shakespeare."),
        ("Who painted the Mona Lisa?",              "The Mona Lisa was painted by Leonardo da Vinci."),
        ("Who composed the Fifth Symphony?",        "The Fifth Symphony was composed by Ludwig van Beethoven."),
        ("What color is the sky?",                  "The sky is blue."),
        ("What color is grass?",                    "Grass is green."),
        ("What is the boiling point of water?",     "Water boils at 100 degrees Celsius."),
        ("What is the freezing point of water?",    "Water freezes at 0 degrees Celsius."),
        ("How many days are in a week?",            "There are 7 days in a week."),
        ("How many months are in a year?",          "There are 12 months in a year."),
        ("How many hours are in a day?",            "There are 24 hours in a day."),
        ("What is the largest planet?",             "Jupiter is the largest planet in our solar system."),
        ("What is the smallest planet?",            "Mercury is the smallest planet in our solar system."),
        ("What is the speed of light?",             "The speed of light is approximately 299792458 meters per second."),
        ("How many continents are there?",          "There are 7 continents on Earth."),
        ("How many oceans are there?",              "There are 5 oceans on Earth."),
        ("What is the chemical symbol for water?",  "The chemical symbol for water is H2O."),
        ("What is the chemical symbol for gold?",   "The chemical symbol for gold is Au."),
        ("What is the chemical symbol for oxygen?", "The chemical symbol for oxygen is O."),
        ("What is the tallest mountain?",           "Mount Everest is the tallest mountain on Earth."),
        ("What is the longest river?",              "The Nile is the longest river on Earth."),
        ("What is the largest ocean?",              "The Pacific Ocean is the largest ocean on Earth."),
    ]
    return pd.DataFrame(pairs, columns=["Question", "Answer"])


if __name__ == "__main__":
    TOKENIZER_PATH  = "tokenizer.json"
    CHECKPOINT_PATH = "qa_transformer.pt"

    # ─────────────────────────────────────────────────────────────
    # Step 1 – Build / load sample DataFrame
    # ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1 — Sample Q&A dataset")
    print("=" * 60)
    df = _make_sample_dataframe()
    print(df.to_string(index=False))

    # ─────────────────────────────────────────────────────────────
    # Step 2 – Train tokenizer
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 — Train BPE tokenizer")
    print("=" * 60)
    tokenizer = BPETokenizer()
    all_texts = df["Question"].tolist() + df["Answer"].tolist()
    tokenizer.train(all_texts, vocab_size=2000, verbose=True)
    tokenizer.save(TOKENIZER_PATH)
    print(f"\nTokenizer saved → {TOKENIZER_PATH}")

    # Sanity-check round-trip
    sample_q = "What is the capital of France?"
    enc = tokenizer.encode(sample_q)
    dec = tokenizer.decode(enc)
    print(f"\nRound-trip check")
    print(f"  Original : {sample_q!r}")
    print(f"  Token IDs: {enc}")
    print(f"  Decoded  : {dec!r}")

    # ─────────────────────────────────────────────────────────────
    # Step 3 – Build model
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 — Build model")
    print("=" * 60)
    model = QATransformer(
        vocab_size   = tokenizer.vocab_size,
        d_model      = 192,
        n_heads      = 8,
        n_enc_layers = 3,
        n_dec_layers = 3,
        d_ff         = 768,
        max_seq_len  = 128,
        dropout      = 0.1,
    )
    model.parameter_breakdown()

    # ─────────────────────────────────────────────────────────────
    # Step 4 – Train
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4 — Train")
    print("=" * 60)
    trainer = Trainer(
        model       = model,
        tokenizer   = tokenizer,
        lr          = 3e-4,
        warmup_steps= 30,
    )
    trainer.train(
        df              = df,
        epochs          = 60,
        batch_size      = 8,
        max_src_len     = 32,
        max_tgt_len     = 32,
        val_split       = 0.15,
        checkpoint_path = CHECKPOINT_PATH,
        verbose         = True,
    )

    # ─────────────────────────────────────────────────────────────
    # Step 5 – Inference
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5 — Inference  (CPU, nucleus sampling)")
    print("=" * 60)
    model_inf = load_model(CHECKPOINT_PATH, tokenizer)

    test_questions = [
        "What is the capital of France?",
        "Who wrote Hamlet?",
        "What color is the sky?",
        "How many days are in a week?",
        "What is the largest planet?",
        "What is the chemical symbol for water?",
    ]

    print(f"\n{'Question':<45}  Answer")
    print("─" * 80)
    for q in test_questions:
        ans = generate_answer(
            question       = q,
            model          = model_inf,
            tokenizer      = tokenizer,
            max_new_tokens = 32,
            temperature    = 0.7,
            top_k          = 50,
            top_p          = 0.9,
            device         = "cpu",
        )
        print(f"{q:<45}  {ans}")
