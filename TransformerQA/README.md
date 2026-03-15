# Transformer Model Sizing Reference
### How many parameters for 500K short Q&A training samples?

---

## The Key Variables

**Effective tokens in your dataset:**
- 500K pairs × ~2 sequences × ~6 words avg × ~1.5 tokens/word (BPE on short words) ≈ **~9M tokens total**
- If you train for 20 epochs, the model sees ~180M token-instances

**Task complexity:**
- Short 2–10 word sequences = low syntactic complexity
- The pattern space is much smaller than open-domain language modeling
- This pulls the required parameter count *down* significantly

---

## Rough Sizing Guidance

| Parameters | Verdict for your setup |
|---|---|
| < 500K | Too small — embedding table alone eats most of this |
| **500K – 1.5M** | **Sweet spot for simple/formulaic Q&A patterns** |
| **1.5M – 3M** | **Sweet spot for moderate answer diversity** |
| 3M – 5M (current) | Slight overkill — workable but needs stronger regularization |
| > 5M | Wasteful; harder to train well on this data volume |

---

## The Dominant Factor: Answer Diversity

The right size depends more on *how many distinct answer patterns exist* than on raw sample count:

**Formulaic answers** ("The capital of X is Y.", "X was written by Y.") → the model is essentially learning a lookup table with grammatical structure. **~500K–1M params** is plenty. The repetitive structure means low effective information content per sample.

**Varied / generative answers** (explanations, multi-clause responses) → the model needs to learn compositional structure. **1.5M–3M params** is better.

---

## Concrete Recommended Architecture

Scaling down from the current model:

```python
# ~1.5M parameters — good for most short-answer Q&A at 500K samples
model = QATransformer(
    vocab_size   = 3000,   # smaller vocab is fine for short/simple text
    d_model      = 128,
    n_heads      = 8,      # head size = 16
    n_enc_layers = 3,
    n_dec_layers = 3,
    d_ff         = 512,
    max_seq_len  = 32,     # plenty for 2-10 word sequences
    dropout      = 0.15,   # slightly higher — still large relative to data
)
```

Approximate parameter breakdown at this size:

| Component | Parameters |
|---|---|
| Embedding (3000 × 128) | 384,000 |
| Encoder (3 blocks) | 395,000 |
| Decoder (3 blocks) | 592,000 |
| Layer norms + biases | 12,000 |
| Output proj bias (tied) | 3,000 |
| **Total** | **~1.39M** |

---

## Two Other Levers That Matter as Much as Parameter Count

**Vocab size** — with 2–10 word sentences, a vocab of 2,000–3,000 is almost certainly sufficient and keeps the embedding table small. The current default of 5,000 is oversized for short text.

**`max_seq_len`** — dropping from 128 → 32 reduces memory and speeds up attention quadratically. At 2–10 words, 32 tokens comfortably covers every sample with room to spare.

---

## Bottom Line

With 500K short-sequence pairs, **1–2M parameters is the practical sweet spot**. The current 4.1M model will still *work* but you'll need higher dropout and more careful LR tuning to prevent overfitting. The smaller model will generalize more naturally, train faster, and be less sensitive to hyperparameters.
