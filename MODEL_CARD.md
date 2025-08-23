# Model Card: CharLM

## Intended Use
Research and experimentation on byte-level language modeling.

## Architecture
Decoder-only Transformer with RMSNorm, SwiGLU, RoPE, and GQA attention.

## Training Data
Datasets referenced in `DATA_MANIFEST.md` processed into byte sequences.

## Risks & Limitations
Model may reproduce biases present in training data. Not suitable for generating sensitive content without human oversight.

## Evaluation
Primary metric is bits-per-byte on a held-out validation set. Evaluation protocol in `docs/eval_protocol.md`.
