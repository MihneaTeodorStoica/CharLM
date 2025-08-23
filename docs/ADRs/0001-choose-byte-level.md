# ADR 0001: Choose byte-level vocabulary

## Status
Accepted

## Context
Byte-level models avoid dependency on tokenizers and handle any text or binary data uniformly.

## Decision
Use a fixed 256 byte vocabulary with an optional separator byte for document boundaries.

## Consequences
Simplifies preprocessing and multilingual support but increases sequence length compared to BPE tokenizers.
