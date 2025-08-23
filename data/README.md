# Data Directory

This folder is structured according to the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) layout.

- `raw/`: immutable raw dumps.
- `interim/`: cleaned but not yet packed.
- `processed/`: tokenized and deduplicated.
- `external/`: third party datasets.
- `out/`: final memmap binaries (`train.bin`, `train.idx`, etc.).

All large files are tracked with DVC, not git.
