# Code layout

`src/ruchi_bench` is the reusable Python package. The main components are:

- `schema/`: unified sample and label representations;
- `data/`: dataset adapters, download helpers, standardization, and corruption
  runners;
- `corruptions/`: the seven registered perturbation strategies;
- `inference/`: prompts, OpenAI-compatible client, request keys, and pipeline;
- `metrics/`: parsers and paired-result metric utilities;
- `tests/`: regression tests for the public implementation.

The scripts in the sibling `Scripts/` directory assume they are run from the
release-package root and use `Code/src` as the import path.
