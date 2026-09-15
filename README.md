# RUChi-Bench public release package

This directory is the intended upload boundary for the public GitHub repository.
It contains the benchmark implementation, tests, configuration guidance, and
auditable metadata for the completed Chinese language-understanding robustness
study.

The Python import namespace is `ruchi_bench`.

## Included

- `Code/src/`: benchmark schema, adapters, corruption strategies, inference client,
  prompts, and metric utilities.
- `Code/tests/`: unit and regression test suite.
- `Scripts/`: active data preparation, perturbation, evaluation, and reporting entry
  points. Historical debugging scripts are intentionally excluded.
- `Data/full_test/`: the authorized full-test benchmark files, including the
  standardized original instances and three independently generated perturbation
  levels.
- `Data/metadata/`: compact manifests containing dataset counts, perturbation policy,
  and audit summaries.
- `docs/`: public-release data-access and reproducibility notes.
- `.env.example`: empty environment-variable template.

## Intentionally excluded

- `Paper/` and `AutoResearchClaw/`;
- model weights and local serving files;
- `.env`, local settings, API keys, passwords, tokens, and private endpoints;
- inference responses, run logs, and result caches;
- training splits and unrelated raw mirrors.

The authorized full-test release contains only the benchmark's five official test
sets and their generated perturbation files. It does not include training data or
unrelated source mirrors. See `Data/README.md` and `docs/DATA_ACCESS.md` for the
dataset scope, attribution, and local file layout.

## Local setup

From this directory, use Python 3.11 or 3.12:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[core,data,api,analysis,dev]"
export PYTHONPATH="$PWD/Code/src"
python -m pytest
```

Copy `.env.example` to `.env` and fill values locally only. Never commit `.env`.
Inference scripts require an OpenAI-compatible endpoint supplied through
environment variables; no endpoint or credential is embedded in this release.

## Data workflow

1. Use the included files under `Data/full_test/` for inference and evaluation.
2. Run `Scripts/audit_full_test_data.py` to verify the included manifests and
   counts.
3. If rebuilding from source files, follow `docs/DATA_ACCESS.md`, place local raw
   files under `Data/raw/`, and run the standardization and perturbation scripts.

The benchmark uses five datasets separately, seven perturbation strategies, and
Low/Medium/High versions generated independently from each original instance.
Perturbed instances inherit the labels of their corresponding original instances.

## Public upload rule

Upload the contents of this directory only. Do not upload its parent project's
`Paper`, `Results`, `Logs`, `Data/raw`, model files, or local environment files.
