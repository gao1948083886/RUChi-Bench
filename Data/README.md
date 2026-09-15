# Data package

This release contains the authorized full-test benchmark under `full_test/` and
compact manifests under `metadata/`. The full-test package contains the five
datasets separately, their standardized original instances, and the generated
Low/Medium/High perturbation files.

The five datasets are processed independently. The included files use the
following layout:

```text
Data/raw/<dataset>/
Data/full_test/standardized/<dataset>_clean.jsonl
Data/full_test/perturbed/<dataset>/<strategy>/<level>.jsonl
```

The five datasets are never mixed. Each original instance has one assigned
perturbation strategy, and its Low/Medium/High versions are generated directly
from that original instance. See `../docs/DATA_ACCESS.md` for source attribution
and rebuild instructions.
