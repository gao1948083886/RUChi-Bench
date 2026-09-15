# Public scripts

The scripts below are the public execution entry points retained in this package:

- `standardize_full_test.py`: convert locally obtained test files to the unified
  clean-sample schema;
- `generate_full_corruptions_parallel.py`: generate the seven perturbation strategies and three
  independently generated levels;
- `audit_full_test_data.py`: audit counts and corruption records;
- `compute_metrics.py`: compute metrics from locally retained inference records
  against the included full-test data;
- `smoke_test_api.py`: perform a small endpoint smoke test;
- `ministral_transformers_server.py`: optional local Transformers server helper.

Scripts that only reproduce obsolete pilot paths or contain historical debugging
code are not part of the public package.
