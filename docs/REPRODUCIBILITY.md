# Reproducibility checklist

The benchmark protocol is fixed by the implementation and manifests:

- five datasets remain separate during construction and reporting;
- each clean instance receives one eligible perturbation strategy;
- the same strategy is used for its Low, Medium, and High versions;
- all three levels are generated independently from the clean instance;
- non-target fields and the original label are preserved in the generated record;
- task-specific prompts and label parsers are used for inference;
- aggregate results are computed per dataset, with macro-averaging only when an
  overall summary is required.

The default data-generation seed is `42`. Run manifests should be retained locally
for audit purposes, but they must not contain credentials or private endpoints.

The public package includes metadata snapshots in `Data/metadata/` so that expected
counts and policy flags can be checked without publishing corpus text or model
responses.
