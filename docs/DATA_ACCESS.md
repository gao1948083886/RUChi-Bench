# Dataset access and redistribution notes

RUChi-Bench uses the official test split of five Chinese datasets. This release
includes the authorized standardized and perturbed full-test files under
`Data/full_test/`; the source access notes below document provenance and how to
rebuild the files when needed.

| Dataset | Local acquisition route | Public-package note |
|---|---|---|
| PAWS-X Chinese | Official PAWS-X source; see `ruchi_bench.data.download.datasets` | Verify the current source terms before redistribution. |
| XNLI Chinese | Official XNLI release; select the Chinese test rows | The source carries non-commercial attribution conditions. |
| C3 | Official C3 repository test files | Check the repository license before redistribution. |
| LCQMC | Approved access route from the HIT-SZ source | Do not redistribute raw or derived text without permission. |
| ASAP | Official Meituan-Dianping repository test file | Verify the repository license and attribution requirements. |

The included downloader covers the publicly addressable sources for PAWS-X, XNLI,
C3, and ASAP. The released full-test files are the authoritative inputs for
reproducing the benchmark results. LCQMC source acquisition remains subject to
the approved access route; the released benchmark files may be used directly.

This file is an operational data-access note, not a legal opinion. Before a public
release containing any text, re-check each dataset's current license and the terms
of the specific source copy.
