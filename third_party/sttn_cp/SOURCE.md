Upstream: https://github.com/kathiresan-jayabalan/sttn-cp-ccfd

Pinned commit: `e5d3657d826ca2e42d6b2489e0577afc45f773df` (2026-09-08).

`src/financial_supervised_ad/recent2026/sttn_architecture.py` preserves upstream
`src/architecture.py` verbatim, preceded only by three attribution comments.
Apache-2.0 LICENSE and upstream NOTICE are retained here. Training and data
adapters are local implementations; they use the common repository split,
exclude Time from model features, and tune the final threshold on validation.

Paper: https://doi.org/10.5281/zenodo.19593993 (published 2026-04-15).
