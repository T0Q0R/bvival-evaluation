# Third-party software and data

This source-only release does not bundle dependency wheels or third-party
model weights. Required packages, versions and wheel hashes are specified in
the individual reconstruction recipes. Each dependency retains its own licence;
our MIT licence does not make CatBoost, PyTorch or source datasets MIT.

The wider study used Ian Covert's `dynamic-selection` implementation at commit
`e2b6f7403fdac4d217ac2ec5dea96acd60240b60`:
<https://github.com/iancovert/dynamic-selection>. Its MIT notice names
Copyright (c) 2023 Ian Covert. This release contains GDFS aggregate diagnostics,
not that upstream source or a standalone GDFS training distribution. The study's
adaptation is not a replication of the authors' published performance.

Data source DOIs, exact versions, access and rights qualifications are retained
in the recipe READMEs and manifests. Depositor licence metadata is evidence of
declared terms, not independent adjudication of marketplace/database rights.
Source rows and all row-level derivatives remain excluded. If recipients build
a different package containing upstream source, binaries or data, they must
separately retain the applicable full notices and determine redistribution rights.
