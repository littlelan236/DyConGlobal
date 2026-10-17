# NAS image runtime

`nas.py` implements content-addressed OCI blobs under the configured NAS root,
short journal transactions, crash-safe image leases and four cross-process load slots.
Task execution imports images on demand and releases owned images after container removal.
`images-check` reads manifests/blobs without loading them. A live four-image Docker
smoke is available through `python -m runtime.scripts.smoke_images`.
