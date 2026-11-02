# Offline steering datasets

Chinese operation guide: [Consensus Steering Dataset 与 LDA](../docs/STEERING-DATASET-LDA-CN.md).

`steering` consumes a frozen source-run selection and the corresponding ACKed
capture publication. It labels every verifiable consensus candidate by exact
canonical action agreement with the historical selected action. Estimator
retention and selected-branch status remain provenance fields; they do not
define the binary label.

The package has three immutable artifact categories:

* `steering/datasets/<ID>` stores metadata, source bindings, all candidate
  rows, excluded rows, and source checksums. Hidden tensors remain in their
  published capture location.
* `steering/cache/<HASH>` stores rebuildable per-site float32 memmaps. The
  cache identity includes the dataset receipt, sample order, site inventory,
  and feature-space metadata.
* `steering/outputs/<ID>` stores one supervised LD1 or a two-dimensional
  unsupervised t-SNE/UMAP embedding per selected site, frozen coordinates,
  fit descriptors, reports, plots, and a receipt.

The formal entrypoints are `dycon steering-dataset build/check/cache`,
`dycon steering-visualize`, and `dycon artifact verify ... --scope
steering-dataset|steering-analysis`. `steering-visualize --method lda|tsne|umap`
selects the fit method (default LDA). t-SNE/UMAP fit all selected samples without
labels, with optional PCA and explicit seeded parameters; a display cap affects
plots only. Their NPZ descriptors are not reusable out-of-sample reducers.
Visualization is exploratory; coordinates are not steering payloads and redraw
consumes frozen coordinates without refitting, even with the source data offline.
