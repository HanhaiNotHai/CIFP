# CIFP architecture audit

## Classifier access boundary

The fake classifier is `FakeClassifier(Linear(128,1))`. Its public `forward` accepts exactly one
tensor named `z_for` and validates its final dimension. The complete upstream access graph is:

```text
image -> forensic DINOv3 true patch tokens F
      -> local projected tokens H
      -> cosine/top-k assignment A and normalized learned dictionary D
      -> mean_usage, max_usage [, projected local cooccurrence]
      -> composition projector -> z_for -> fake classifier -> fake logit
```

Thus the classifier can indirectly access only primitive assignment use and optional assignment
cooccurrence. It cannot access raw image, DINOv3 CLS/register tokens, raw patch tokens, semantic
embedding, content environment, generator/source metadata, labels, or teacher outputs. The
environment classifier consumes `z_for` independently and never feeds a value back into the fake
classifier.

## Semantic teacher location

`FrozenSemanticTeacher` exists only in `cifp.environments.teacher` and is instantiated only by
`cifp.cli.extract_semantics`. It is frozen, kept in evaluation mode, and its extraction method uses
`torch.inference_mode`. `CIFP`, `build_model`, training forward, evaluation, and inference contain
no teacher member or teacher import. The teacher is consequently absent from CIFP checkpoints.

## Training versus inference

Training evaluates the student, projector, dictionary, composition projector, fake classifier,
and the training-only `environment_classifier` behind GRL. It consumes labels and assigned
`content_env` only in the loss. `CIFP.inference(images)` supplies no GRL coefficient, skips the
environment classifier, and returns only fake logits. Neither path reads an offline semantic store
at runtime.

## Absence of normal-reference and residual paths

The model output schema is limited to fake logits, `z_for`, assignments, primitive usage, dynamic
grid, optional cooccurrence, and optional environment logits. It has no predicted-normal feature,
reconstructed image, residual feature, reconstruction loss, deviation score, anomaly score, real
prototype, or nearest-real decision. Dictionary rows are shared compositional directions and are
never used to reconstruct or subtract from a token. Tests assert these output and loss boundaries.

## Token and shape boundaries

Token slicing reads `hidden_size`, `patch_size`, and `num_register_tokens` from loaded config. It
requires exactly `1 + registers + grid_h*grid_w` final tokens. At 128 square pixels and patch size
16 the verified grid is 8x8, but neither patch count nor grid is hardcoded. Cooccurrence constructs
horizontal and vertical edges from this runtime grid and verifies `N = grid_h*grid_w`.

## SemTrace independence

There is no SemTrace dependency in `pyproject.toml`, no SemTrace import, no CIFP runtime reference
to `/data/zhy/SemTrace`, and no copied normal predictor, reconstruction, or residual module.
SemTrace was inspected read-only only for generic project organization. CIFP uses its own src-layout
package, configuration loader, manifest builders, DDP initialization, checkpoint schema, metrics,
logging, and tests.

## Automated enforcement

- `test_classifier_access` checks the one-tensor classifier boundary.
- `test_no_reconstruction_path` checks prohibited output/loss names.
- `test_frozen_teacher` checks teacher freezing and absence from CIFP.
- Patch extraction, sparse assignment, cooccurrence, GRL, checkpoint, metrics, and end-to-end tests
  enforce the remaining device/shape/gradient behavior.
