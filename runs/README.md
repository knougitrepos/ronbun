# Experiment runs

New LFW and SurvFace experiment runs are written under
`lfw/YYYY/MM/DD/` and `survface/YYYY/MM/DD/` respectively and are not committed.
Legacy in-progress LFW runs created before the notebook split may remain directly
under `YYYY/MM/DD/`; the LFW notebooks retain a compatible fallback to
their `active_run.json` pointer.
Each new run receives a daily sequence such as `20260714-R003-<config-hash>`.
The run directory contains an immutable manifest, structured JSONL logs, phase
attempts, artifacts, figures, and models.

New LFW Step 2 runs use the same `RunStore` contract under a readable daily
root such as `runs/lfw_20260727/`. The run directory is placed directly below
that root, and its `active_run.json` selects exactly one incomplete
ArcFace/AdaFace/MagFace workflow. Restarting notebook 00 with the same model
and config reuses an incomplete run even after the calendar date changes; a
different incomplete run is not overwritten. The representative-case stage
marks the run complete, after which it is immutable.

`runs/step2/` is retained for the existing model registry and legacy completed
Step 2 runs. Existing run directories are not moved because their manifests
freeze absolute input paths and checksums.

## Run reset and quarantine

Use `notebooks/common/maintenance/00_selective_cleanup.ipynb` with
`RESET_MODE="complete_run_reset"` when an exact `run_uid` must be removed before
a clean re-run. Run-owned preprocessing, embeddings, PCA/PQ models and
codebooks, Grad-CAM/LOO artifacts, evaluation outputs, figures, and logs are
moved rather than permanently deleted:

```text
runs/database_cleanup/quarantine/<operation>/payload/
```

The same guarded plan also covers run-scoped PostgreSQL rows, result bundles
whose `result_manifest.json` names the run, and `active_run.json` pointers that
still select it. The plan digest and confirmation token cover all of these
targets together. Cleanup audit files and existing quarantine payloads are
never reset recursively.

Shared raw data, common aligned crops and dataset manifests, checkpoints/model
registries, `images` rows, and other runs are not owned by one experimental run
and are always preserved. A quarantine payload can preserve local files, but
the audit JSON does not contain a PostgreSQL row snapshot and cannot restore
deleted embedding vectors.
