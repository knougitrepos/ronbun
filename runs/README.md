# Experiment runs

New LFW and SurvFace experiment runs are written under
`lfw/YYYY/MM/DD/` and `survface/YYYY/MM/DD/` respectively and are not committed.
Legacy in-progress LFW runs created before the notebook split may remain directly
under `YYYY/MM/DD/`; the LFW notebooks retain a compatible fallback to
their `active_run.json` pointer.
Each new run receives a daily sequence such as `20260714-R003-<config-hash>`.
The run directory contains an immutable manifest, structured JSONL logs, phase
attempts, artifacts, figures, and models.
