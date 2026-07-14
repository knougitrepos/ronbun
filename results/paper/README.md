# Paper results

Notebook 05 automatically exports compact, publication-ready tables and figures
to `<dataset>/<run_id>/`. Each bundle includes `result_manifest.json` with the
source run, phase, attempt, file size, and SHA-256 checksum.

Raw provenance remains under `runs/` and is ignored by Git. Do not copy files
here manually or edit an exported bundle in place; create a new run instead.
