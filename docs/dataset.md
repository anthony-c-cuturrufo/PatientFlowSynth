# Paper Data

The compact download contains the original training and development portion of
PatientFlowSynth, with unchanged task definitions, charts, and patient splits.

| Split | Patients | Tasks |
| --- | ---: | ---: |
| Training | 157 | 12,560 |
| Development | 34 | 2,720 |
| Total | 191 | 15,280 |

The 34 test patients and their 2,720 tasks are not included. This download covers
the primary depth-one/two benchmark, not the separate depth-three or external
evaluation cohorts.

## Download and Run

After installing the repository:

```bash
python scripts/download_data.py --output data/paper
patientflowsynth verify data/paper/release
patientflowsynth prepare data/paper/release work/paper/runtime
patientflowsynth run data/paper/release work/paper/runtime work/paper/episodes.jsonl \
  --adapter reference --cohort dev --limit 8
patientflowsynth summarize data/paper/release work/paper/runtime work/paper/episodes.jsonl
```

The downloader verifies the archive's published SHA-256 before extraction and
refuses to overwrite an existing destination. `verify` reconstructs all 6,112 charts
and replays every task program. This full audit is CPU-intensive and can take tens
of minutes; normal runs reconstruct only the charts they use.
The example run uses reference answers to check the pipeline. To evaluate a model,
replace `reference` with your adapter as described in the [runtime guide](runtime.md).

## Compact Format

The package stores 191 base charts and 6,112 edit chains, not thousands of repeated
full charts. It also includes labeled task definitions, family contracts, original
train/dev assignments, and the non-patient dependency records needed for replay.
Runtime charts are reconstructed on demand, using a small in-memory cache.

Each package has its own closed file inventory and checksums. Original task and
chart hashes are checked during reconstruction. The manifest identifies the
frozen source release without copying its private test-set inventory.
Keep evaluator task files and reference answers separate from model inputs.

## Provenance

Patient records were generated with Synthea at commit
`2ab21caa03f8c6a0400c06593e53f1c1b7510abe`. The paper's source cohort used pinned
generation settings followed by admission, patient selection, and chart editing.
The download preserves that selected train/dev data; the
[generation helper](generate_data.md) creates a new cohort instead.

The archive includes the project code license and upstream Synthea license and
notices. Clinical terminology has its own upstream terms. Patients are synthetic;
facility descriptions may originate from upstream provider directories.
