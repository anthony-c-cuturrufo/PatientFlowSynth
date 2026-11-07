# PatientFlowSynth

Build and evaluate FHIR-agent tasks on matched patient charts. PatientFlowSynth
fixes a typed program and question, edits the chart into four evidence states,
and replays the program to verify the expected answers and evidence.

Code accompanying **PatientFlowSynth: Evaluating FHIR Agents with Matched Patient
Charts**, accepted to the ML4H 2026 Proceedings Track.

## Quick Start

Requires Python 3.11 or newer. From a clone of this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

patientflowsynth compile examples/demo_patient.json work/demo/release --seed 31
patientflowsynth verify work/demo/release
patientflowsynth prepare work/demo/release work/demo/runtime
patientflowsynth run work/demo/release work/demo/runtime work/demo/episodes.jsonl \
  --adapter reference --limit 8
patientflowsynth summarize work/demo/release work/demo/runtime work/demo/episodes.jsonl
```

The example patient is entirely fabricated. The commands run locally and need no
API key. **The reference adapter is an oracle diagnostic, not a model baseline:**
it uses the expected program and answer to check the pipeline. Its episodes are
marked `diagnostic_only`.

The demo produces 80 matched tasks and 32 chart variants for one patient family.
Only eight episodes are executed above. Use a new output file and omit `--limit`
to run the complete demo. Existing outputs are not overwritten.

For a locked development environment, use `uv sync --extra dev --locked` and
prefix the commands with `uv run` instead.

## Using Your Own Data or Agent

`compile` accepts a local FHIR Bundle JSON file or a directory of Bundle JSON
files. Each bundle must describe one patient and contain the records needed for
an eligible task family. Not every bundle supplies eligible evidence. The
compiler reports the actual admitted census; it does not recreate the paper's
patient selection automatically.

The preserved compiler uses `synthea`/`Synthea` as legacy dataset labels, including
for the hand-fabricated demo. Those labels do not verify how an input was created.
This CLI is intended for synthetic inputs; it is not a clinical-data deidentifier.

Run an adapter through a zero-argument Python factory:

```bash
patientflowsynth run work/demo/release work/demo/runtime work/demo/baseline.jsonl \
  --adapter my_agent:make_agent --limit 8
```

Here `my_agent` is your own installed or importable Python module, not a bundled
model. Its `make_agent()` factory must return an `AgentAdapter`. The agent receives
the question, public tool contracts,
and tool responses—not the evaluator's reference answer or program. See
[the runtime guide](docs/runtime.md) for the interface and scoring.

Custom adapters are trusted Python code, not sandboxed plugins. A network-backed
adapter can transmit chart contents and incur charges. Configure data permissions,
provider settings, and spending limits before using one. No hosted experiments
run automatically.

## What This Release Contains

- The chart compiler, typed programs, FHIR normalization and tools, replay verifier,
  evidence/search scoring, and episode runner.
- Depth-one/two matched-chart generation and a depth-three program builder.
- A fabricated demo, offline regression tests, and a five-command CLI.

This is a **source-code release**, not the frozen 18,000-task benchmark download
or a complete reproduction package for every paper experiment. It does not
contain the paper's source-patient selection, held-out test data, clinical-review
records, model traces, or MIMIC patient data. The depth-three builder is available
as a Python API; the CLI does not reconstruct the paper's depth-three cohort.
Historical provider/retry scripts and experiment-specific orchestration are not
part of this minimal release.

Generated releases contain evaluator-only answers, programs, and evidence. Keep
them private when evaluating agents. The included `.gitignore` excludes the
recommended `work/`, `data/`, and `artifacts/` output locations, but is not a
substitute for reviewing files before publication. This software is for research,
not clinical decision-making.

## Development

```bash
python -m pytest
ruff check .
```

Tests use fabricated records and offline test adapters. They require no credentials
or external patient data. The internal package remains `patient_tool_gym` to keep
the compiler's imports and serialized schemas stable.

## Citation and License

Citation metadata is in [CITATION.cff](CITATION.cff). Code is licensed under
[Apache-2.0](LICENSE). The code license does not grant rights to redistribute
third-party patient data or terminology datasets.
