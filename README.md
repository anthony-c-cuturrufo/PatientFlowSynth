# PatientFlowSynth

PatientFlowSynth tests AI agents that read FHIR patient records. It creates
matched versions of a patient chart with supporting, refuting, missing, or
conflicting evidence, then checks the agent's answers, citations, and searches.

Code accompanying **PatientFlowSynth: Evaluating FHIR Agents with Matched Patient
Charts**, accepted to the ML4H 2026 Proceedings Track.

This repository contains the source code, a runnable demo, and tests. It includes
chart generation, verification, scoring, and agent execution. The CLI generates
depth-one/two tasks; the depth-three builder is available through the Python API.

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

The fabricated demo creates 80 tasks across 32 chart variants. The commands above
check eight tasks using reference answers, with no API key needed. To run all 80,
omit `--limit` and choose a new output file.

## Using Your Own Data or Agent

`compile` accepts a FHIR Bundle JSON file or a directory of single-patient bundles.
It builds tasks from the eligible records it finds.

To use your own agent:

```bash
patientflowsynth run work/demo/release work/demo/runtime work/demo/baseline.jsonl \
  --adapter my_agent:make_agent --limit 8
```

Define `make_agent()` in an importable `my_agent` module to return an
`AgentAdapter`. See the [runtime guide](docs/runtime.md) for the agent interface,
tool calls, and scoring details.

## Development

```bash
python -m pytest
ruff check .
```

Tests run offline. For locked dependencies, use `uv sync --extra dev --locked`
and prefix commands with `uv run`.

## Citation and License

Citation metadata is in [CITATION.cff](CITATION.cff). Code is licensed under
[Apache-2.0](LICENSE).
