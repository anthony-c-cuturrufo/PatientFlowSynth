# Runtime Guide

## Workflow

1. `compile INPUT OUTPUT` normalizes local FHIR Bundles, discovers eligible
   source tasks, and builds matched chart families. The four states are
   supported, refuted, insufficient evidence, and conflicting evidence.
2. `verify RELEASE` checks the release inventory, hashes, permitted edits,
   invariant questions/programs, and reference replay.
3. `prepare RELEASE OUTPUT` materializes the task-to-snapshot bindings used by
   the runner. No model is called.
4. `run RELEASE RUNTIME EPISODES --adapter FACTORY` records bounded tool-use
   episodes, submissions, and scores. `reference` selects the oracle diagnostic;
   `package.module:factory` loads a custom adapter.
5. `summarize RELEASE RUNTIME EPISODES` verifies task bindings and replays scores
   before reporting results. Add `--output report.json` for the complete analysis
   rather than just the console summary. A partial run is not a complete benchmark evaluation.

Use `patientflowsynth COMMAND --help` for options. Use a new output location for
each build or run. Preserve failed episodes when comparing systems, and report
patient counts as well as episode counts: matched tasks are not independent
patient samples.

## Agent Interface

Implement the protocols in `patient_tool_gym.agent_adapters`:

- `AgentAdapter.descriptor` identifies the adapter, model, and configuration.
- `AgentAdapter.start(request, seed=...)` creates a session. `AgentRequest`
  contains a question and public tool specifications, not gold evaluator fields.
- `AgentSession.next(tool_results)` returns an `AdapterTurn` containing tool calls.
  The runner executes permitted tools against the assigned chart and returns
  their results. Submit a final answer with `submit_answer` using its public schema.

Expose your adapter through a zero-argument `make_agent()` factory in an
importable `my_agent` module, then select `--adapter my_agent:make_agent`.
The factory receives no task or reference data. Compositional tasks require the
question's structured branches-and-overall response, not just a scalar truth
label. The legacy `ClinicalRetrievalAgent` class does not implement that response
shape and is not a baseline for this CLI's tasks.
The tool interface is an evaluation boundary, not a security sandbox: an
untrusted adapter must run in a separate isolation layer.

## Scoring

The evaluator keeps label/answer agreement, exact evidence matching, evidence
sufficiency, and required search completion separate. Returning the reference
records alone does not prove a complete search. A submission can contain
sufficient evidence and still fail exact citation matching. Conflict requires
opposing evidence under the benchmark's rules; an absence claim requires the
applicable complete-search check.

The summary's strict endpoint combines these checks. Its names and schemas are
retained from the research implementation. Reference-agent results check that
the construction and execution pipeline agree; they do not measure an agent's
clinical reasoning or retrieval capability.

## Code Layout

| Component | Modules |
| --- | --- |
| FHIR records and tools | `fhir`, `raw_fhir`, `tools` |
| Typed programs and rules | `program`, `lifecycle`, `evidence_contracts` |
| Source-task discovery | `patient_v1`, `patient_p2` |
| Matched-chart construction | `compositional_matrix`, `medical_interventions`, `compositional_release` |
| Reference-path tasks | `depth3_extension` |
| Validation and scoring | `task`, `evidence`, `verifier` |
| Runtime and analysis | `compositional_experiments`, `episodes`, `compositional_analysis` |
| Agent integration | `agent_adapters`, `llm`, `openai_costs`, `openrouter_costs` |

The older-named `patient_v1` and `patient_p2` modules remain because current
construction and verification depend on them. Provider cost helpers remain
runtime dependencies; the private study's recovery/orchestration scripts do not.
Likewise, serialized `synthea`/`Synthea` labels are legacy compiler namespaces,
not a check that supplied records were produced by Synthea.
