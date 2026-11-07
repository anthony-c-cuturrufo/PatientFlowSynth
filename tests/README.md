# Offline regression tests

These tests exercise the public compiler, FHIR normalizer, read-only tools, release
integrity checks, and scorer using deliberately fabricated records. The files in
`fixtures/` are test examples, not benchmark patients or deidentified clinical data.
Temporary releases are built inside pytest's temporary directories.

Coverage includes execute-before-render ordering, patient boundaries, deterministic
snapshots, patch replay, participant-package field exclusion, held-out split binding,
four-state labels, exact witnesses, complete-search certificates, and rejection of
unsupported or excessive citations.

The source-only selection excludes private-artifact integration tests and hosted-model
experiments. Two legacy CLI checks were omitted because those commands are outside
the public interface; the corresponding function-level tests remain. The evidence
quality tests use a local positive-case program instead of the legacy batch compiler
and omit assertions on the research-only metrics script. Public CLI behavior is
covered separately in `test_cli.py`.

Run from the repository root after installing the development dependencies:

```sh
uv run pytest
```

Passing these tests checks software behavior on fabricated examples; it does not
reproduce the paper's private evaluations or establish clinical validity.
