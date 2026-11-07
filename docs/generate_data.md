# Generate a new synthetic cohort

Use this workflow to generate new Synthea patients and compile eligible matched
tasks. It does **not** recreate the paper's selected 225-patient cohort: that
cohort used multiple batches, selection, exclusions, and frozen split assignments.
Use the [dataset download](dataset.md) for the exact released training/development data.

## Install the generator

You need Git, Bash, Python 3, and a Java 17 JDK. The helper pins Synthea to commit
`2ab21caa03f8c6a0400c06593e53f1c1b7510abe`
(`v4.0.0-1-g2ab21caa0`), the source version recorded for the paper cohort.

From the PatientFlowSynth repository, clone into a **new** directory:

```bash
git clone https://github.com/synthetichealth/synthea.git work/synthea-source
git -C work/synthea-source checkout --detach 2ab21caa03f8c6a0400c06593e53f1c1b7510abe
java -version
```

Cloning downloads external source code. The first generator run can also download
the Gradle distribution and Maven build dependencies. These are ordinary local
build steps, not model API calls. The helper does not install software, clone a
repository, or change the checkout's revision for you. Keep that checkout clean.

## Generate and compile

First validate the settings without starting Java or writing output:

```bash
bash scripts/generate_synthea.sh \
  --source work/synthea-source --output work/new-cohort \
  --patients 100 --seed 31 --clinician-seed 32 \
  --reference-date 20260707 --end-date 20260707 \
  --state Massachusetts --dry-run
```

Remove `--dry-run` to generate the cohort. Both dates use `YYYYMMDD`; the end
date must be no earlier than the reference date and no later than today. All
settings shown above are configurable. Use a new output directory for each run;
the helper never overwrites or resumes an existing one. Paths with apostrophes,
backslashes, or control characters are rejected because of the pinned upstream
launcher's argument format.

The helper writes:

- `generation.json`: generator commit, seeds, dates, settings, and launcher hash.
- `generation.log` and `result.json`: local execution log and completion status.
- `output/fhir/`: patient Bundles and provider/dependency Bundles.

After generation succeeds, compile the **whole FHIR directory**, retaining its
provider/dependency files:

```bash
patientflowsynth compile work/new-cohort/output/fhir work/new-release --seed 31
patientflowsynth verify work/new-release
patientflowsynth prepare work/new-release work/new-runtime
patientflowsynth run work/new-release work/new-runtime work/new-episodes.jsonl \
  --adapter reference --limit 8
patientflowsynth summarize work/new-release work/new-runtime work/new-episodes.jsonl
```

The requested population is not the admitted benchmark population. Only patients
with eligible records become matched task families, and a small cohort may admit
none. Inspect the compiler's reported counts; generate a larger, separately
versioned cohort if needed. The reference adapter checks the local pipeline with
reference answers. See [the runtime guide](runtime.md) to use your own agent.

## Provenance and redistribution

Patients and clinical histories are simulated. Synthea can also include public
facility names, locations, and other provider-directory descriptors; do not
describe every organization or geographic field as fictional. Keep generation
logs local and review any archive before sharing it.

The upstream Synthea checkout contains `LICENSE` (Apache-2.0) and `NOTICE`.
Retain appropriate upstream attribution when distributing generated material.
Its notice separately identifies RxNorm, LOINC, and SNOMED CT content and their
respective terms. PatientFlowSynth's code license is not a blanket license for
third-party terminology or provider data. A data archive should include its own
scope/provenance statement, applicable licenses/notices, and file checksums.
