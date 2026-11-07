"""Bounded, replayable agent episodes over a patient-bound tool registry."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import AliasChoices, Field, model_serializer, model_validator

from patient_tool_gym.agent_adapters import (
    AdapterDescriptor,
    AdapterTokenUsage,
    AgentAdapter,
    AgentAdapterError,
    AgentRequest,
    AgentToolResult,
)
from patient_tool_gym.evidence import PatientSnapshot
from patient_tool_gym.evidence_contracts import (
    ObservedToolCall,
    build_complete_search_certificates,
)
from patient_tool_gym.llm import LLMError
from patient_tool_gym.openai_costs import OpenAIResponseCostLimitError
from patient_tool_gym.openrouter_costs import OpenRouterResponseCostLimitError
from patient_tool_gym.program import (
    ProgramExecutionError,
    StrictModel,
    TypedValue,
    content_hash,
    program_submission_answer_schema,
)
from patient_tool_gym.task import TaskArtifact
from patient_tool_gym.tools import ToolRegistry
from patient_tool_gym.verifier import SubmissionScore, score_submission, verify_task_artifact

EPISODE_SCHEMA_VERSION = "patient-tool-gym.episode.v3"


class EpisodeResponseGuard(Protocol):
    """Optional paid-response accounting hook shared across an episode batch."""

    def before_request(self) -> None: ...

    def after_response(
        self,
        response_id: str | None,
        response_metadata: Mapping[str, Any],
    ) -> None: ...


class EpisodeLimits(StrictModel):
    max_turns: int = Field(default=12, ge=1)
    max_tool_calls: int = Field(default=24, ge=1)
    tool_exposure: Literal["task_allowed", "all_public"] = "task_allowed"
    verification_policy: Literal["standard", "selector_hidden_ablation"] = "standard"


class EpisodeTokenUsage(StrictModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    responses: int = Field(default=0, ge=0)


class EpisodeError(StrictModel):
    code: str
    message: str
    turn: int | None = None
    call_id: str | None = None
    status_code: int | None = Field(default=None, ge=100, le=599)
    request_id: str | None = Field(default=None, min_length=1, max_length=255)


class EpisodeMessage(StrictModel):
    turn: int = Field(ge=0)
    role: Literal["user", "assistant", "tool"]
    content: str
    response_id: str | None = None
    model: str | None = None
    call_id: str | None = None
    name: str | None = None
    tool_call_ids: tuple[str, ...] = ()
    response_metadata: dict[str, Any] = Field(default_factory=dict)


class EpisodeToolCall(StrictModel):
    turn: int = Field(ge=1)
    call_id: str
    name: str
    arguments: dict[str, Any]
    executed: bool


class EpisodeToolResult(StrictModel):
    turn: int = Field(ge=1)
    call_id: str
    name: str
    result: dict[str, Any]


class EpisodeRecord(StrictModel):
    schema_version: Literal["patient-tool-gym.episode.v2", "patient-tool-gym.episode.v3"] = (
        EPISODE_SCHEMA_VERSION
    )
    record_hash: str
    episode_id: str
    task_id: str
    source_task_hash: str
    patient_snapshot_id: str
    question: str
    status: Literal["completed", "failed", "turn_limit", "tool_call_limit"]
    adapter_id: str
    adapter_descriptor: AdapterDescriptor | None = None
    model: str | None
    diagnostic_only: bool
    config_hash: str
    seed: int | None
    limits: EpisodeLimits
    turns: int = Field(ge=0)
    tool_call_count: int = Field(ge=0)
    latency_ms: float = Field(ge=0)
    tokens: EpisodeTokenUsage
    messages: tuple[EpisodeMessage, ...]
    tool_calls: tuple[EpisodeToolCall, ...]
    tool_results: tuple[EpisodeToolResult, ...]
    final_answer: TypedValue | None = None
    evidence_ids: tuple[str, ...] = ()
    score: SubmissionScore | None = None
    errors: tuple[EpisodeError, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_tool_exposure(cls, value: Any) -> Any:
        """Preserve the interface actually used by pre-tool-policy episodes.

        Historical v2 records omitted ``tool_exposure`` because the runner exposed every
        public tool. New runs explicitly default to ``task_allowed``; only persisted records
        with the missing field are migrated to ``all_public`` here.
        """

        if not isinstance(value, dict):
            return value
        limits = value.get("limits")
        if not isinstance(limits, dict) or "tool_exposure" in limits:
            return value
        migrated = dict(value)
        migrated["limits"] = {**limits, "tool_exposure": "all_public"}
        return migrated

    @model_serializer(mode="wrap")
    def serialize_without_legacy_null_descriptor(self, handler: Any) -> dict[str, Any]:
        payload = handler(self)
        if self.adapter_descriptor is None:
            payload.pop("adapter_descriptor", None)
        return payload

    @model_validator(mode="after")
    def validate_answer_schema(self) -> EpisodeRecord:
        if (
            self.final_answer is not None
            and self.final_answer.kind == "truth_state"
            and self.schema_version != "patient-tool-gym.episode.v3"
        ):
            raise ValueError("truth_state episode answers require patient-tool-gym.episode.v3")
        if self.adapter_descriptor is not None and (
            self.adapter_descriptor.adapter_id != self.adapter_id
            or self.adapter_descriptor.diagnostic_only != self.diagnostic_only
        ):
            raise ValueError("episode adapter descriptor does not match adapter identity")
        return self


class EpisodeBatchConfig(StrictModel):
    adapter_id: str
    model: str | None
    diagnostic_only: bool
    config_hash: str
    seed: int | None
    limits: EpisodeLimits


class EpisodeBatchFailure(StrictModel):
    task_id: str
    episode_id: str
    status: Literal["completed", "failed", "turn_limit", "tool_call_limit"]
    failure_codes: tuple[str, ...] = Field(
        default=(),
        validation_alias=AliasChoices("failure_codes", "error_codes"),
    )

    @property
    def error_codes(self) -> tuple[str, ...]:
        """Backward-compatible attribute for pre-rename callers."""

        return self.failure_codes


class EpisodeBatchSummary(StrictModel):
    selected: int = Field(ge=0)
    new: int = Field(ge=0)
    resumed: int = Field(ge=0)
    completed: int = Field(ge=0)
    passed: int = Field(ge=0)
    failures: int = Field(ge=0)
    failure_details: tuple[EpisodeBatchFailure, ...] = ()
    tokens: EpisodeTokenUsage
    latency_ms: float = Field(ge=0)
    config: EpisodeBatchConfig


class EpisodeBatchFailFastError(RuntimeError):
    """Raised after durably recording an integrity-invalid failed episode."""

    def __init__(self, record: EpisodeRecord, *, resumed: bool) -> None:
        error_codes = sorted({error.code for error in record.errors})
        disposition = "encountering persisted" if resumed else "durably recording"
        super().__init__(
            f"batch fail-fast policy stopped after {disposition} failed episode "
            f"{record.episode_id} (codes={error_codes})"
        )
        self.episode_id = record.episode_id
        self.error_codes = tuple(error_codes)
        self.errors = record.errors
        self.resumed = resumed

    def public_dict(self) -> dict[str, Any]:
        """Return only the already-sanitized diagnostics persisted with the episode."""

        return {
            "code": "episode_batch_fail_fast",
            "message": str(self),
            "episode_id": self.episode_id,
            "error_codes": list(self.error_codes),
            "episode_errors": [
                error.model_dump(mode="json", exclude_none=True) for error in self.errors
            ],
            "resumed": self.resumed,
        }


def is_terminal_model_no_answer(record: EpisodeRecord) -> bool:
    """Return whether a failed episode contains a service-observed terminal no-answer.

    A Responses API ``incomplete`` object is a model outcome: the service returned a
    response ID, model identity, status, and usage, but no valid submission.  It is
    therefore scored as an unsuccessful observation instead of being treated as an
    indeterminate transport failure.  Fail closed for legacy or transport failures
    that do not contain the service response metadata.
    """

    if record.status != "failed" or record.final_answer is not None or record.score is not None:
        return False
    if not record.errors or {error.code for error in record.errors} != {"incomplete_response"}:
        return False
    assistant = next(
        (
            message
            for message in reversed(record.messages)
            if message.role == "assistant" and message.turn == record.turns
        ),
        None,
    )
    return bool(
        assistant is not None
        and assistant.response_id
        and assistant.response_metadata.get("response_status") == "incomplete"
        and assistant.response_metadata.get("returned_model_identifier_source")
        == "service_response"
    )


SnapshotResolver = Callable[[str], PatientSnapshot | ToolRegistry]
AdapterFactory = Callable[[TaskArtifact], AgentAdapter]
EpisodeSchemaVersion = Literal["patient-tool-gym.episode.v2", "patient-tool-gym.episode.v3"]


def _episode_schema_version(task: TaskArtifact) -> EpisodeSchemaVersion:
    if task.schema_version == "patient-tool-gym.task.v2":
        return "patient-tool-gym.episode.v3"
    return "patient-tool-gym.episode.v2"


def episode_config_hash(
    task: TaskArtifact,
    adapter: AgentAdapter,
    *,
    limits: EpisodeLimits,
    seed: int | None,
) -> str:
    descriptor = _descriptor(adapter)
    return _episode_config_hash_from_descriptor(
        episode_schema_version=_episode_schema_version(task),
        descriptor=descriptor,
        limits=limits,
        seed=seed,
    )


def _episode_config_hash_from_descriptor(
    *,
    episode_schema_version: EpisodeSchemaVersion,
    descriptor: AdapterDescriptor,
    limits: EpisodeLimits,
    seed: int | None,
) -> str:
    return content_hash(
        {
            "episode_schema_version": episode_schema_version,
            "adapter": descriptor.model_dump(mode="json"),
            "limits": limits.model_dump(mode="json"),
            "seed": seed,
        }
    )


def source_task_hash(task: TaskArtifact) -> str:
    """Hash the complete immutable source artifact, not only its derived task ID."""

    return content_hash(task)


def episode_id(task: TaskArtifact, adapter: AgentAdapter, *, config_hash: str) -> str:
    descriptor = _descriptor(adapter)
    return _episode_id_from_parts(
        task_id=task.task_id,
        source_hash=source_task_hash(task),
        adapter_id=descriptor.adapter_id,
        config_hash=config_hash,
    )


def run_episode(
    task: TaskArtifact,
    registry: ToolRegistry,
    adapter: AgentAdapter,
    *,
    limits: EpisodeLimits | None = None,
    seed: int | None = None,
    response_guard: EpisodeResponseGuard | None = None,
) -> EpisodeRecord:
    """Run one bounded episode and score a successful ``submit_answer`` call."""

    limits = limits or EpisodeLimits()
    _validate_preconditions(task, registry, verification_policy=limits.verification_policy)
    return _run_episode_validated(
        task,
        registry,
        adapter,
        limits=limits,
        seed=seed,
        response_guard=response_guard,
    )


def _run_episode_validated(
    task: TaskArtifact,
    registry: ToolRegistry,
    adapter: AgentAdapter,
    *,
    limits: EpisodeLimits,
    seed: int | None,
    response_guard: EpisodeResponseGuard | None = None,
) -> EpisodeRecord:
    descriptor = _descriptor(adapter)
    task_hash = source_task_hash(task)
    config_hash = episode_config_hash(task, adapter, limits=limits, seed=seed)
    run_id = episode_id(task, adapter, config_hash=config_hash)
    visible_tools = _agent_visible_tools(registry, task, exposure=limits.tool_exposure)
    visible_tool_names = {str(tool["name"]) for tool in visible_tools}
    request = AgentRequest(
        question=task.question,
        tools=visible_tools,
    )

    started_ns = time.perf_counter_ns()
    messages = [EpisodeMessage(turn=0, role="user", content=task.question)]
    calls: list[EpisodeToolCall] = []
    results: list[EpisodeToolResult] = []
    errors: list[EpisodeError] = []
    usage_accumulator = _UsageAccumulator()
    final_answer: TypedValue | None = None
    evidence_ids: tuple[str, ...] = ()
    submission_score: SubmissionScore | None = None
    status: Literal["completed", "failed", "turn_limit", "tool_call_limit"] = "failed"
    executed_call_count = 0
    turns = 0
    model = descriptor.model
    seen_call_ids: set[str] = set()
    observed_evidence_ids: set[str] = set()
    successful_task_retrievals = 0
    complete_task_retrievals = 0

    try:
        session = adapter.start(request, seed=seed)
        pending_results: tuple[AgentToolResult, ...] = ()
        while turns < limits.max_turns:
            if executed_call_count >= limits.max_tool_calls:
                status = "tool_call_limit"
                errors.append(
                    EpisodeError(
                        code="max_tool_calls_exceeded",
                        message="The episode reached its tool-call limit before submission.",
                        turn=turns,
                    )
                )
                break

            turns += 1
            if response_guard is not None:
                response_guard.before_request()
            turn = session.next(pending_results)
            pending_results = ()
            usage_accumulator.add(turn.usage)
            model = turn.model or model
            messages.append(
                EpisodeMessage(
                    turn=turns,
                    role="assistant",
                    content=turn.message,
                    response_id=turn.response_id,
                    model=turn.model,
                    response_metadata=copy.deepcopy(turn.response_metadata),
                    tool_call_ids=tuple(call.call_id for call in turn.tool_calls),
                )
            )
            if response_guard is not None:
                response_guard.after_response(turn.response_id, turn.response_metadata)

            duplicate_ids = [
                call.call_id
                for call in turn.tool_calls
                if call.call_id in seen_call_ids
                or sum(other.call_id == call.call_id for other in turn.tool_calls) > 1
            ]
            submit_positions = [
                index for index, call in enumerate(turn.tool_calls) if call.name == "submit_answer"
            ]
            unexposed_calls = [
                call.call_id for call in turn.tool_calls if call.name not in visible_tool_names
            ]
            terminal_invalid = bool(submit_positions) and (
                len(submit_positions) != 1 or len(turn.tool_calls) != 1
            )
            capacity_exceeded = executed_call_count + len(turn.tool_calls) > limits.max_tool_calls
            if duplicate_ids or unexposed_calls or terminal_invalid or capacity_exceeded:
                calls.extend(
                    EpisodeToolCall(
                        turn=turns,
                        call_id=call.call_id,
                        name=call.name,
                        arguments=copy.deepcopy(call.arguments),
                        executed=False,
                    )
                    for call in turn.tool_calls
                )
                if duplicate_ids:
                    status = "failed"
                    errors.append(
                        EpisodeError(
                            code="duplicate_call_id",
                            message="An adapter reused a tool call identifier.",
                            turn=turns,
                            call_id=sorted(duplicate_ids)[0],
                        )
                    )
                elif terminal_invalid:
                    status = "failed"
                    errors.append(
                        EpisodeError(
                            code="invalid_terminal_submission",
                            message="submit_answer must be the single final call in its turn.",
                            turn=turns,
                        )
                    )
                elif capacity_exceeded:
                    status = "tool_call_limit"
                    errors.append(
                        EpisodeError(
                            code="max_tool_calls_exceeded",
                            message="The requested tool calls would exceed the episode limit.",
                            turn=turns,
                        )
                    )
                else:
                    status = "failed"
                    errors.append(
                        EpisodeError(
                            code="tool_not_exposed",
                            message=(
                                "The adapter requested a tool outside the episode's exposed "
                                "tool contract."
                            ),
                            turn=turns,
                            call_id=sorted(unexposed_calls)[0],
                        )
                    )
                break

            turn_results: list[AgentToolResult] = []
            submitted = False
            for call in turn.tool_calls:
                seen_call_ids.add(call.call_id)
                calls.append(
                    EpisodeToolCall(
                        turn=turns,
                        call_id=call.call_id,
                        name=call.name,
                        arguments=copy.deepcopy(call.arguments),
                        executed=True,
                    )
                )
                result = registry.call(call.name, copy.deepcopy(call.arguments))
                executed_call_count += 1
                result_record = EpisodeToolResult(
                    turn=turns,
                    call_id=call.call_id,
                    name=call.name,
                    result=copy.deepcopy(result),
                )
                results.append(result_record)
                messages.append(
                    EpisodeMessage(
                        turn=turns,
                        role="tool",
                        content=_canonical_json(result),
                        call_id=call.call_id,
                        name=call.name,
                    )
                )
                turn_results.append(
                    AgentToolResult(
                        call_id=call.call_id,
                        name=call.name,
                        result=copy.deepcopy(result),
                    )
                )
                if call.name != "submit_answer" and result.get("ok"):
                    observed_evidence_ids.update(_result_evidence_ids(result))
                    if call.name in task.allowed_tools and call.name != "submit_answer":
                        successful_task_retrievals += 1
                        if bool(result.get("complete", True)):
                            complete_task_retrievals += 1
                if call.name == "submit_answer" and result.get("ok"):
                    submission = result.get("submission")
                    if not isinstance(submission, dict):
                        raise ProgramExecutionError("submit_answer returned no submission")
                    final_answer = TypedValue.from_submission(submission.get("answer"))
                    submitted_ids = submission.get("evidence_ids", [])
                    if not isinstance(submitted_ids, list) or not all(
                        isinstance(item, str) for item in submitted_ids
                    ):
                        raise ProgramExecutionError("submit_answer returned invalid evidence IDs")
                    evidence_ids = tuple(submitted_ids)
                    observed_tool_calls = _observed_tool_calls(calls, results)
                    complete_search_certificates = build_complete_search_certificates(
                        task,
                        registry,
                        observed_tool_calls,
                    )
                    submission_score = score_submission(
                        task,
                        answer=final_answer,
                        evidence_ids=evidence_ids,
                        registry=registry,
                        complete_search_certificates=complete_search_certificates,
                        observed_tool_calls=observed_tool_calls,
                    )
                    provenance_errors: list[EpisodeError] = []
                    unobserved = sorted(set(evidence_ids) - observed_evidence_ids)
                    if unobserved:
                        provenance_errors.append(
                            EpisodeError(
                                code="unobserved_evidence_ids",
                                message=(
                                    "The submission cited evidence that no prior tool result "
                                    "returned."
                                ),
                                turn=turns,
                                call_id=call.call_id,
                            )
                        )
                    retrieval_count = (
                        complete_task_retrievals
                        if not task.gold_evidence_ids
                        else successful_task_retrievals
                    )
                    if retrieval_count == 0 and not submission_score.clarification_pass:
                        provenance_errors.append(
                            EpisodeError(
                                code="answer_submitted_without_retrieval",
                                message=(
                                    "The submission was not preceded by a complete successful "
                                    "task-relevant retrieval."
                                ),
                                turn=turns,
                                call_id=call.call_id,
                            )
                        )
                    if provenance_errors:
                        submission_score = submission_score.model_copy(update={"passed": False})
                        errors.extend(provenance_errors)
                    status = "completed"
                    submitted = True
                    break
            if submitted:
                break
            pending_results = tuple(turn_results)
        else:
            status = "turn_limit"
            errors.append(
                EpisodeError(
                    code="max_turns_exceeded",
                    message="The episode reached its turn limit before submission.",
                    turn=turns,
                )
            )
    except Exception as exc:
        caught_exc = exc
        if isinstance(exc, AgentAdapterError) and exc.response_id is not None:
            usage_accumulator.add(exc.usage)
            model = exc.model or model
            messages.append(
                EpisodeMessage(
                    turn=turns,
                    role="assistant",
                    content=exc.partial_output,
                    response_id=exc.response_id,
                    model=exc.model,
                    response_metadata=copy.deepcopy(exc.response_metadata),
                    tool_call_ids=(),
                )
            )
            if response_guard is not None:
                try:
                    response_guard.after_response(exc.response_id, exc.response_metadata)
                except Exception as guard_exc:
                    caught_exc = guard_exc
        status = "failed"
        errors.append(_safe_error(caught_exc, turn=turns or None))

    latency_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
    draft = EpisodeRecord(
        schema_version=_episode_schema_version(task),
        record_hash="",
        episode_id=run_id,
        task_id=task.task_id,
        source_task_hash=task_hash,
        patient_snapshot_id=task.patient_snapshot_id,
        question=task.question,
        status=status,
        adapter_id=descriptor.adapter_id,
        adapter_descriptor=descriptor,
        model=model,
        diagnostic_only=descriptor.diagnostic_only,
        config_hash=config_hash,
        seed=seed,
        limits=limits,
        turns=turns,
        tool_call_count=executed_call_count,
        latency_ms=round(latency_ms, 3),
        tokens=usage_accumulator.finish(),
        messages=tuple(messages),
        tool_calls=tuple(calls),
        tool_results=tuple(results),
        final_answer=final_answer,
        evidence_ids=evidence_ids,
        score=submission_score,
        errors=tuple(errors),
    )
    return draft.model_copy(update={"record_hash": _episode_record_hash(draft)})


def run_episode_resumable(
    task: TaskArtifact,
    registry: ToolRegistry,
    adapter: AgentAdapter,
    output_path: str | Path,
    *,
    limits: EpisodeLimits | None = None,
    seed: int | None = None,
    response_guard: EpisodeResponseGuard | None = None,
) -> EpisodeRecord:
    """Return an existing keyed record or atomically append one completed attempt."""

    limits = limits or EpisodeLimits()
    _validate_preconditions(task, registry, verification_policy=limits.verification_policy)
    config_hash = episode_config_hash(task, adapter, limits=limits, seed=seed)
    expected_id = episode_id(task, adapter, config_hash=config_hash)
    path = Path(output_path)
    existing = load_episode_records(path)
    if expected_id in existing:
        _validate_resumed_record(
            existing[expected_id],
            task=task,
            registry=registry,
            adapter=adapter,
            limits=limits,
            seed=seed,
        )
        return existing[expected_id]

    record = _run_episode_validated(
        task,
        registry,
        adapter,
        limits=limits,
        seed=seed,
        response_guard=response_guard,
    )
    if record.episode_id != expected_id:
        raise RuntimeError("episode identity changed during the run")
    _atomic_append_record(path, record)
    return record


def run_episodes_file(
    tasks: Iterable[TaskArtifact],
    *,
    snapshot_resolver: SnapshotResolver,
    adapter_factory: AdapterFactory,
    output_path: str | Path,
    limits: EpisodeLimits | None = None,
    seed: int | None = None,
    fail_on_episode_failure: bool = False,
    fail_on_response_cost_limit: bool = False,
    response_guard: EpisodeResponseGuard | None = None,
) -> EpisodeBatchSummary:
    """Run or resume one compatible cohort with a single-pass JSONL append log.

    All tasks, snapshots, adapters, and existing output are validated before any
    new agent session starts. New records are appended and fsynced one at a time,
    so an interrupted cohort can resume without rereading or rewriting the log for
    every task. When ``fail_on_episode_failure`` is enabled, an episode with
    ``status=failed`` is durably appended and then aborts the batch; completed wrong
    submissions and preregistered limit terminations remain ordinary observations.
    """

    selected_tasks = tuple(tasks)
    if not selected_tasks:
        raise ValueError("at least one task is required for a batch run")
    duplicate_task_ids = _duplicates(task.task_id for task in selected_tasks)
    if duplicate_task_ids:
        raise ValueError(f"duplicate task_id in batch: {duplicate_task_ids[0]}")

    path = Path(output_path)
    existing = load_episode_records(path)
    limits = limits or EpisodeLimits()
    prepared: list[_PreparedEpisode] = []
    batch_descriptor: AdapterDescriptor | None = None
    batch_config_hash: str | None = None
    for task in selected_tasks:
        registry = _resolve_registry(task, snapshot_resolver)
        _validate_preconditions(task, registry, verification_policy=limits.verification_policy)
        adapter = adapter_factory(task)
        descriptor = _descriptor(adapter)
        config_hash = episode_config_hash(task, adapter, limits=limits, seed=seed)
        if batch_config_hash is None:
            batch_config_hash = config_hash
            batch_descriptor = descriptor
        elif config_hash != batch_config_hash or descriptor != batch_descriptor:
            raise ValueError("adapter_factory produced mixed incompatible configurations")
        prepared.append(
            _PreparedEpisode(
                task=task,
                registry=registry,
                adapter=adapter,
                episode_id=episode_id(task, adapter, config_hash=config_hash),
            )
        )

    assert batch_descriptor is not None and batch_config_hash is not None
    _validate_batch_output(
        existing,
        prepared=prepared,
        descriptor=batch_descriptor,
        config_hash=batch_config_hash,
        limits=limits,
        seed=seed,
    )

    selected_records: list[EpisodeRecord] = []
    resumed = 0
    new = 0
    new_prepared = [item for item in prepared if item.episode_id not in existing]
    output_handle = None
    try:
        if new_prepared:
            path.parent.mkdir(parents=True, exist_ok=True)
            output_handle = path.open("ab")
        for item in prepared:
            record = existing.get(item.episode_id)
            was_resumed = record is not None
            if record is not None:
                resumed += 1
            else:
                record = _run_episode_validated(
                    item.task,
                    item.registry,
                    item.adapter,
                    limits=limits,
                    seed=seed,
                    response_guard=response_guard,
                )
                if record.episode_id != item.episode_id:
                    raise RuntimeError("episode identity changed during the batch run")
                assert output_handle is not None
                _append_record_to_handle(output_handle, record)
                existing[record.episode_id] = record
                new += 1
            selected_records.append(record)
            if (
                fail_on_episode_failure
                and record.status == "failed"
                and not is_terminal_model_no_answer(record)
            ) or (
                fail_on_response_cost_limit
                and any(error.code == "response_cost_limit" for error in record.errors)
            ):
                raise EpisodeBatchFailFastError(record, resumed=was_resumed)
    finally:
        if output_handle is not None:
            output_handle.close()

    usage = _UsageAccumulator()
    failures: list[EpisodeBatchFailure] = []
    for record in selected_records:
        usage.add_episode(record.tokens)
        if record.score is None or not record.score.passed:
            failure_codes = tuple(error.code for error in record.errors)
            if not failure_codes and record.status == "completed":
                failure_codes = ("submission_incorrect",)
            failures.append(
                EpisodeBatchFailure(
                    task_id=record.task_id,
                    episode_id=record.episode_id,
                    status=record.status,
                    failure_codes=failure_codes,
                )
            )

    return EpisodeBatchSummary(
        selected=len(prepared),
        new=new,
        resumed=resumed,
        completed=sum(record.status == "completed" for record in selected_records),
        passed=sum(bool(record.score and record.score.passed) for record in selected_records),
        failures=len(failures),
        failure_details=tuple(failures),
        tokens=usage.finish(),
        latency_ms=round(sum(record.latency_ms for record in selected_records), 3),
        config=EpisodeBatchConfig(
            adapter_id=batch_descriptor.adapter_id,
            model=batch_descriptor.model,
            diagnostic_only=batch_descriptor.diagnostic_only,
            config_hash=batch_config_hash,
            seed=seed,
            limits=limits,
        ),
    )


# A name that reads naturally for callers that do not care about the storage format.
run_episode_batch = run_episodes_file


@dataclass(frozen=True)
class _PreparedEpisode:
    task: TaskArtifact
    registry: ToolRegistry
    adapter: AgentAdapter
    episode_id: str


def load_episode_records(path: str | Path) -> dict[str, EpisodeRecord]:
    """Load a JSONL episode log, failing closed on malformed or duplicate records."""

    source = Path(path)
    if not source.exists():
        return {}
    if source.stat().st_size:
        with source.open("rb") as binary_handle:
            binary_handle.seek(-1, os.SEEK_END)
            if binary_handle.read(1) != b"\n":
                raise ValueError("episode JSONL file has a truncated final line")
    records: dict[str, EpisodeRecord] = {}
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid episode JSONL record at line {line_number}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"episode record is not an object at line {line_number}")
            recorded_hash = raw.get("record_hash")
            hash_payload = {key: value for key, value in raw.items() if key != "record_hash"}
            if recorded_hash != content_hash(hash_payload):
                raise ValueError(f"invalid episode content hash at line {line_number}")
            try:
                record = EpisodeRecord.model_validate(raw)
            except ValueError as exc:
                raise ValueError(f"invalid episode JSONL record at line {line_number}") from exc
            if record.episode_id in records:
                raise ValueError(f"duplicate episode_id at line {line_number}")
            expected_id = _episode_id_from_parts(
                task_id=record.task_id,
                source_hash=record.source_task_hash,
                adapter_id=record.adapter_id,
                config_hash=record.config_hash,
            )
            if record.episode_id != expected_id:
                raise ValueError(f"invalid episode identity at line {line_number}")
            if record.adapter_descriptor is not None:
                expected_config_hash = _episode_config_hash_from_descriptor(
                    episode_schema_version=record.schema_version,
                    descriptor=record.adapter_descriptor,
                    limits=record.limits,
                    seed=record.seed,
                )
                if record.config_hash != expected_config_hash:
                    raise ValueError(
                        f"adapter descriptor does not match config hash at line {line_number}"
                    )
            records[record.episode_id] = record
    return records


class _UsageAccumulator:
    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.seen_input = False
        self.seen_output = False
        self.seen_total = False
        self.responses = 0

    def add(self, usage: AdapterTokenUsage) -> None:
        self.responses += 1
        if usage.input_tokens is not None:
            self.input_tokens += usage.input_tokens
            self.seen_input = True
        if usage.output_tokens is not None:
            self.output_tokens += usage.output_tokens
            self.seen_output = True
        if usage.total_tokens is not None:
            self.total_tokens += usage.total_tokens
            self.seen_total = True

    def add_episode(self, usage: EpisodeTokenUsage) -> None:
        if usage.input_tokens is not None:
            self.input_tokens += usage.input_tokens
            self.seen_input = True
        if usage.output_tokens is not None:
            self.output_tokens += usage.output_tokens
            self.seen_output = True
        if usage.total_tokens is not None:
            self.total_tokens += usage.total_tokens
            self.seen_total = True
        self.responses += usage.responses

    def finish(self) -> EpisodeTokenUsage:
        return EpisodeTokenUsage(
            input_tokens=self.input_tokens if self.seen_input else None,
            output_tokens=self.output_tokens if self.seen_output else None,
            total_tokens=self.total_tokens if self.seen_total else None,
            responses=self.responses,
        )


def _descriptor(adapter: AgentAdapter) -> AdapterDescriptor:
    descriptor = AdapterDescriptor.model_validate(adapter.descriptor)
    # Prove that config hashing is deterministic before starting a potentially costly run.
    _canonical_json(descriptor.model_dump(mode="json"))
    return descriptor


def _validate_preconditions(
    task: TaskArtifact,
    registry: ToolRegistry,
    *,
    verification_policy: Literal["standard", "selector_hidden_ablation"] = "standard",
) -> None:
    if task.patient_snapshot_id != registry.snapshot_id:
        raise ValueError("task and registry snapshot IDs do not match")
    if not task.question.strip():
        raise ValueError("task question must not be empty")
    replayed = verify_task_artifact(task, registry, policy=verification_policy)
    if not replayed.passed:
        reasons = ", ".join(replayed.errors) or "unknown verification failure"
        raise ValueError(f"task failed replay verification: {reasons}")
    if not task.verification.passed or task.verification != replayed:
        raise ValueError("task does not contain the current replay verification report")


def _resolve_registry(task: TaskArtifact, resolver: SnapshotResolver) -> ToolRegistry:
    try:
        resolved = resolver(task.patient_snapshot_id)
    except Exception as exc:
        raise ValueError(f"snapshot resolver failed for task {task.task_id}") from exc
    if isinstance(resolved, ToolRegistry):
        return resolved
    if isinstance(resolved, PatientSnapshot):
        return ToolRegistry(resolved)
    raise TypeError("snapshot_resolver must return PatientSnapshot or ToolRegistry")


def _validate_resumed_record(
    record: EpisodeRecord,
    *,
    task: TaskArtifact,
    registry: ToolRegistry,
    adapter: AgentAdapter,
    limits: EpisodeLimits,
    seed: int | None,
) -> None:
    descriptor = _descriptor(adapter)
    expected_hash = source_task_hash(task)
    expected_config = episode_config_hash(task, adapter, limits=limits, seed=seed)
    expected_id = _episode_id_from_parts(
        task_id=task.task_id,
        source_hash=expected_hash,
        adapter_id=descriptor.adapter_id,
        config_hash=expected_config,
    )
    if (
        record.episode_id != expected_id
        or record.schema_version != _episode_schema_version(task)
        or record.task_id != task.task_id
        or record.source_task_hash != expected_hash
        or record.patient_snapshot_id != task.patient_snapshot_id
        or record.question != task.question
        or record.adapter_id != descriptor.adapter_id
        or (record.adapter_descriptor is not None and record.adapter_descriptor != descriptor)
        or record.diagnostic_only != descriptor.diagnostic_only
        or record.config_hash != expected_config
        or record.seed != seed
        or record.limits != limits
    ):
        raise ValueError("resumed episode does not match its task, adapter, or configuration")
    _validate_completed_episode_replay(record, task=task, registry=registry)


def _validate_batch_output(
    existing: dict[str, EpisodeRecord],
    *,
    prepared: list[_PreparedEpisode],
    descriptor: AdapterDescriptor,
    config_hash: str,
    limits: EpisodeLimits,
    seed: int | None,
) -> None:
    duplicate_output_tasks = _duplicates(record.task_id for record in existing.values())
    if duplicate_output_tasks:
        raise ValueError(f"duplicate task_id in episode output: {duplicate_output_tasks[0]}")
    for record in existing.values():
        if (
            record.adapter_id != descriptor.adapter_id
            or (record.adapter_descriptor is not None and record.adapter_descriptor != descriptor)
            or record.diagnostic_only != descriptor.diagnostic_only
            or record.config_hash != config_hash
            or record.seed != seed
            or record.limits != limits
        ):
            raise ValueError("episode output contains mixed incompatible configurations")

    prepared_by_task_id = {item.task.task_id: item for item in prepared}
    for record in existing.values():
        item = prepared_by_task_id.get(record.task_id)
        if item is None:
            continue
        if record.episode_id != item.episode_id:
            raise ValueError("episode output contains a conflicting source task artifact")
        _validate_resumed_record(
            record,
            task=item.task,
            registry=item.registry,
            adapter=item.adapter,
            limits=limits,
            seed=seed,
        )


def _validate_completed_episode_replay(
    record: EpisodeRecord,
    *,
    task: TaskArtifact,
    registry: ToolRegistry,
) -> None:
    """Replay persisted calls and recompute the terminal score before resumption."""

    if record.status != "completed":
        return
    visible_names = {
        str(tool["name"])
        for tool in _agent_visible_tools(
            registry,
            task,
            exposure=record.limits.tool_exposure,
        )
    }
    unexposed = sorted(call.call_id for call in record.tool_calls if call.name not in visible_names)
    if unexposed:
        raise ValueError("completed episode invoked a tool outside its exposed contract")
    executed = tuple(call for call in record.tool_calls if call.executed)
    if record.tool_call_count != len(executed) or len(record.tool_results) != len(executed):
        raise ValueError("completed episode has inconsistent tool-call accounting")

    observed_evidence_ids: set[str] = set()
    successful_task_retrievals = 0
    complete_task_retrievals = 0
    submission: dict[str, Any] | None = None
    submit_call: EpisodeToolCall | None = None
    for call, stored_result in zip(executed, record.tool_results, strict=True):
        if stored_result.call_id != call.call_id or stored_result.name != call.name:
            raise ValueError("completed episode tool calls and results are misaligned")
        replayed = registry.call(call.name, copy.deepcopy(call.arguments))
        if replayed != stored_result.result:
            raise ValueError("completed episode tool result failed deterministic replay")
        if call.name == "submit_answer":
            if submit_call is not None or not replayed.get("ok"):
                raise ValueError("completed episode has an invalid terminal submission")
            submit_call = call
            candidate = replayed.get("submission")
            if not isinstance(candidate, dict):
                raise ValueError("completed episode submission is missing")
            submission = candidate
        elif replayed.get("ok"):
            observed_evidence_ids.update(_result_evidence_ids(replayed))
            if call.name in task.allowed_tools:
                successful_task_retrievals += 1
                if bool(replayed.get("complete", True)):
                    complete_task_retrievals += 1

    if submit_call is None or submission is None or executed[-1] != submit_call:
        raise ValueError("completed episode has no final submit_answer call")
    if sum(call.turn == submit_call.turn for call in record.tool_calls) != 1:
        raise ValueError("completed episode submission was not alone in its turn")

    expected_answer = TypedValue.from_submission(submission.get("answer"))
    submitted_ids = submission.get("evidence_ids")
    if not isinstance(submitted_ids, list) or not all(
        isinstance(item, str) for item in submitted_ids
    ):
        raise ValueError("completed episode has invalid submitted evidence IDs")
    expected_evidence_ids = tuple(submitted_ids)
    observed_tool_calls = _observed_tool_calls(executed, record.tool_results)
    complete_search_certificates = build_complete_search_certificates(
        task,
        registry,
        observed_tool_calls,
    )
    expected_score = score_submission(
        task,
        answer=expected_answer,
        evidence_ids=expected_evidence_ids,
        registry=registry,
        complete_search_certificates=complete_search_certificates,
        observed_tool_calls=observed_tool_calls,
    )
    expected_error_codes: list[str] = []
    if set(expected_evidence_ids) - observed_evidence_ids:
        expected_error_codes.append("unobserved_evidence_ids")
    retrieval_count = (
        complete_task_retrievals if not task.gold_evidence_ids else successful_task_retrievals
    )
    if retrieval_count == 0 and not expected_score.clarification_pass:
        expected_error_codes.append("answer_submitted_without_retrieval")
    if expected_error_codes:
        expected_score = expected_score.model_copy(update={"passed": False})

    if (
        record.final_answer != expected_answer
        or record.evidence_ids != expected_evidence_ids
        or record.score != expected_score
        or tuple(error.code for error in record.errors) != tuple(expected_error_codes)
    ):
        raise ValueError("completed episode answer, evidence, or score failed replay")


def _observed_tool_calls(
    calls: Iterable[EpisodeToolCall],
    results: Iterable[EpisodeToolResult],
) -> tuple[ObservedToolCall, ...]:
    executed = tuple(call for call in calls if call.executed)
    stored_results = tuple(results)
    observations = []
    for call, result in zip(executed, stored_results, strict=True):
        if call.call_id != result.call_id or call.name != result.name:
            raise ValueError("executed tool calls and results are misaligned")
        observations.append(
            ObservedToolCall(
                call_id=call.call_id,
                tool=call.name,
                arguments=copy.deepcopy(call.arguments),
                result=copy.deepcopy(result.result),
            )
        )
    return tuple(observations)


def _agent_visible_tools(
    registry: ToolRegistry,
    task: TaskArtifact,
    *,
    exposure: Literal["task_allowed", "all_public"] = "task_allowed",
) -> tuple[dict[str, Any], ...]:
    """Project public tools plus the task's non-oracular answer shape."""

    projected = []
    for contract in registry.list_tools():
        if exposure == "task_allowed" and contract["name"] not in task.allowed_tools:
            continue
        item = {
            "name": contract["name"],
            "description": contract["description"],
            "input_schema": copy.deepcopy(contract["input_schema"]),
        }
        schema_mode = (
            task.metadata.get("interface_ablation", {}).get("schema_mode")
            if isinstance(task.metadata.get("interface_ablation"), dict)
            else None
        )
        allow_clarification = bool(
            task.question_contract is not None and task.question_contract.allow_clarification
        )
        if item["name"] == "submit_answer" and schema_mode != "generic":
            item["input_schema"]["properties"]["answer"] = program_submission_answer_schema(
                task.program,
                allow_clarification=allow_clarification,
            )
        projected.append(item)
    visible_names = {item["name"] for item in projected}
    required_names = set(task.allowed_tools) if exposure == "task_allowed" else {"submit_answer"}
    missing = sorted(required_names - visible_names)
    if missing:
        raise ValueError(f"task references unavailable public tool(s): {missing}")
    return tuple(projected)


def _result_evidence_ids(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        evidence_id = value.get("evidence_id")
        if isinstance(evidence_id, str):
            found.add(evidence_id)
        for child in value.values():
            found.update(_result_evidence_ids(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.update(_result_evidence_ids(child))
    return found


def _episode_record_hash(record: EpisodeRecord) -> str:
    payload = record.model_dump(mode="json", exclude={"record_hash"})
    if record.adapter_descriptor is None:
        payload.pop("adapter_descriptor", None)
    return content_hash(payload)


def _duplicates(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return tuple(sorted(duplicates))


def _episode_id_from_parts(
    *,
    task_id: str,
    source_hash: str,
    adapter_id: str,
    config_hash: str,
) -> str:
    digest = content_hash(
        {
            "task_id": task_id,
            "source_task_hash": source_hash,
            "adapter_id": adapter_id,
            "config_hash": config_hash,
        }
    ).removeprefix("sha256:")
    return f"patient-episode-{digest[:24]}"


def _safe_error(exc: Exception, *, turn: int | None) -> EpisodeError:
    if isinstance(exc, (OpenAIResponseCostLimitError, OpenRouterResponseCostLimitError)):
        return EpisodeError(
            code="response_cost_limit",
            message=str(exc),
            turn=turn,
        )
    if isinstance(exc, AgentAdapterError):
        return EpisodeError(code=exc.code, message=exc.public_message, turn=turn)
    if isinstance(exc, LLMError):
        return EpisodeError(
            code=exc.code,
            message=exc.public_message,
            turn=turn,
            status_code=exc.status_code,
            request_id=exc.request_id,
        )
    if isinstance(exc, ProgramExecutionError):
        return EpisodeError(
            code="invalid_submission",
            message="The submitted answer could not be represented for scoring.",
            turn=turn,
        )
    return EpisodeError(
        code="adapter_error",
        message=f"The adapter failed with {type(exc).__name__}.",
        turn=turn,
    )


def _atomic_append_record(path: Path, record: EpisodeRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_bytes() if path.exists() else b""
    if existing and not existing.endswith(b"\n"):
        raise ValueError("episode JSONL file has a truncated final line")
    encoded = record.model_dump_json().encode("utf-8") + b"\n"
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(existing)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _append_record_to_handle(handle: Any, record: EpisodeRecord) -> None:
    """Append one already validated JSON line and durably flush it."""

    handle.write(record.model_dump_json().encode("utf-8") + b"\n")
    handle.flush()
    os.fsync(handle.fileno())


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
