"""Pluggable agent boundaries for patient-tool-gym episodes.

Adapters intentionally receive a narrow :class:`AgentRequest`: the patient-facing
question and public tool contracts.  Gold programs, reference traces, answers, and
task-policy hints are runner concerns and are not part of the adapter protocol.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Protocol, runtime_checkable

from openai import OpenAI
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_input_param import FunctionCallOutput
from pydantic import Field

from patient_tool_gym.lifecycle import lifecycle_polarity
from patient_tool_gym.llm import OpenAISettings, OpenAITextClient
from patient_tool_gym.program import ReferenceCall, StrictModel, TypedValue, content_hash
from patient_tool_gym.task import TaskArtifact

OPENAI_AGENT_INSTRUCTIONS = (
    "You are a patient-facing chart assistant operating in a tool sandbox. "
    "Use only the provided tools and their returned records to answer the patient's question. "
    "Do not invent evidence identifiers or submit an identifier that was not returned by a prior "
    "tool call. For a yes/no question, submit the JSON Boolean true or false, not the strings "
    "'yes' or 'no'. The public submit_answer answer schema is authoritative. When a question "
    "requests an evidence state and that schema requires a structured branches-and-overall "
    "object, do not use the single-state truth_state envelope. Submit the "
    'raw JSON object shaped as {"branches":{"<required_branch_name>":"<state>"},'
    '"overall":"<state>"}, including every branch name required by that schema and no '
    "additional keys. Each state in this "
    'object is a bare string: "supported", "refuted", "insufficient_evidence", or '
    '"conflicting_evidence". Otherwise, for a single evidence-state answer, submit exactly '
    '{"kind":"truth_state","value":STATE} using one of those same four states. Insufficient '
    "evidence is not the same as refutation, and conflicting evidence requires both supporting "
    "and refuting records. Compute a structured overall value with evidence-state "
    "conjunction: encode supported=(true,false), refuted=(false,true), "
    "insufficient_evidence=(false,false), and conflicting_evidence=(true,true); AND the first "
    "bits across branches, OR the second bits, then map the two resulting bits back to a state. "
    "For a structured branches-and-overall question, complete every named clinical branch "
    "before submitting. For a branch with an explicit target date, call its domain search with "
    "system, date_from and date_to both equal to that date in YYYY-MM-DD format, and limit 100. "
    "Use http://loinc.org for LOINC, http://snomed.info/sct for SNOMED CT, and "
    "http://hl7.org/fhir/sid/cvx for CVX. "
    "For a branch relative to the most recent completed encounter, first search encounters for "
    "the finished encounter type named in the question, choose the newest returned match, and "
    "then use that returned "
    "encounter's evidence_id and date as encounter_id and the equal date bounds in the branch "
    "search. In each clinical branch search, omit code, text, status, clinical_status, "
    "verification_status, and cursor so same-system contradictory or distracting records "
    "remain observable. A required branch "
    "search is complete only when ok is true, complete is true, next_cursor is null, and the "
    "number of returned records equals total_matches. Inspect all returned records and classify "
    "only records matching the branch code named in the question. Cite all and only those "
    "matching records that determine the branch state, plus the selected encounter record for "
    "an encounter-relative branch. Cite both supporting and refuting matches for conflicting "
    "evidence, cite an explicit missing-data record for insufficient evidence, and exclude "
    "different-code distractors. For other answers, preserve the JSON value and units returned "
    "by the tools. For any evidence-state "
    "answer, cite the matching status-bearing record; cite no record only when a complete search "
    "returns no match. "
    "If the question does not identify which clinical concept to evaluate, do not guess the "
    "hidden concept. Submit exactly "
    '{"kind":"clarification","value":"missing_clinical_selector"} as the answer with an '
    "empty evidence_ids list. "
    "Include exactly the evidence IDs that directly support the answer. Finish by calling "
    "submit_answer exactly once, as the only call in its turn. A successful submit_answer call "
    "ends the episode."
)


class AgentAdapterError(RuntimeError):
    """An adapter failure with a message safe to persist in an episode record."""

    def __init__(
        self,
        code: str,
        public_message: str,
        *,
        response_id: str | None = None,
        model: str | None = None,
        response_metadata: Mapping[str, Any] | None = None,
        usage: AdapterTokenUsage | None = None,
        partial_output: str = "",
    ) -> None:
        super().__init__(public_message)
        self.code = code
        self.public_message = public_message
        self.response_id = response_id
        self.model = model
        self.response_metadata = copy.deepcopy(dict(response_metadata or {}))
        self.usage = usage or AdapterTokenUsage()
        self.partial_output = partial_output


class AgentRequest(StrictModel):
    """The complete initial information made visible to a non-oracle agent."""

    question: str = Field(min_length=1)
    tools: tuple[dict[str, Any], ...]


class AgentToolCall(StrictModel):
    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, Any]


class AgentToolResult(StrictModel):
    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    result: dict[str, Any]


class AdapterTokenUsage(StrictModel):
    input_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    cache_write_input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    reasoning_output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class AdapterTurn(StrictModel):
    """One assistant response; final answers must still be submitted as a tool call."""

    message: str = ""
    tool_calls: tuple[AgentToolCall, ...] = ()
    response_id: str | None = None
    model: str | None = None
    response_metadata: dict[str, Any] = Field(default_factory=dict)
    usage: AdapterTokenUsage = AdapterTokenUsage()


class AdapterDescriptor(StrictModel):
    """Secret-free identity and configuration used to key resumable runs."""

    adapter_id: str = Field(min_length=1)
    model: str | None = None
    diagnostic_only: bool = False
    configuration: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class AgentSession(Protocol):
    def next(self, tool_results: tuple[AgentToolResult, ...]) -> AdapterTurn:
        """Produce the next assistant turn after receiving the last tool results."""


@runtime_checkable
class AgentAdapter(Protocol):
    """Boundary implemented by an LLM, internal PCA wrapper, or diagnostic oracle."""

    @property
    def descriptor(self) -> AdapterDescriptor:
        """Return stable, secret-free adapter identity and configuration."""

    def start(self, request: AgentRequest, *, seed: int | None) -> AgentSession:
        """Start an independent episode session."""


class ReferenceAgent:
    """Offline trace-replay oracle for diagnostics, never a fair model baseline."""

    def __init__(
        self,
        *,
        reference_trace: Sequence[ReferenceCall],
        answer: TypedValue,
        evidence_ids: Sequence[str],
    ) -> None:
        self._reference_trace = tuple(reference_trace)
        self._answer = answer
        self._evidence_ids = tuple(evidence_ids)
        self._descriptor = AdapterDescriptor(
            adapter_id="reference-agent",
            model="deterministic-trace-replay",
            diagnostic_only=True,
            configuration={"oracle_version": "1", "uses_gold_reference_trace": True},
        )

    @classmethod
    def from_task(cls, task: TaskArtifact) -> ReferenceAgent:
        """Construct the explicitly diagnostic oracle from one task's gold fields."""

        return cls(
            reference_trace=task.reference_trace,
            answer=task.gold_answer,
            evidence_ids=task.gold_evidence_ids,
        )

    @property
    def descriptor(self) -> AdapterDescriptor:
        return self._descriptor

    def start(self, request: AgentRequest, *, seed: int | None) -> AgentSession:
        del request, seed
        return _ReferenceSession(
            reference_trace=self._reference_trace,
            answer=self._answer,
            evidence_ids=self._evidence_ids,
        )


class _ReferenceSession:
    def __init__(
        self,
        *,
        reference_trace: tuple[ReferenceCall, ...],
        answer: TypedValue,
        evidence_ids: tuple[str, ...],
    ) -> None:
        self._reference_trace = reference_trace
        self._answer = answer
        self._evidence_ids = evidence_ids
        self._index = 0

    def next(self, tool_results: tuple[AgentToolResult, ...]) -> AdapterTurn:
        del tool_results
        if self._index < len(self._reference_trace):
            reference = self._reference_trace[self._index]
            self._index += 1
            return AdapterTurn(
                message="Diagnostic replay of a gold reference call.",
                tool_calls=(
                    AgentToolCall(
                        call_id=f"reference-call-{self._index}",
                        name=reference.tool,
                        arguments=copy.deepcopy(reference.model_arguments),
                    ),
                ),
                model="deterministic-trace-replay",
            )

        self._index += 1
        return AdapterTurn(
            message="Diagnostic replay submission.",
            tool_calls=(
                AgentToolCall(
                    call_id=f"reference-submit-{self._index}",
                    name="submit_answer",
                    arguments={
                        "answer": _typed_value_to_python(self._answer),
                        "evidence_ids": list(self._evidence_ids),
                    },
                ),
            ),
            model="deterministic-trace-replay",
        )


class ClinicalRetrievalAgent:
    """Deterministic, non-oracle parallel retrieval baseline for truth-state tasks."""

    def __init__(self) -> None:
        self._descriptor = AdapterDescriptor(
            adapter_id="clinical-retrieval-baseline",
            model="deterministic-public-tool-rules-v1",
            configuration={
                "uses_gold_fields": False,
                "parallel_domain_search": True,
                "classification_version": "3",
            },
        )

    @property
    def descriptor(self) -> AdapterDescriptor:
        return self._descriptor

    def start(self, request: AgentRequest, *, seed: int | None) -> AgentSession:
        del seed
        concept = _question_concept(request.question)
        available = {tool.get("name") for tool in request.tools}
        searches = [
            ("search_conditions", {"text": concept, "limit": 100}),
            ("search_observations", {"text": concept, "limit": 100}),
            ("search_medications", {"text": concept, "record_kind": "any", "limit": 100}),
            ("search_immunizations", {"text": concept, "limit": 100}),
            ("search_procedures", {"text": concept, "limit": 100}),
        ]
        return _ClinicalRetrievalSession(
            concept=concept,
            searches=tuple(item for item in searches if item[0] in available),
        )


class _ClinicalRetrievalSession:
    def __init__(self, *, concept: str, searches: tuple[tuple[str, dict[str, Any]], ...]) -> None:
        self._concept = concept.casefold()
        self._searches = searches
        self._searched = False

    def next(self, tool_results: tuple[AgentToolResult, ...]) -> AdapterTurn:
        if not self._searched:
            self._searched = True
            return AdapterTurn(
                message="Search all clinical domains for the patient-facing concept.",
                tool_calls=tuple(
                    AgentToolCall(
                        call_id=f"clinical-search-{index}",
                        name=name,
                        arguments=arguments,
                    )
                    for index, (name, arguments) in enumerate(self._searches, start=1)
                ),
            )

        records = [
            record
            for result in tool_results
            if result.result.get("ok")
            for record in result.result.get("records", [])
            if isinstance(record, dict)
            and str(record.get("display", "")).casefold() == self._concept
        ]
        classified = [(record, _clinical_record_polarity(record)) for record in records]
        polarities = {polarity for _, polarity in classified}
        if {"supported", "refuted"}.issubset(polarities):
            answer = "conflicting_evidence"
            accepted = {"supported", "refuted"}
        elif "refuted" in polarities:
            answer = "refuted"
            accepted = {"refuted"}
        elif "supported" in polarities:
            answer = "supported"
            accepted = {"supported"}
        else:
            answer = "insufficient_evidence"
            accepted = {"insufficient_evidence"}
        evidence_ids = sorted(
            {
                str(record["evidence_id"])
                for record, polarity in classified
                if polarity in accepted and isinstance(record.get("evidence_id"), str)
            }
        )
        return AdapterTurn(
            message="Aggregate status-bearing records and submit the evidence state.",
            tool_calls=(
                AgentToolCall(
                    call_id="clinical-submit",
                    name="submit_answer",
                    arguments={
                        "answer": {"kind": "truth_state", "value": answer},
                        "evidence_ids": evidence_ids,
                    },
                ),
            ),
        )


def _clinical_record_polarity(record: Mapping[str, Any]) -> str:
    resource_type = record.get("resource_type")
    attributes = record.get("attributes")
    attributes = attributes if isinstance(attributes, Mapping) else {}
    if resource_type == "Condition":
        verification = attributes.get("verificationStatus")
        status = verification.get("code") if isinstance(verification, Mapping) else None
        if status == "confirmed":
            return "supported"
        if status == "refuted":
            return "refuted"
        return "insufficient_evidence"
    status = record.get("status")
    if status is None:
        status = attributes.get("status")
    return lifecycle_polarity(
        str(resource_type),
        status if isinstance(status, str) else None,
        has_value=record.get("value") is not None,
    )


def _question_concept(question: str) -> str:
    patterns = (
        r"chart documents (.+?) as supported",
        r"final result for (.+?) as supported",
        r"whether (.+?) was actually administered as supported",
        r"whether (.+?) was ordered as supported",
        r"whether there is a currently actionable order for (.+?) as supported",
        r"whether (.+?) immunization was completed as supported",
    )
    for pattern in patterns:
        if match := re.search(pattern, question, flags=re.IGNORECASE):
            return re.sub(r"^the\s+", "", match.group(1).strip(), flags=re.IGNORECASE)
    return question.strip()


class OpenAIResponsesAgent:
    """OpenAI Responses function-calling adapter with an injectable SDK client."""

    def __init__(
        self,
        settings: OpenAISettings,
        *,
        sdk_client: Any | None = None,
        max_output_tokens: int = 2048,
        instructions: str = OPENAI_AGENT_INSTRUCTIONS,
        reasoning_effort: str = "medium",
        parallel_tool_calls: bool = False,
        service_tier: str = "auto",
        max_retries: int = 0,
        runtime_binding_hash: str | None = None,
    ) -> None:
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than zero")
        if not instructions.strip():
            raise ValueError("instructions must not be empty")
        if reasoning_effort not in {"none", "minimal", "low", "medium", "high", "xhigh"}:
            raise ValueError("unsupported reasoning effort")
        if service_tier not in {"auto", "default", "flex", "priority"}:
            raise ValueError("unsupported OpenAI service tier")
        if type(max_retries) is not int or max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer")
        if runtime_binding_hash is not None and not _is_sha256(runtime_binding_hash):
            raise ValueError("runtime_binding_hash must be a sha256 content hash")
        self.settings = settings
        self._sdk_client = (
            sdk_client
            if sdk_client is not None
            else OpenAI(
                api_key=settings.api_key,
                base_url=settings.base_url,
                timeout=settings.timeout_seconds,
                max_retries=max_retries,
            )
        )
        self._max_output_tokens = max_output_tokens
        self._instructions = instructions
        self._reasoning_effort = reasoning_effort
        self._parallel_tool_calls = parallel_tool_calls
        self._service_tier = service_tier
        runtime_configuration = (
            {"runtime_binding_hash": runtime_binding_hash}
            if runtime_binding_hash is not None
            else {}
        )
        self._descriptor = AdapterDescriptor(
            adapter_id="openai-responses",
            model=settings.model,
            configuration={
                "base_url_hash": content_hash(settings.base_url),
                "base_url_policy": "https_no_credentials_query_or_fragment",
                "instructions_hash": content_hash(instructions),
                "max_output_tokens": max_output_tokens,
                "max_retries": max_retries,
                "openai_sdk_version": _package_version("openai"),
                "stateless_function_loop": True,
                "store": False,
                "timeout_seconds": settings.timeout_seconds,
                "seed_applied": False,
                # ``model`` above is the requested identifier. The service-returned
                # identifier is recorded per response and is never synthesized from it.
                "model_identifier_is_service_resolved": False,
                "service_returned_model_recorded_per_response": True,
                "reasoning_effort": reasoning_effort,
                "parallel_tool_calls": parallel_tool_calls,
                "service_tier": service_tier,
                "tool_choice": "required",
                **runtime_configuration,
            },
        )

    @property
    def descriptor(self) -> AdapterDescriptor:
        return self._descriptor

    def start(self, request: AgentRequest, *, seed: int | None) -> AgentSession:
        # Responses currently has no seed parameter; the runner still records and keys it.
        del seed
        return _OpenAIResponsesSession(
            sdk_client=self._sdk_client,
            model=self.settings.model,
            instructions=self._instructions,
            max_output_tokens=self._max_output_tokens,
            request=request,
            reasoning_effort=self._reasoning_effort,
            parallel_tool_calls=self._parallel_tool_calls,
            service_tier=self._service_tier,
        )


class _OpenAIResponsesSession:
    def __init__(
        self,
        *,
        sdk_client: Any,
        model: str,
        instructions: str,
        max_output_tokens: int,
        request: AgentRequest,
        reasoning_effort: str,
        parallel_tool_calls: bool,
        service_tier: str,
    ) -> None:
        self._sdk_client = sdk_client
        self._model = model
        self._instructions = instructions
        self._max_output_tokens = max_output_tokens
        self._tools = [_openai_tool(tool) for tool in request.tools]
        self._input: list[Any] = [{"role": "user", "content": request.question}]
        self._reasoning_effort = reasoning_effort
        self._parallel_tool_calls = parallel_tool_calls
        self._service_tier = service_tier

    def next(self, tool_results: tuple[AgentToolResult, ...]) -> AdapterTurn:
        for tool_result in tool_results:
            output: FunctionCallOutput = {
                "type": "function_call_output",
                "call_id": tool_result.call_id,
                "output": _json_dumps(tool_result.result),
            }
            self._input.append(output)

        response = OpenAITextClient._call(
            self._sdk_client.responses.create,
            model=self._model,
            instructions=self._instructions,
            input=copy.deepcopy(self._input),
            include=["reasoning.encrypted_content"],
            tools=copy.deepcopy(self._tools),
            tool_choice="required",
            reasoning={"effort": self._reasoning_effort},
            parallel_tool_calls=self._parallel_tool_calls,
            service_tier=self._service_tier,
            max_output_tokens=self._max_output_tokens,
            store=False,
        )
        response_status = getattr(response, "status", "completed")
        if response_status != "completed":
            token_usage = _adapter_token_usage(response)
            returned_model = _returned_model(response)
            metadata = _response_metadata(
                response,
                token_usage=token_usage,
                returned_model=returned_model,
                requested_parallel_tool_calls=self._parallel_tool_calls,
                requested_reasoning_effort=self._reasoning_effort,
                requested_service_tier=self._service_tier,
            )
            incomplete_details = _response_object_payload(
                getattr(response, "incomplete_details", None)
            )
            partial_output = str(getattr(response, "output_text", "") or "")
            partial_output_items = [
                _response_input_item(item)
                for item in tuple(getattr(response, "output", ()) or ())
            ]
            metadata.update(
                {
                    "response_status": str(response_status),
                    "incomplete_details": incomplete_details,
                    "partial_output_present": bool(partial_output),
                    "partial_output_items": partial_output_items,
                }
            )
            reason = incomplete_details.get("reason")
            reason_suffix = f" ({reason})" if isinstance(reason, str) and reason else ""
            raise AgentAdapterError(
                "incomplete_response",
                f"The OpenAI response did not complete{reason_suffix}.",
                response_id=str(getattr(response, "id", "") or "") or None,
                model=returned_model,
                response_metadata=metadata,
                usage=token_usage,
                partial_output=partial_output,
            )

        output_items = tuple(getattr(response, "output", ()) or ())
        self._input.extend(_response_input_item(item) for item in output_items)

        calls: list[AgentToolCall] = []
        for item in output_items:
            if not isinstance(item, ResponseFunctionToolCall):
                continue
            try:
                arguments = json.loads(item.arguments)
            except json.JSONDecodeError as exc:
                raise AgentAdapterError(
                    "invalid_function_arguments",
                    "The OpenAI response contained invalid function arguments.",
                ) from exc
            if not isinstance(arguments, dict):
                raise AgentAdapterError(
                    "invalid_function_arguments",
                    "The OpenAI function arguments were not a JSON object.",
                )
            calls.append(
                AgentToolCall(
                    call_id=item.call_id,
                    name=item.name,
                    arguments=arguments,
                )
            )

        token_usage = _adapter_token_usage(response)
        returned_model = _returned_model(response)
        return AdapterTurn(
            message=str(getattr(response, "output_text", "") or ""),
            tool_calls=tuple(calls),
            response_id=str(getattr(response, "id", "") or "") or None,
            model=returned_model,
            response_metadata=_response_metadata(
                response,
                token_usage=token_usage,
                returned_model=returned_model,
                requested_parallel_tool_calls=self._parallel_tool_calls,
                requested_reasoning_effort=self._reasoning_effort,
                requested_service_tier=self._service_tier,
            ),
            usage=token_usage,
        )


CommandInvoker = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class CommandAgentAdapter:
    """JSON command boundary for an internal PCA wrapper.

    The injected callable may invoke a local process, an internal agent SDK, or a
    test double.  This package deliberately defines no private HTTP assumptions.
    """

    def __init__(
        self,
        *,
        adapter_id: str,
        invoke: CommandInvoker,
        model: str | None = None,
        configuration: Mapping[str, Any] | None = None,
    ) -> None:
        self._invoke = invoke
        self._descriptor = AdapterDescriptor(
            adapter_id=adapter_id,
            model=model,
            configuration=copy.deepcopy(dict(configuration or {})),
        )
        # Fail early if the claimed public configuration cannot be deterministically keyed.
        _json_dumps(self._descriptor.model_dump(mode="json"))

    @property
    def descriptor(self) -> AdapterDescriptor:
        return self._descriptor

    def start(self, request: AgentRequest, *, seed: int | None) -> AgentSession:
        return _CommandSession(
            invoke=self._invoke,
            descriptor=self._descriptor,
            request=request,
            seed=seed,
        )


class _CommandSession:
    def __init__(
        self,
        *,
        invoke: CommandInvoker,
        descriptor: AdapterDescriptor,
        request: AgentRequest,
        seed: int | None,
    ) -> None:
        self._invoke = invoke
        self._descriptor = descriptor
        self._request = request
        self._seed = seed
        self._history: list[dict[str, Any]] = []

    def next(self, tool_results: tuple[AgentToolResult, ...]) -> AdapterTurn:
        for result in tool_results:
            self._history.append(
                {
                    "role": "tool",
                    "call_id": result.call_id,
                    "name": result.name,
                    "result": copy.deepcopy(result.result),
                }
            )
        payload = {
            "schema_version": "patient-tool-gym.command-turn.v1",
            "question": self._request.question,
            "tools": copy.deepcopy(list(self._request.tools)),
            "history": copy.deepcopy(self._history),
            "seed": self._seed,
        }
        try:
            raw_turn = self._invoke(payload)
        except Exception as exc:
            raise AgentAdapterError(
                "command_invocation_failed",
                "The configured command/PCA adapter failed.",
            ) from exc
        try:
            turn = AdapterTurn.model_validate(dict(raw_turn))
        except (TypeError, ValueError) as exc:
            raise AgentAdapterError(
                "invalid_command_response",
                "The command/PCA adapter returned an invalid turn.",
            ) from exc
        self._history.append(
            {
                "role": "assistant",
                "message": turn.message,
                "tool_calls": [call.model_dump(mode="json") for call in turn.tool_calls],
            }
        )
        return turn


def _adapter_token_usage(response: Any) -> AdapterTokenUsage:
    usage = getattr(response, "usage", None)
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    return AdapterTokenUsage(
        input_tokens=getattr(usage, "input_tokens", None),
        cached_input_tokens=getattr(input_details, "cached_tokens", None),
        cache_write_input_tokens=getattr(input_details, "cache_write_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None),
        reasoning_output_tokens=getattr(output_details, "reasoning_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
    )


def _returned_model(response: Any) -> str | None:
    value = getattr(response, "model", None)
    return value if isinstance(value, str) and value else None


def _response_metadata(
    response: Any,
    *,
    token_usage: AdapterTokenUsage,
    returned_model: str | None,
    requested_parallel_tool_calls: bool,
    requested_reasoning_effort: str,
    requested_service_tier: str,
) -> dict[str, Any]:
    response_parallel = getattr(response, "parallel_tool_calls", None)
    parallel_value = (
        response_parallel if isinstance(response_parallel, bool) else requested_parallel_tool_calls
    )
    response_reasoning = getattr(getattr(response, "reasoning", None), "effort", None)
    reasoning_value = (
        response_reasoning
        if isinstance(response_reasoning, str) and response_reasoning
        else requested_reasoning_effort
    )
    response_tool_choice = getattr(response, "tool_choice", None)
    tool_choice_value = (
        response_tool_choice
        if isinstance(response_tool_choice, str) and response_tool_choice
        else "required"
    )
    response_service_tier = getattr(response, "service_tier", None)
    service_tier_value = (
        response_service_tier
        if isinstance(response_service_tier, str) and response_service_tier
        else requested_service_tier
    )
    return {
        "returned_model_identifier": returned_model,
        "returned_model_identifier_source": (
            "service_response" if returned_model is not None else "missing"
        ),
        "parallel_tool_calls": parallel_value,
        "parallel_tool_calls_source": (
            "service_response" if isinstance(response_parallel, bool) else "request"
        ),
        "reasoning_effort": reasoning_value,
        "reasoning_effort_source": (
            "service_response"
            if isinstance(response_reasoning, str) and response_reasoning
            else "request"
        ),
        "tool_choice": tool_choice_value,
        "tool_choice_source": (
            "service_response"
            if isinstance(response_tool_choice, str) and response_tool_choice
            else "request"
        ),
        "service_tier": service_tier_value,
        "service_tier_source": (
            "service_response"
            if isinstance(response_service_tier, str) and response_service_tier
            else "request"
        ),
        # Response-level accounting is required for cache and long-context billing.
        "token_usage": token_usage.model_dump(mode="json"),
    }


def _response_object_payload(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json", exclude_none=True)
        return dumped if isinstance(dumped, dict) else {"value": dumped}
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    reason = getattr(value, "reason", None)
    return {"reason": reason} if isinstance(reason, str) and reason else {}


def _openai_tool(public_contract: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one registry contract to the official Responses function shape."""

    return {
        "type": "function",
        "name": str(public_contract["name"]),
        "description": str(public_contract["description"]),
        "parameters": copy.deepcopy(public_contract["input_schema"]),
        # Registry/Pydantic validation is authoritative.  Current schemas contain
        # optional fields and are not OpenAI strict-schema compatible as written.
        "strict": False,
    }


def _response_input_item(item: Any) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        dumped = item.model_dump(mode="json", exclude_none=True)
    elif isinstance(item, Mapping):
        dumped = dict(item)
    else:
        raise AgentAdapterError(
            "invalid_response_item",
            "The OpenAI response contained an unsupported output item.",
        )
    if not isinstance(dumped, dict):
        raise AgentAdapterError(
            "invalid_response_item",
            "The OpenAI response contained an unsupported output item.",
        )
    return dumped


def _typed_value_to_python(value: TypedValue) -> Any:
    if value.kind == "quantity":
        return {"value": value.value, "unit": value.unit}
    return copy.deepcopy(value.submission_value())


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unavailable"


def _is_sha256(value: str) -> bool:
    if not value.startswith("sha256:") or len(value) != 71:
        return False
    try:
        int(value[7:], 16)
    except ValueError:
        return False
    return True
