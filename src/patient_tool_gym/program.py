from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from statistics import mean
from types import MappingProxyType
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from patient_tool_gym.evidence import (
    LEGACY_LOGICAL_GROUP_RULES_VERSION,
    LOGICAL_GROUP_RULES_VERSION,
    parse_clinical_instant,
)
from patient_tool_gym.tools import ToolRegistry

PathPart = str | int
PROGRAM_SIGNATURE_VERSION = "patient-tool-gym.program-signature.v4"
LEGACY_PROGRAM_SIGNATURE_VERSION = "patient-tool-gym.program-signature.v3"


@dataclass(frozen=True)
class MedicationDecisionRuleSemantics:
    endpoint: str
    supported_statuses: frozenset[str]
    refuted_statuses: frozenset[str]
    deprecated_for_new_tasks: bool = False
    replacement_rule_id: str | None = None


MEDICATION_DECISION_RULE_SEMANTICS = MappingProxyType(
    {
        # Frozen legacy behavior: despite its old name, this is a mixed
        # current-lifecycle endpoint, not historical evidence that an author once
        # issued an order. It remains executable for artifact replay only.
        "medication_ordered.v2": MedicationDecisionRuleSemantics(
            endpoint="legacy_current_lifecycle_not_historical_order_authorship",
            supported_statuses=frozenset({"active", "completed"}),
            refuted_statuses=frozenset({"cancelled", "stopped", "revoked"}),
            deprecated_for_new_tasks=True,
            replacement_rule_id="medication_order_actionable.v3",
        ),
        "medication_order_actionable.v3": MedicationDecisionRuleSemantics(
            endpoint="currently_actionable_order",
            supported_statuses=frozenset({"active"}),
            refuted_statuses=frozenset({"cancelled", "completed", "stopped"}),
        ),
        "medication_administered.v2": MedicationDecisionRuleSemantics(
            endpoint="any_real_world_administration_occurred",
            supported_statuses=frozenset({"completed", "stopped"}),
            refuted_statuses=frozenset({"not-done"}),
        ),
    }
)


class ProgramExecutionError(RuntimeError):
    """Raised when a valid task program cannot be executed."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProgramOutputSchema(StrictModel):
    """A value-free declaration of a program's semantic output shape.

    ``truth_state`` and ``clarification`` are semantic kinds: when they are the
    top-level answer, the public submission schema uses the corresponding typed
    envelope. Nested truth states remain their JSON string representation.
    """

    kind: Literal[
        "any",
        "null",
        "boolean",
        "integer",
        "number",
        "string",
        "quantity",
        "object",
        "array",
        "truth_state",
        "clarification",
        "union",
    ]
    fields: dict[str, ProgramOutputSchema] = Field(default_factory=dict)
    items: ProgramOutputSchema | None = None
    options: tuple[ProgramOutputSchema, ...] = ()

    @model_validator(mode="after")
    def validate_shape(self) -> ProgramOutputSchema:
        if self.kind == "object":
            if self.items is not None or self.options:
                raise ValueError("object output schemas may only declare fields")
            if any(not name for name in self.fields):
                raise ValueError("object output schema field names must not be empty")
            return self
        if self.kind == "array":
            if self.fields or self.options:
                raise ValueError("array output schemas may only declare items")
            return self
        if self.kind == "union":
            if self.fields or self.items is not None:
                raise ValueError("union output schemas may only declare options")
            if len(self.options) < 2:
                raise ValueError("union output schemas require at least two options")
            return self
        if self.fields or self.items is not None or self.options:
            raise ValueError(f"{self.kind} output schemas cannot declare child schemas")
        return self


class ValueRef(StrictModel):
    node_id: str = Field(min_length=1)
    path: tuple[PathPart, ...] = ()


class ToolNode(StrictModel):
    op: Literal["tool"] = "tool"
    id: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class SelectNode(StrictModel):
    op: Literal["select"] = "select"
    id: str = Field(min_length=1)
    source: ValueRef


class ExistsNode(StrictModel):
    op: Literal["exists"] = "exists"
    id: str = Field(min_length=1)
    source: ValueRef


class AggregateNode(StrictModel):
    op: Literal["aggregate"] = "aggregate"
    id: str = Field(min_length=1)
    function: Literal["count", "sum", "mean", "min", "max"]
    source: ValueRef


class CompareNode(StrictModel):
    op: Literal["compare"] = "compare"
    id: str = Field(min_length=1)
    operator: Literal["eq", "ne", "lt", "le", "gt", "ge"]
    left: ValueRef
    right: ValueRef | str | int | float | bool | None


class CalculateNode(StrictModel):
    op: Literal["calculate"] = "calculate"
    id: str = Field(min_length=1)
    function: Literal["bmi"]
    arguments: tuple[ValueRef, ...]


class BuildObjectNode(StrictModel):
    op: Literal["build_object"] = "build_object"
    id: str = Field(min_length=1)
    fields: dict[str, ValueRef]


class AllNode(StrictModel):
    op: Literal["all"] = "all"
    id: str = Field(min_length=1)
    sources: tuple[ValueRef, ...] = Field(min_length=2)


class TruthStateAggregateNode(StrictModel):
    """Combine named medical evidence states while preserving branch outcomes.

    ``all`` uses Belnap truth conjunction: supported=(1,0), refuted=(0,1),
    insufficient=(0,0), and conflicting=(1,1). Positive support is conjoined and
    negative support is disjoined. The result includes each named branch as well
    as the derived overall state, so aggregation cannot hide the failed branch.
    """

    op: Literal["truth_state_aggregate"] = "truth_state_aggregate"
    id: str = Field(min_length=1)
    function: Literal["all"] = "all"
    branches: dict[str, ValueRef] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_branch_names(self) -> TruthStateAggregateNode:
        if any(not name.strip() for name in self.branches):
            raise ValueError("truth-state branch names must not be empty")
        return self


class ChartAbsenceNode(StrictModel):
    op: Literal["chart_absence"] = "chart_absence"
    id: str = Field(min_length=1)
    source: ValueRef


class ClinicalRecordSelector(StrictModel):
    system: str = Field(min_length=1)
    code: str = Field(min_length=1)
    date_from: date | None = None
    date_to: date | None = None

    @model_validator(mode="after")
    def validate_date_range(self) -> ClinicalRecordSelector:
        if (
            self.date_from is not None
            and self.date_to is not None
            and self.date_from > self.date_to
        ):
            raise ValueError("selector date_from must not be after date_to")
        return self


class EvidenceStateNode(StrictModel):
    op: Literal["evidence_state"] = "evidence_state"
    id: str = Field(min_length=1)
    source: ValueRef
    decision_rule_id: Literal[
        "condition_documented.v1",
        "procedure_completed.v1",
        "immunization_completed.v1",
        "condition_documented.v2",
        "procedure_completed.v2",
        "immunization_completed.v2",
        "observation_final_result.v2",
        "medication_ordered.v2",
        "medication_order_actionable.v3",
        "medication_administered.v2",
    ]
    selector: ClinicalRecordSelector | None = None

    @model_validator(mode="after")
    def validate_selector_version(self) -> EvidenceStateNode:
        is_selector_rule = self.decision_rule_id.endswith((".v2", ".v3"))
        if is_selector_rule and self.selector is None:
            raise ValueError("v2 evidence-state and v3 evidence-state rules require a selector")
        if not is_selector_rule and self.selector is not None:
            raise ValueError("selectors require a v2/v3 evidence-state decision rule")
        return self


class VersionTrajectoryNode(StrictModel):
    op: Literal["version_trajectory"] = "version_trajectory"
    id: str = Field(min_length=1)
    source: ValueRef
    provenance_source: ValueRef | None = None


ProgramNode = Annotated[
    ToolNode
    | SelectNode
    | ExistsNode
    | AggregateNode
    | CompareNode
    | CalculateNode
    | BuildObjectNode
    | AllNode
    | TruthStateAggregateNode
    | ChartAbsenceNode
    | EvidenceStateNode
    | VersionTrajectoryNode,
    Field(discriminator="op"),
]


class TruthState(StrEnum):
    SUPPORTED = "supported"
    REFUTED = "refuted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"


def truth_state_all(states: Sequence[TruthState]) -> TruthState:
    """Return Belnap conjunction for one or more medical evidence states."""

    if not states:
        raise ValueError("truth-state conjunction requires at least one branch")
    bits = {
        TruthState.SUPPORTED: (True, False),
        TruthState.REFUTED: (False, True),
        TruthState.INSUFFICIENT_EVIDENCE: (False, False),
        TruthState.CONFLICTING_EVIDENCE: (True, True),
    }
    positive = all(bits[state][0] for state in states)
    negative = any(bits[state][1] for state in states)
    by_bits = {
        (True, False): TruthState.SUPPORTED,
        (False, True): TruthState.REFUTED,
        (False, False): TruthState.INSUFFICIENT_EVIDENCE,
        (True, True): TruthState.CONFLICTING_EVIDENCE,
    }
    return by_bits[(positive, negative)]


class ClarificationReason(StrEnum):
    MISSING_CLINICAL_SELECTOR = "missing_clinical_selector"


class TaskProgram(StrictModel):
    schema_version: Literal[
        "patient-tool-gym.program.v1",
        "patient-tool-gym.program.v2",
        "patient-tool-gym.program.v3",
        "patient-tool-gym.program.v4",
    ] = "patient-tool-gym.program.v1"
    nodes: tuple[ProgramNode, ...]
    output: ValueRef
    output_schema: ProgramOutputSchema | None = None

    @model_serializer(mode="wrap")
    def serialize_without_legacy_null_schema(self, handler: Any) -> dict[str, Any]:
        """Keep artifacts created before ``output_schema`` byte-shape compatible."""

        payload = handler(self)
        if self.output_schema is None:
            payload.pop("output_schema", None)
        return payload

    @model_validator(mode="after")
    def validate_graph(self) -> TaskProgram:
        seen: set[str] = set()
        for node in self.nodes:
            if node.id in seen:
                raise ValueError(f"duplicate program node id: {node.id}")
            missing = sorted(ref.node_id for ref in _node_refs(node) if ref.node_id not in seen)
            if missing:
                raise ValueError(
                    f"node {node.id!r} references unknown or forward node(s): {missing}"
                )
            seen.add(node.id)
        if self.output.node_id not in seen:
            raise ValueError(f"program output references unknown node: {self.output.node_id}")
        evidence_state_nodes = tuple(
            node for node in self.nodes if isinstance(node, EvidenceStateNode)
        )
        if evidence_state_nodes and self.schema_version not in {
            "patient-tool-gym.program.v2",
            "patient-tool-gym.program.v3",
            "patient-tool-gym.program.v4",
        }:
            raise ValueError("evidence_state nodes require patient-tool-gym.program.v2 or newer")
        if self.schema_version == "patient-tool-gym.program.v2" and any(
            node.decision_rule_id.endswith((".v2", ".v3"))
            for node in evidence_state_nodes
        ):
            raise ValueError(
                "v2/v3 evidence-state decision rules require patient-tool-gym.program.v3"
            )
        if self.schema_version == "patient-tool-gym.program.v3" and any(
            node.decision_rule_id.endswith(".v1") for node in evidence_state_nodes
        ):
            raise ValueError("program.v3 evidence_state nodes require v2/v3 decision rules")
        truth_state_aggregates = tuple(
            node for node in self.nodes if isinstance(node, TruthStateAggregateNode)
        )
        if truth_state_aggregates and self.schema_version != "patient-tool-gym.program.v4":
            raise ValueError("truth_state_aggregate nodes require patient-tool-gym.program.v4")
        if self.schema_version == "patient-tool-gym.program.v4" and any(
            node.decision_rule_id.endswith(".v1") for node in evidence_state_nodes
        ):
            raise ValueError("program.v4 evidence_state nodes require v2/v3 decision rules")
        if self.output_schema is not None:
            inferred = infer_program_output_schema(self)
            if self.output_schema != inferred:
                raise ValueError(
                    "declared output_schema must equal the schema inferred from program operations"
                )
        return self


def infer_program_output_schema(program: TaskProgram) -> ProgramOutputSchema:
    """Infer an answer schema solely from declared program operations.

    Unknown tool payload fields deliberately remain ``any``. Inference never
    executes the program and never inspects a task's gold answer instance.
    """

    schemas: dict[str, ProgramOutputSchema] = {}
    for node in program.nodes:
        if isinstance(node, ToolNode):
            schemas[node.id] = _tool_result_output_schema(node.tool)
        elif isinstance(node, SelectNode):
            schemas[node.id] = _schema_for_ref(node.source, schemas)
        elif isinstance(node, (ExistsNode, CompareNode, AllNode, ChartAbsenceNode)):
            schemas[node.id] = ProgramOutputSchema(kind="boolean")
        elif isinstance(node, AggregateNode):
            schemas[node.id] = ProgramOutputSchema(
                kind="integer" if node.function == "count" else "number"
            )
        elif isinstance(node, CalculateNode):
            schemas[node.id] = ProgramOutputSchema(kind="quantity")
        elif isinstance(node, BuildObjectNode):
            schemas[node.id] = ProgramOutputSchema(
                kind="object",
                fields={name: _schema_for_ref(ref, schemas) for name, ref in node.fields.items()},
            )
        elif isinstance(node, TruthStateAggregateNode):
            schemas[node.id] = ProgramOutputSchema(
                kind="object",
                fields={
                    "branches": ProgramOutputSchema(
                        kind="object",
                        fields={
                            name: ProgramOutputSchema(kind="truth_state")
                            for name in sorted(node.branches)
                        },
                    ),
                    "overall": ProgramOutputSchema(kind="truth_state"),
                },
            )
        elif isinstance(node, EvidenceStateNode):
            schemas[node.id] = ProgramOutputSchema(kind="truth_state")
        elif isinstance(node, VersionTrajectoryNode):
            schemas[node.id] = _version_trajectory_output_schema()
        else:  # pragma: no cover - the ProgramNode union makes this defensive.
            schemas[node.id] = ProgramOutputSchema(kind="any")
    return _schema_for_ref(program.output, schemas)


def program_output_schema(program: TaskProgram) -> ProgramOutputSchema:
    """Return a program-derived schema and reject arbitrary explicit overrides."""

    inferred = infer_program_output_schema(program)
    if program.output_schema is not None and program.output_schema != inferred:
        # ``model_copy(update=...)`` does not re-run Pydantic validation. Keep the
        # public schema boundary fail-closed even for program objects made that way.
        raise ValueError(
            "declared output_schema must equal the schema inferred from program operations"
        )
    return program.output_schema or inferred


def program_submission_answer_schema(
    program: TaskProgram,
    *,
    allow_clarification: bool = False,
) -> dict[str, Any]:
    """Render the program's public ``submit_answer.answer`` JSON Schema."""

    schema = program_output_schema(program)
    if allow_clarification and not _accepts_top_level_clarification(schema):
        schema = ProgramOutputSchema(
            kind="union",
            options=(schema, ProgramOutputSchema(kind="clarification")),
        )
    return output_schema_to_json_schema(schema)


def output_schema_to_json_schema(
    schema: ProgramOutputSchema,
    *,
    top_level: bool = True,
) -> dict[str, Any]:
    """Render a value-free semantic output declaration as JSON Schema."""

    if schema.kind == "any":
        return {}
    if schema.kind in {"null", "boolean", "integer", "number", "string"}:
        return {"type": schema.kind}
    if schema.kind == "truth_state":
        values = [state.value for state in TruthState]
        if not top_level:
            return {"type": "string", "enum": values}
        return _typed_envelope_schema(kind="truth_state", values=values)
    if schema.kind == "clarification":
        values = [reason.value for reason in ClarificationReason]
        if not top_level:
            return {"type": "string", "enum": values}
        return _typed_envelope_schema(kind="clarification", values=values)
    if schema.kind == "quantity":
        return {
            "type": "object",
            "properties": {
                "value": {"type": "number"},
                "unit": {"type": "string"},
            },
            "required": ["value", "unit"],
            "additionalProperties": False,
        }
    if schema.kind == "array":
        rendered: dict[str, Any] = {"type": "array"}
        if schema.items is not None:
            rendered["items"] = output_schema_to_json_schema(
                schema.items,
                top_level=False,
            )
        return rendered
    if schema.kind == "object":
        return {
            "type": "object",
            "properties": {
                name: output_schema_to_json_schema(child, top_level=False)
                for name, child in schema.fields.items()
            },
            "required": sorted(schema.fields),
            "additionalProperties": False,
        }
    if schema.kind == "union":
        return {
            "anyOf": [
                output_schema_to_json_schema(option, top_level=top_level)
                for option in schema.options
            ]
        }
    raise ValueError(f"unknown output schema kind: {schema.kind}")


def _typed_envelope_schema(*, kind: str, values: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "const": kind},
            "value": {"type": "string", "enum": values},
        },
        "required": ["kind", "value"],
        "additionalProperties": False,
    }


def _accepts_top_level_clarification(schema: ProgramOutputSchema) -> bool:
    if schema.kind == "clarification":
        return True
    return schema.kind == "union" and any(
        _accepts_top_level_clarification(option) for option in schema.options
    )


def _schema_for_ref(
    ref: ValueRef,
    schemas: dict[str, ProgramOutputSchema],
) -> ProgramOutputSchema:
    schema = schemas.get(ref.node_id, ProgramOutputSchema(kind="any"))
    for part in ref.path:
        if schema.kind == "object" and isinstance(part, str):
            schema = schema.fields.get(part, ProgramOutputSchema(kind="any"))
        elif schema.kind == "array" and isinstance(part, int):
            schema = schema.items or ProgramOutputSchema(kind="any")
        elif schema.kind == "quantity" and part == "value":
            schema = ProgramOutputSchema(kind="number")
        elif schema.kind == "quantity" and part == "unit":
            schema = ProgramOutputSchema(kind="string")
        else:
            return ProgramOutputSchema(kind="any")
    return schema


def _tool_result_output_schema(tool: str) -> ProgramOutputSchema:
    if tool == "submit_answer":
        return ProgramOutputSchema(kind="any")
    return ProgramOutputSchema(
        kind="object",
        fields={
            "ok": ProgramOutputSchema(kind="boolean"),
            "records": ProgramOutputSchema(
                kind="array",
                items=ProgramOutputSchema(kind="any"),
            ),
            "searched_scope": ProgramOutputSchema(kind="any"),
            "total_matches": ProgramOutputSchema(kind="integer"),
            "complete": ProgramOutputSchema(kind="boolean"),
            "next_cursor": ProgramOutputSchema(
                kind="union",
                options=(
                    ProgramOutputSchema(kind="string"),
                    ProgramOutputSchema(kind="null"),
                ),
            ),
            "snapshot_id": ProgramOutputSchema(kind="string"),
            "error": ProgramOutputSchema(kind="any"),
        },
    )


def _version_trajectory_output_schema() -> ProgramOutputSchema:
    version = ProgramOutputSchema(
        kind="object",
        fields={
            "version_id": ProgramOutputSchema(kind="string"),
            "last_updated": ProgramOutputSchema(kind="string"),
            "status": ProgramOutputSchema(kind="any"),
            "value": ProgramOutputSchema(kind="any"),
        },
    )
    return ProgramOutputSchema(
        kind="object",
        fields={
            "logical_resource_id": ProgramOutputSchema(kind="string"),
            "version_count": ProgramOutputSchema(kind="integer"),
            "earliest": version,
            "latest": version,
            "answer_changed": ProgramOutputSchema(kind="boolean"),
        },
    )


class TypedValue(StrictModel):
    kind: Literal[
        "null",
        "boolean",
        "integer",
        "number",
        "string",
        "quantity",
        "object",
        "list",
        "truth_state",
        "clarification",
    ]
    value: Any = None
    unit: str | None = None

    @model_validator(mode="after")
    def validate_value(self) -> TypedValue:
        checks = {
            "null": self.value is None,
            "boolean": isinstance(self.value, bool),
            "integer": isinstance(self.value, int) and not isinstance(self.value, bool),
            "number": isinstance(self.value, (int, float)) and not isinstance(self.value, bool),
            "string": isinstance(self.value, str),
            "quantity": isinstance(self.value, (int, float))
            and not isinstance(self.value, bool)
            and bool(self.unit),
            "object": isinstance(self.value, dict),
            "list": isinstance(self.value, list),
            "truth_state": isinstance(self.value, str)
            and self.value in {state.value for state in TruthState},
            "clarification": isinstance(self.value, str)
            and self.value in {reason.value for reason in ClarificationReason},
        }
        if not checks[self.kind]:
            raise ValueError(f"value does not match declared kind {self.kind!r}")
        if self.kind != "quantity" and self.unit is not None:
            raise ValueError("unit is only valid for quantity values")
        return self

    @classmethod
    def from_python(cls, value: Any) -> TypedValue:
        if isinstance(value, TruthState):
            return cls(kind="truth_state", value=value.value)
        if isinstance(value, ClarificationReason):
            return cls(kind="clarification", value=value.value)
        if value is None:
            return cls(kind="null", value=None)
        if isinstance(value, bool):
            return cls(kind="boolean", value=value)
        if isinstance(value, int):
            return cls(kind="integer", value=value)
        if isinstance(value, float):
            return cls(kind="number", value=value)
        if isinstance(value, str):
            return cls(kind="string", value=value)
        if isinstance(value, dict):
            if (
                set(value).issuperset({"value", "unit"})
                and isinstance(value["value"], (int, float))
                and not isinstance(value["value"], bool)
                and isinstance(value["unit"], str)
            ):
                return cls(kind="quantity", value=value["value"], unit=value["unit"])
            return cls(kind="object", value=value)
        if isinstance(value, (list, tuple)):
            return cls(kind="list", value=list(value))
        raise ProgramExecutionError(f"unsupported answer type: {type(value).__name__}")

    @classmethod
    def from_submission(cls, value: Any) -> TypedValue:
        if (
            isinstance(value, dict)
            and value.get("kind") in {"truth_state", "clarification"}
            and set(value) == {"kind", "value"}
        ):
            return cls.model_validate(value)
        return cls.from_python(value)

    def submission_value(self) -> Any:
        if self.kind in {"truth_state", "clarification"}:
            return {"kind": self.kind, "value": self.value}
        if self.kind == "quantity":
            return {"value": self.value, "unit": self.unit}
        return self.value


class ReferenceCall(StrictModel):
    step: int = Field(ge=1)
    node_id: str
    tool: str
    model_arguments: dict[str, Any]
    resolved_arguments: dict[str, Any]
    depends_on: tuple[str, ...] = ()
    result_hash: str
    evidence_ids: tuple[str, ...] = ()
    complete: bool = True


class ProgramComplexity(StrictModel):
    dependency_depth: int = Field(ge=0)
    independent_width: int = Field(ge=0)
    evidence_width: int = Field(ge=0)
    answer_width: int = Field(ge=0)


class ProgramResult(StrictModel):
    answer: TypedValue
    evidence_ids: tuple[str, ...]
    reference_trace: tuple[ReferenceCall, ...]
    complexity: ProgramComplexity


@dataclass(frozen=True)
class _RuntimeValue:
    value: Any
    evidence_ids: frozenset[str]
    dependency_evidence_ids: frozenset[str] = frozenset()


def binding(node_id: str, *path: PathPart) -> dict[str, Any]:
    """Create a JSON-serializable tool-argument binding to a prior node."""

    return {"$ref": node_id, "path": list(path)}


def canonical_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def program_signature(program: TaskProgram) -> str:
    """Hash program structure while replacing patient-specific argument literals."""

    signature_version = (
        PROGRAM_SIGNATURE_VERSION
        if program.output_schema is not None
        else LEGACY_PROGRAM_SIGNATURE_VERSION
    )
    node_signatures: dict[str, str] = {}
    shaped_nodes: list[dict[str, Any]] = []
    for node in program.nodes:
        payload = _canonicalize_references(node.model_dump(mode="json"), node_signatures)
        payload.pop("id")
        if isinstance(node, ToolNode):
            payload["arguments"] = _argument_shape(payload["arguments"])
        if isinstance(node, EvidenceStateNode) and payload.get("selector") is not None:
            payload["selector"] = _argument_shape(payload["selector"])
        if isinstance(node, AllNode):
            payload["sources"] = sorted(payload["sources"], key=canonical_json)
        if isinstance(node, TruthStateAggregateNode):
            payload["branches"] = {
                name: payload["branches"][name] for name in sorted(payload["branches"])
            }
        node_signatures[node.id] = content_hash(
            {"signature_version": signature_version, "node": payload}
        )
        shaped_nodes.append(payload)
    shaped_nodes.sort(key=canonical_json)
    signature_payload = {
        "signature_version": signature_version,
        "schema_version": program.schema_version,
        "nodes": shaped_nodes,
        "output": _canonicalize_references(program.output.model_dump(mode="json"), node_signatures),
    }
    if program.output_schema is not None:
        signature_payload["output_schema"] = program.output_schema.model_dump(mode="json")
    return content_hash(signature_payload)


class ProgramExecutor:
    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def execute(self, program: TaskProgram) -> ProgramResult:
        values: dict[str, _RuntimeValue] = {}
        trace: list[ReferenceCall] = []
        logical_group_rules_version = self._registry.snapshot.logical_group_rules_version
        if logical_group_rules_version == LOGICAL_GROUP_RULES_VERSION:
            identity_scoped_conflicts = True
        elif logical_group_rules_version == LEGACY_LOGICAL_GROUP_RULES_VERSION:
            # Frozen v2 snapshots predate explicit FHIR business-identifier grouping.
            # Preserve their recorded global support+refute interpretation only for
            # reproducible replay. All newly normalized v3 snapshots require one shared
            # logical identity before two records can constitute a contradiction.
            identity_scoped_conflicts = False
        else:
            raise ProgramExecutionError(
                "unsupported logical-group rules version: "
                f"{logical_group_rules_version}"
            )

        for node in program.nodes:
            if isinstance(node, ToolNode):
                model_arguments = _resolve_arguments(node.arguments, values)
                result = self._registry.call(node.tool, model_arguments)
                if not result.get("ok"):
                    raise ProgramExecutionError(
                        f"tool node {node.id!r} failed: {result.get('error')}"
                    )
                returned_evidence_ids = frozenset(_collect_evidence_ids(result))
                dependency_evidence_ids = frozenset().union(
                    *(_resolve_ref(ref, values).evidence_ids for ref in _node_refs(node))
                )
                values[node.id] = _RuntimeValue(
                    result,
                    returned_evidence_ids | dependency_evidence_ids,
                    dependency_evidence_ids,
                )
                dependencies = tuple(sorted(ref.node_id for ref in _node_refs(node)))
                trace.append(
                    ReferenceCall(
                        step=len(trace) + 1,
                        node_id=node.id,
                        tool=node.tool,
                        model_arguments=model_arguments,
                        resolved_arguments={
                            "patient_id": self._registry.patient_id,
                            "snapshot_id": self._registry.snapshot_id,
                            **model_arguments,
                        },
                        depends_on=dependencies,
                        result_hash=content_hash(result),
                        evidence_ids=tuple(sorted(returned_evidence_ids)),
                        complete=bool(result.get("complete", True)),
                    )
                )
                continue

            values[node.id] = self._execute_local(
                node,
                values,
                identity_scoped_conflicts=identity_scoped_conflicts,
            )

        output = _resolve_ref(program.output, values)
        complexity = analyze_complexity(program, output.evidence_ids, output.value)
        return ProgramResult(
            answer=TypedValue.from_python(output.value),
            evidence_ids=tuple(sorted(output.evidence_ids)),
            reference_trace=tuple(trace),
            complexity=complexity,
        )

    def _execute_local(
        self,
        node: ProgramNode,
        values: dict[str, _RuntimeValue],
        *,
        identity_scoped_conflicts: bool,
    ) -> _RuntimeValue:
        if isinstance(node, SelectNode):
            return _resolve_ref(node.source, values)
        if isinstance(node, ExistsNode):
            source = _resolve_ref(node.source, values)
            return _RuntimeValue(bool(source.value), source.evidence_ids)
        if isinstance(node, AggregateNode):
            source = _resolve_ref(node.source, values)
            collection = source.value
            if not isinstance(collection, (list, tuple)):
                raise ProgramExecutionError(f"aggregate node {node.id!r} requires a collection")
            if node.function == "count":
                value: Any = len(collection)
            else:
                if not collection:
                    raise ProgramExecutionError(
                        f"aggregate node {node.id!r} cannot {node.function} an empty collection"
                    )
                if not all(
                    isinstance(item, (int, float)) and not isinstance(item, bool)
                    for item in collection
                ):
                    raise ProgramExecutionError(
                        f"aggregate node {node.id!r} requires numeric values"
                    )
                functions = {"sum": sum, "mean": mean, "min": min, "max": max}
                value = functions[node.function](collection)
            return _RuntimeValue(value, source.evidence_ids)
        if isinstance(node, CompareNode):
            left = _resolve_ref(node.left, values)
            if isinstance(node.right, ValueRef):
                right_runtime = _resolve_ref(node.right, values)
                right = right_runtime.value
                evidence = left.evidence_ids | right_runtime.evidence_ids
            else:
                right = node.right
                evidence = left.evidence_ids
            operations = {
                "eq": lambda a, b: a == b,
                "ne": lambda a, b: a != b,
                "lt": lambda a, b: a < b,
                "le": lambda a, b: a <= b,
                "gt": lambda a, b: a > b,
                "ge": lambda a, b: a >= b,
            }
            try:
                value = operations[node.operator](left.value, right)
            except TypeError as exc:
                raise ProgramExecutionError(
                    f"compare node {node.id!r} has incompatible values"
                ) from exc
            return _RuntimeValue(value, evidence)
        if isinstance(node, CalculateNode):
            args = [_resolve_ref(ref, values) for ref in node.arguments]
            evidence = frozenset().union(*(arg.evidence_ids for arg in args))
            if node.function == "bmi":
                if len(args) != 2:
                    raise ProgramExecutionError("bmi requires height and weight")
                height_m = _quantity_as(args[0].value, "m")
                weight_kg = _quantity_as(args[1].value, "kg")
                value = {"value": round(weight_kg / (height_m**2), 2), "unit": "kg/m2"}
                return _RuntimeValue(value, evidence)
        if isinstance(node, BuildObjectNode):
            resolved = {name: _resolve_ref(ref, values) for name, ref in node.fields.items()}
            evidence = frozenset().union(*(item.evidence_ids for item in resolved.values()))
            return _RuntimeValue({name: item.value for name, item in resolved.items()}, evidence)
        if isinstance(node, AllNode):
            resolved = [_resolve_ref(ref, values) for ref in node.sources]
            if not all(isinstance(item.value, bool) for item in resolved):
                raise ProgramExecutionError("all node requires boolean inputs")
            evidence = frozenset().union(*(item.evidence_ids for item in resolved))
            return _RuntimeValue(all(item.value for item in resolved), evidence)
        if isinstance(node, TruthStateAggregateNode):
            resolved = {
                name: _resolve_ref(reference, values)
                for name, reference in sorted(node.branches.items())
            }
            if not all(isinstance(item.value, TruthState) for item in resolved.values()):
                raise ProgramExecutionError(
                    "truth-state aggregate requires truth_state inputs"
                )
            branch_states = {name: item.value for name, item in resolved.items()}
            evidence = frozenset().union(*(item.evidence_ids for item in resolved.values()))
            return _RuntimeValue(
                {
                    "branches": branch_states,
                    "overall": truth_state_all(tuple(branch_states.values())),
                },
                evidence,
            )
        if isinstance(node, EvidenceStateNode):
            source = _resolve_ref(node.source, values)
            return _execute_evidence_state(
                node,
                source,
                identity_scoped_conflicts=identity_scoped_conflicts,
            )
        if isinstance(node, VersionTrajectoryNode):
            source = _resolve_ref(node.source, values)
            provenance = (
                _resolve_ref(node.provenance_source, values)
                if node.provenance_source is not None
                else None
            )
            return _execute_version_trajectory(node, source, provenance)
        if isinstance(node, ChartAbsenceNode):
            source = _resolve_ref(node.source, values)
            result = source.value
            if not isinstance(result, dict):
                raise ProgramExecutionError("chart absence requires a tool result")
            if not result.get("ok") or not result.get("complete"):
                raise ProgramExecutionError("chart absence requires a complete successful search")
            records = result.get("records")
            if not isinstance(records, list):
                raise ProgramExecutionError("chart absence search did not return a record list")
            return _RuntimeValue(len(records) == 0, source.evidence_ids)
        raise ProgramExecutionError(f"unsupported node type: {type(node).__name__}")


def _execute_version_trajectory(
    node: VersionTrajectoryNode,
    source: _RuntimeValue,
    provenance_source: _RuntimeValue | None,
) -> _RuntimeValue:
    result = source.value
    if (
        not isinstance(result, dict)
        or result.get("ok") is not True
        or result.get("complete") is not True
    ):
        raise ProgramExecutionError("version trajectory requires a complete successful search")
    records = result.get("records")
    if (
        not isinstance(records, list)
        or not records
        or not all(isinstance(record, dict) for record in records)
    ):
        raise ProgramExecutionError("version trajectory requires versioned records")
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        evidence_id = record.get("evidence_id")
        attributes = record.get("attributes")
        meta = attributes.get("resource_meta") if isinstance(attributes, dict) else None
        if (
            isinstance(evidence_id, str)
            and ":version:" in evidence_id
            and isinstance(meta, dict)
            and meta.get("versionId")
        ):
            groups.setdefault(evidence_id.rsplit(":version:", 1)[0], []).append(record)
    candidates = [group for group in groups.values() if len(group) >= 2]
    if len(candidates) != 1:
        raise ProgramExecutionError("version trajectory requires one versioned logical resource")
    records = candidates[0]
    logical_resource_types = {
        str(record.get("resource_type"))
        for record in records
        if isinstance(record.get("resource_type"), str)
    }
    if len(logical_resource_types) != 1:
        raise ProgramExecutionError("version trajectory requires one resource type")
    logical_resource_type = next(iter(logical_resource_types))
    logical_resource_id = next(
        key.rsplit(":", 1)[-1] for key, value in groups.items() if value is records
    )

    def version_key(record: dict[str, Any]) -> tuple[datetime, tuple[int, int | str]]:
        attributes = record.get("attributes")
        meta = attributes.get("resource_meta") if isinstance(attributes, dict) else None
        updated = meta.get("lastUpdated") if isinstance(meta, dict) else None
        if not isinstance(updated, str) or not updated:
            raise ProgramExecutionError("version trajectory requires meta.lastUpdated")
        version = meta.get("versionId")
        if not isinstance(version, str) or not version:
            raise ProgramExecutionError("version trajectory requires meta.versionId")
        instant = parse_clinical_instant(updated)
        if instant is None:
            raise ProgramExecutionError("version trajectory has invalid meta.lastUpdated")
        version_key: tuple[int, int | str]
        try:
            version_key = (0, int(version))
        except ValueError:
            version_key = (1, version)
        return instant, version_key

    ordered = sorted(records, key=version_key)

    def summary(record: dict[str, Any]) -> dict[str, Any]:
        meta = record["attributes"]["resource_meta"]
        return {
            "version_id": meta["versionId"],
            "last_updated": meta["lastUpdated"],
            "status": record.get("status"),
            "value": record.get("value"),
        }

    earliest = summary(ordered[0])
    latest = summary(ordered[-1])
    provenance_ids: frozenset[str] = frozenset()
    if node.provenance_source is not None:
        if provenance_source is None:
            raise ProgramExecutionError("version trajectory provenance dependency is missing")
        provenance_result = provenance_source.value
        if (
            not isinstance(provenance_result, dict)
            or provenance_result.get("ok") is not True
            or provenance_result.get("complete") is not True
        ):
            raise ProgramExecutionError(
                "version trajectory requires a complete successful Provenance search"
            )
        provenance_records = provenance_result.get("records")
        if not isinstance(provenance_records, list) or len(provenance_records) != 1:
            raise ProgramExecutionError("version trajectory requires one revision Provenance")
        provenance = provenance_records[0]
        if not isinstance(provenance, dict) or provenance.get("resource_type") != "Provenance":
            raise ProgramExecutionError("version trajectory received invalid Provenance evidence")
        links = provenance.get("links")
        target_ids = (
            {
                link.get("target_evidence_id")
                for link in links
                if isinstance(link, dict) and link.get("relation") == "target"
            }
            if isinstance(links, list)
            else set()
        )
        latest_evidence_id = ordered[-1].get("evidence_id")
        attributes = provenance.get("attributes")
        entities = attributes.get("entities") if isinstance(attributes, dict) else None
        revision_references = (
            {
                entity.get("what", {}).get("reference")
                for entity in entities
                if isinstance(entity, dict)
                and entity.get("role") == "revision"
                and isinstance(entity.get("what"), dict)
            }
            if isinstance(entities, list)
            else set()
        )
        expected_revision_reference = (
            f"{logical_resource_type}/{logical_resource_id}/_history/{earliest['version_id']}"
        )
        if latest_evidence_id not in target_ids or not any(
            isinstance(reference, str)
            and (
                reference == expected_revision_reference
                or reference.endswith(f"/{expected_revision_reference}")
            )
            for reference in revision_references
        ):
            raise ProgramExecutionError("Provenance does not link latest and prior versions")
        provenance_ids = provenance_source.evidence_ids
    return _RuntimeValue(
        {
            "logical_resource_id": logical_resource_id,
            "version_count": len(ordered),
            "earliest": earliest,
            "latest": latest,
            "answer_changed": (earliest["status"], earliest["value"])
            != (latest["status"], latest["value"]),
        },
        frozenset(
            record["evidence_id"]
            for record in records
            if isinstance(record.get("evidence_id"), str)
        )
        | provenance_ids,
    )


def _execute_evidence_state(
    node: EvidenceStateNode,
    source: _RuntimeValue,
    *,
    identity_scoped_conflicts: bool,
) -> _RuntimeValue:
    result = source.value
    if not isinstance(result, dict):
        raise ProgramExecutionError("evidence state requires a complete tool result")
    if result.get("ok") is not True or result.get("complete") is not True:
        raise ProgramExecutionError("evidence state requires a complete successful search")
    records = result.get("records")
    if not isinstance(records, list):
        raise ProgramExecutionError("evidence state search did not return a record list")
    if not all(isinstance(record, dict) for record in records):
        raise ProgramExecutionError("evidence state search returned an invalid record")

    selected_records = (
        select_clinical_records(records, node.selector)
        if node.selector is not None
        else tuple(records)
    )
    selected_evidence_ids = (
        frozenset(
            record["evidence_id"]
            for record in selected_records
            if isinstance(record.get("evidence_id"), str)
        )
        | source.dependency_evidence_ids
        if node.selector is not None
        else source.evidence_ids
    )
    if not selected_records:
        return _RuntimeValue(TruthState.INSUFFICIENT_EVIDENCE, selected_evidence_ids)
    if node.selector is not None and any(
        not isinstance(record.get("evidence_id"), str) for record in selected_records
    ):
        raise ProgramExecutionError("selected evidence records require evidence_id values")
    states_by_identity: dict[str, set[TruthState]] = {}
    states: set[TruthState] = set()
    for record in selected_records:
        record_state = _classify_evidence_record(node.decision_rule_id, record)
        states.add(record_state)
        logical_group = record.get("logical_evidence_group")
        if not isinstance(logical_group, str) or not logical_group:
            evidence_id = record.get("evidence_id")
            logical_group = (
                f"record:{evidence_id}"
                if isinstance(evidence_id, str)
                else f"anonymous:{len(states_by_identity)}"
            )
        states_by_identity.setdefault(logical_group, set()).add(record_state)
    identity_conflict = any(
        TruthState.SUPPORTED in identity_states
        and TruthState.REFUTED in identity_states
        for identity_states in states_by_identity.values()
    )
    legacy_global_conflict = (
        not identity_scoped_conflicts
        and TruthState.SUPPORTED in states
        and TruthState.REFUTED in states
    )
    if identity_conflict or legacy_global_conflict:
        state = TruthState.CONFLICTING_EVIDENCE
    elif TruthState.SUPPORTED in states:
        state = TruthState.SUPPORTED
    elif TruthState.REFUTED in states:
        state = TruthState.REFUTED
    else:
        state = TruthState.INSUFFICIENT_EVIDENCE
    return _RuntimeValue(state, selected_evidence_ids)


def select_clinical_records(
    records: Sequence[dict[str, Any]],
    selector: ClinicalRecordSelector,
) -> tuple[dict[str, Any], ...]:
    """Select exact code/date matches from a complete broad clinical search result."""

    return tuple(record for record in records if _record_matches_selector(record, selector))


def _record_matches_selector(
    record: dict[str, Any],
    selector: ClinicalRecordSelector,
) -> bool:
    if not _concept_has_coding(record.get("code"), selector.system, selector.code):
        return False
    if selector.date_from is None and selector.date_to is None:
        return True
    clinical_date = _record_clinical_date(record)
    if clinical_date is None:
        return False
    if selector.date_from is not None and clinical_date < selector.date_from:
        return False
    if selector.date_to is not None and clinical_date > selector.date_to:
        return False
    return True


def _record_clinical_date(record: dict[str, Any]) -> date | None:
    time = record.get("time")
    if not isinstance(time, dict):
        return None
    for field_name in ("start", "recorded", "issued", "authored", "onset", "onset_start"):
        instant = time.get(field_name)
        if not isinstance(instant, str):
            continue
        try:
            return date.fromisoformat(instant[:10])
        except ValueError:
            return None
    return None


def _concept_has_coding(value: Any, system: str, code: str) -> bool:
    if not isinstance(value, dict):
        return False
    if value.get("system") == system and value.get("code") == code:
        return True
    codings = value.get("codings")
    return isinstance(codings, list) and any(
        isinstance(coding, dict) and coding.get("system") == system and coding.get("code") == code
        for coding in codings
    )


def _classify_evidence_record(
    rule_id: str,
    record: dict[str, Any],
) -> TruthState:
    expected_resource_type = {
        "condition_documented.v1": "Condition",
        "procedure_completed.v1": "Procedure",
        "immunization_completed.v1": "Immunization",
        "condition_documented.v2": "Condition",
        "procedure_completed.v2": "Procedure",
        "immunization_completed.v2": "Immunization",
        "observation_final_result.v2": "Observation",
        "medication_ordered.v2": "MedicationRequest",
        "medication_order_actionable.v3": "MedicationRequest",
    }.get(rule_id)
    if rule_id == "medication_administered.v2":
        expected_resource_types = {"MedicationRequest", "MedicationAdministration"}
    elif expected_resource_type is None:
        raise ProgramExecutionError(f"unknown evidence decision rule: {rule_id}")
    else:
        expected_resource_types = {expected_resource_type}
    if record.get("resource_type") not in expected_resource_types:
        raise ProgramExecutionError("evidence decision rule received the wrong resource type")

    attributes = record.get("attributes")
    attributes = attributes if isinstance(attributes, dict) else {}
    if rule_id in {"condition_documented.v1", "condition_documented.v2"}:
        verification = _concept_code(attributes.get("verificationStatus"))
        if verification == "confirmed":
            return TruthState.SUPPORTED
        if verification == "refuted":
            return TruthState.REFUTED
        return TruthState.INSUFFICIENT_EVIDENCE

    status = record.get("status")
    if rule_id in {
        "procedure_completed.v1",
        "procedure_completed.v2",
        "immunization_completed.v1",
        "immunization_completed.v2",
    }:
        if status == "completed":
            return TruthState.SUPPORTED
        if status == "not-done":
            return TruthState.REFUTED
        return TruthState.INSUFFICIENT_EVIDENCE
    if rule_id == "observation_final_result.v2":
        status = record.get("status")
        attributes = record.get("attributes")
        attributes = attributes if isinstance(attributes, dict) else {}
        has_value = record.get("value") is not None
        has_absent_reason = isinstance(attributes.get("data_absent_reason"), dict)
        if status in {"final", "amended", "corrected"} and has_value:
            return TruthState.SUPPORTED
        if status == "cancelled":
            return TruthState.REFUTED
        if has_absent_reason or status in {
            "registered",
            "preliminary",
            "entered-in-error",
            "unknown",
        }:
            return TruthState.INSUFFICIENT_EVIDENCE
        return TruthState.INSUFFICIENT_EVIDENCE
    if rule_id == "medication_ordered.v2":
        status = record.get("status")
        semantics = MEDICATION_DECISION_RULE_SEMANTICS[rule_id]
        if status in semantics.supported_statuses:
            return TruthState.SUPPORTED
        if status in semantics.refuted_statuses:
            return TruthState.REFUTED
        return TruthState.INSUFFICIENT_EVIDENCE
    if rule_id == "medication_order_actionable.v3":
        status = record.get("status")
        semantics = MEDICATION_DECISION_RULE_SEMANTICS[rule_id]
        if status in semantics.supported_statuses:
            return TruthState.SUPPORTED
        if status in semantics.refuted_statuses:
            return TruthState.REFUTED
        return TruthState.INSUFFICIENT_EVIDENCE
    if rule_id == "medication_administered.v2":
        if record.get("resource_type") == "MedicationRequest":
            return TruthState.INSUFFICIENT_EVIDENCE
        status = record.get("status")
        semantics = MEDICATION_DECISION_RULE_SEMANTICS[rule_id]
        if status in semantics.supported_statuses:
            return TruthState.SUPPORTED
        if status in semantics.refuted_statuses:
            return TruthState.REFUTED
        return TruthState.INSUFFICIENT_EVIDENCE
    raise ProgramExecutionError(f"unknown evidence decision rule: {rule_id}")


def _concept_code(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    code = value.get("code")
    if isinstance(code, str):
        return code
    codings = value.get("codings")
    if isinstance(codings, list):
        for coding in codings:
            if isinstance(coding, dict) and isinstance(coding.get("code"), str):
                return coding["code"]
    return None


def analyze_complexity(
    program: TaskProgram,
    evidence_ids: frozenset[str],
    answer: Any,
) -> ProgramComplexity:
    tool_depth: dict[str, int] = {}
    root_tools: dict[str, frozenset[str]] = {}

    for node in program.nodes:
        refs = _node_refs(node)
        upstream_depth = max((tool_depth[ref.node_id] for ref in refs), default=0)
        upstream_roots = (
            frozenset().union(
                *(root_tools[ref.node_id] for ref in refs),
            )
            if refs
            else frozenset()
        )
        if isinstance(node, ToolNode):
            tool_depth[node.id] = upstream_depth + 1
            root_tools[node.id] = upstream_roots or frozenset({node.id})
        else:
            tool_depth[node.id] = upstream_depth
            root_tools[node.id] = upstream_roots

    output_id = program.output.node_id
    output_node = next(node for node in program.nodes if node.id == output_id)
    answer_width = (
        len(output_node.branches)
        if isinstance(output_node, TruthStateAggregateNode)
        else len(answer)
        if isinstance(answer, dict)
        else 1
    )
    return ProgramComplexity(
        dependency_depth=tool_depth[output_id],
        independent_width=len(root_tools[output_id]),
        evidence_width=len(evidence_ids),
        answer_width=answer_width,
    )


def _node_refs(node: ProgramNode) -> tuple[ValueRef, ...]:
    if isinstance(node, ToolNode):
        return tuple(_argument_refs(node.arguments))
    if isinstance(
        node,
        (
            SelectNode,
            ExistsNode,
            AggregateNode,
            ChartAbsenceNode,
            EvidenceStateNode,
        ),
    ):
        return (node.source,)
    if isinstance(node, VersionTrajectoryNode):
        return (
            (node.source, node.provenance_source)
            if node.provenance_source is not None
            else (node.source,)
        )
    if isinstance(node, CompareNode):
        refs = [node.left]
        if isinstance(node.right, ValueRef):
            refs.append(node.right)
        return tuple(refs)
    if isinstance(node, CalculateNode):
        return node.arguments
    if isinstance(node, BuildObjectNode):
        return tuple(node.fields.values())
    if isinstance(node, AllNode):
        return node.sources
    if isinstance(node, TruthStateAggregateNode):
        return tuple(node.branches.values())
    return ()


def node_references(node: ProgramNode) -> tuple[ValueRef, ...]:
    """Return the direct data dependencies of a program node."""

    return _node_refs(node)


def _argument_refs(value: Any) -> list[ValueRef]:
    refs: list[ValueRef] = []
    if isinstance(value, dict):
        if set(value) == {"$ref", "path"} and isinstance(value.get("$ref"), str):
            refs.append(ValueRef(node_id=value["$ref"], path=tuple(value["path"])))
        else:
            for child in value.values():
                refs.extend(_argument_refs(child))
    elif isinstance(value, list):
        for child in value:
            refs.extend(_argument_refs(child))
    return refs


def _argument_shape(value: Any, *, argument_name: str | None = None) -> Any:
    if isinstance(value, dict):
        if set(value) == {"$ref", "path"}:
            return {"$ref": value["$ref"], "path": value["path"]}
        return {
            key: _argument_shape(child, argument_name=key) for key, child in sorted(value.items())
        }
    if isinstance(value, list):
        return [_argument_shape(child, argument_name=argument_name) for child in value]
    if argument_name in {
        "direction",
        "document_kind",
        "record_kind",
        "relation",
        "resource_type",
        "system",
    }:
        return value
    if isinstance(value, bool):
        return "<boolean>"
    if isinstance(value, int):
        return "<integer>"
    if isinstance(value, float):
        return "<number>"
    if value is None:
        return "<null>"
    return "<string>"


def _canonicalize_references(value: Any, node_ids: dict[str, str]) -> Any:
    if isinstance(value, dict):
        normalized = {
            key: _canonicalize_references(child, node_ids) for key, child in value.items()
        }
        if isinstance(normalized.get("node_id"), str):
            normalized["node_id"] = node_ids[normalized["node_id"]]
        if isinstance(normalized.get("$ref"), str):
            normalized["$ref"] = node_ids[normalized["$ref"]]
        return normalized
    if isinstance(value, list):
        return [_canonicalize_references(child, node_ids) for child in value]
    return value


def _resolve_arguments(value: Any, values: dict[str, _RuntimeValue]) -> Any:
    if isinstance(value, dict):
        if set(value) == {"$ref", "path"} and isinstance(value.get("$ref"), str):
            return _resolve_ref(
                ValueRef(node_id=value["$ref"], path=tuple(value["path"])), values
            ).value
        return {key: _resolve_arguments(child, values) for key, child in value.items()}
    if isinstance(value, list):
        return [_resolve_arguments(child, values) for child in value]
    return value


def _resolve_ref(ref: ValueRef, values: dict[str, _RuntimeValue]) -> _RuntimeValue:
    source = values[ref.node_id]
    current = source.value
    selected_evidence_ids = frozenset(_collect_evidence_ids(current))
    for part in ref.path:
        try:
            current = current[part]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProgramExecutionError(
                f"cannot resolve {ref.node_id!r} at path {list(ref.path)!r}"
            ) from exc
        narrowed_evidence_ids = frozenset(_collect_evidence_ids(current))
        if narrowed_evidence_ids:
            selected_evidence_ids = narrowed_evidence_ids
    evidence_ids = (
        selected_evidence_ids | source.dependency_evidence_ids
        if selected_evidence_ids
        else source.evidence_ids
    )
    return _RuntimeValue(
        current,
        evidence_ids,
        source.dependency_evidence_ids,
    )


def _collect_evidence_ids(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        evidence_id = value.get("evidence_id")
        if isinstance(evidence_id, str):
            found.add(evidence_id)
        for child in value.values():
            found.update(_collect_evidence_ids(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.update(_collect_evidence_ids(child))
    return found


def _quantity_as(value: Any, target_unit: str) -> float:
    if not isinstance(value, dict):
        raise ProgramExecutionError("calculation requires typed quantity objects")
    number = value.get("value")
    unit = value.get("unit")
    if not isinstance(number, (int, float)) or isinstance(number, bool):
        raise ProgramExecutionError("quantity value must be numeric")
    conversions = {
        ("cm", "m"): 0.01,
        ("m", "m"): 1.0,
        ("kg", "kg"): 1.0,
        ("g", "kg"): 0.001,
        ("lb", "kg"): 0.45359237,
    }
    factor = conversions.get((unit, target_unit))
    if factor is None:
        raise ProgramExecutionError(f"cannot convert {unit!r} to {target_unit!r}")
    return float(number) * factor
