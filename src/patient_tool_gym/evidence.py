from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

SNAPSHOT_SCHEMA_VERSION = "patient-tool-gym.snapshot.v1"
HASH_VERSION = "patient-tool-gym.snapshot-hash.v1"
LEGACY_NORMALIZER_VERSION = "patient-tool-gym.legacy-jsonl.v1"
FHIR_NORMALIZER_VERSION = "patient-tool-gym.fhir-r4-selected.v6"
RELATIONSHIP_RULES_VERSION = "patient-tool-gym.fhir-reference-graph.r4.v1"
TERMINOLOGY_MAP_VERSION = "patient-tool-gym.source-codings.identity.v1"
LOGICAL_GROUP_RULES_VERSION = "patient-tool-gym.logical-group.fhir-equivalence.v3"
LEGACY_LOGICAL_GROUP_RULES_VERSION = "patient-tool-gym.logical-group.fhir-equivalence.v2"


class EvidenceFormatError(ValueError):
    """Raised when evidence does not satisfy the input contract."""


class SnapshotIntegrityError(EvidenceFormatError):
    """Raised when a serialized snapshot does not match its content hash."""


class UnknownEvidenceIdError(KeyError):
    """Raised when graph traversal starts from an unknown evidence ID."""


class _FrozenDict(dict[Any, Any]):
    """A JSON-compatible dictionary that rejects mutation.

    ``dict`` subclassing preserves the existing normalization and serialization
    contracts while preventing a caller from changing hash-bound snapshot state.
    ``deepcopy`` deliberately returns an ordinary mutable dictionary for patch and
    artifact-writing code.
    """

    def _immutable(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("snapshot values are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[Any, Any]:
        return {copy.deepcopy(key, memo): copy.deepcopy(value, memo) for key, value in self.items()}


class _FrozenList(list[Any]):
    """A JSON-compatible list that rejects mutation."""

    def _immutable(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("snapshot values are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable
    __iadd__ = _immutable
    __imul__ = _immutable

    def __deepcopy__(self, memo: dict[int, Any]) -> list[Any]:
        return [copy.deepcopy(value, memo) for value in self]


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return _FrozenDict({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return _FrozenList(_deep_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


class DuplicateEvidenceIdError(EvidenceFormatError):
    def __init__(self, evidence_id: str, first_line: int, duplicate_line: int) -> None:
        self.evidence_id = evidence_id
        self.line_numbers = (first_line, duplicate_line)
        super().__init__(
            f"Duplicate evidence_id {evidence_id!r} on lines {first_line} and {duplicate_line}"
        )


class PatientBoundaryError(EvidenceFormatError):
    def __init__(
        self,
        expected_patient_id: str,
        observed_patient_id: str,
        line_number: int,
    ) -> None:
        self.expected_patient_id = expected_patient_id
        self.observed_patient_id = observed_patient_id
        self.line_number = line_number
        super().__init__(
            "Evidence snapshot crosses patient boundary: "
            f"expected {expected_patient_id!r}, observed {observed_patient_id!r} "
            f"on line {line_number}"
        )


@dataclass(frozen=True, order=True)
class EvidenceLink:
    source_evidence_id: str
    relation: str
    target_evidence_id: str


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    patient_id: str
    resource_type: str
    resource_id: str
    logical_evidence_group: str
    category: str
    display: str
    code: dict[str, Any]
    references: dict[str, Any]
    text: str
    time: dict[str, Any]
    value: Any
    source_dataset: str
    payload: dict[str, Any]

    @classmethod
    def from_payload(cls, payload: dict[str, Any], line_number: int) -> EvidenceRecord:
        required = ("evidence_id", "patient_id", "resource_type")
        missing = [name for name in required if not isinstance(payload.get(name), str)]
        if missing:
            joined = ", ".join(missing)
            raise EvidenceFormatError(
                f"Line {line_number} is missing required string field(s): {joined}"
            )
        evidence_id = payload["evidence_id"]
        source = _mapping(payload.get("source"))
        resource_id = payload.get("resource_id") or source.get("resource_id")
        if not isinstance(resource_id, str) or not resource_id:
            resource_id = evidence_id.rsplit(":", 1)[-1]
        logical_group = payload.get("logical_evidence_group")
        if not isinstance(logical_group, str) or not logical_group:
            logical_group = f"record:{evidence_id}"
        source_dataset = payload.get("source_dataset")
        if not isinstance(source_dataset, str) or not source_dataset:
            source_dataset = "unknown"
        return cls(
            evidence_id=evidence_id,
            patient_id=payload["patient_id"],
            resource_type=payload["resource_type"],
            resource_id=resource_id,
            logical_evidence_group=logical_group,
            category=str(payload.get("category", "")),
            display=str(payload.get("display", "")),
            code=_deep_freeze(_mapping(payload.get("code"))),
            references=_deep_freeze(_mapping(payload.get("references"))),
            text=str(payload.get("text", "")),
            time=_deep_freeze(_mapping(payload.get("time"))),
            value=_deep_freeze(payload.get("value")),
            source_dataset=source_dataset,
            payload=_deep_freeze(copy.deepcopy(payload)),
        )

    @property
    def clinical_start(self) -> str:
        for field_name in (
            "start",
            "recorded",
            "issued",
            "authored",
            "onset",
            "onset_start",
        ):
            value = self.time.get(field_name)
            if isinstance(value, str):
                return value
        return ""

    @property
    def local_date(self) -> str:
        return self.clinical_start[:10]


@dataclass(frozen=True)
class PatientSnapshot:
    patient_id: str
    snapshot_id: str
    records: tuple[EvidenceRecord, ...]
    links: tuple[EvidenceLink, ...] = ()
    schema_version: str = SNAPSHOT_SCHEMA_VERSION
    hash_version: str = HASH_VERSION
    normalizer_version: str = LEGACY_NORMALIZER_VERSION
    relationship_rules_version: str = RELATIONSHIP_RULES_VERSION
    terminology_map_version: str = TERMINOLOGY_MAP_VERSION
    logical_group_rules_version: str = LOGICAL_GROUP_RULES_VERSION
    _by_id: Mapping[str, EvidenceRecord] = field(
        default_factory=lambda: MappingProxyType({}), repr=False, compare=False
    )
    _by_resource_type: Mapping[str, tuple[EvidenceRecord, ...]] = field(
        default_factory=lambda: MappingProxyType({}), repr=False, compare=False
    )
    _by_code: Mapping[str, tuple[EvidenceRecord, ...]] = field(
        default_factory=lambda: MappingProxyType({}), repr=False, compare=False
    )
    _outgoing: Mapping[str, tuple[EvidenceLink, ...]] = field(
        default_factory=lambda: MappingProxyType({}), repr=False, compare=False
    )
    _incoming: Mapping[str, tuple[EvidenceLink, ...]] = field(
        default_factory=lambda: MappingProxyType({}), repr=False, compare=False
    )

    def canonical_json(self) -> str:
        return _canonical_json(_semantic_snapshot_payload(self))

    def record_by_id(self, evidence_id: str) -> EvidenceRecord | None:
        return self._by_id.get(evidence_id)

    def records_by_resource_type(self, resource_type: str) -> tuple[EvidenceRecord, ...]:
        return self._by_resource_type.get(resource_type, ())

    def records_by_code(self, code: str) -> tuple[EvidenceRecord, ...]:
        return self._by_code.get(code, ())

    def outgoing_links(
        self, evidence_id: str, relation: str | None = None
    ) -> tuple[EvidenceLink, ...]:
        self._require_evidence_id(evidence_id)
        links = self._outgoing.get(evidence_id, ())
        return tuple(link for link in links if relation is None or link.relation == relation)

    def incoming_links(
        self, evidence_id: str, relation: str | None = None
    ) -> tuple[EvidenceLink, ...]:
        self._require_evidence_id(evidence_id)
        links = self._incoming.get(evidence_id, ())
        return tuple(link for link in links if relation is None or link.relation == relation)

    def related_records(
        self,
        evidence_id: str,
        *,
        relation: str | None = None,
        direction: str = "both",
    ) -> tuple[EvidenceRecord, ...]:
        if direction not in {"outgoing", "incoming", "both"}:
            raise ValueError("direction must be outgoing, incoming, or both")
        target_ids: set[str] = set()
        if direction in {"outgoing", "both"}:
            target_ids.update(
                link.target_evidence_id for link in self.outgoing_links(evidence_id, relation)
            )
        if direction in {"incoming", "both"}:
            target_ids.update(
                link.source_evidence_id for link in self.incoming_links(evidence_id, relation)
            )
        return tuple(self._by_id[target_id] for target_id in sorted(target_ids))

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "hash_version": self.hash_version,
            "normalizer_version": self.normalizer_version,
            "relationship_rules_version": self.relationship_rules_version,
            "terminology_map_version": self.terminology_map_version,
            "logical_group_rules_version": self.logical_group_rules_version,
            "patient_id": self.patient_id,
            "snapshot_id": self.snapshot_id,
            "records": [copy.deepcopy(record.payload) for record in self.records],
            "links": [
                {
                    "source_evidence_id": link.source_evidence_id,
                    "relation": link.relation,
                    "target_evidence_id": link.target_evidence_id,
                }
                for link in self.links
            ],
        }

    def _require_evidence_id(self, evidence_id: str) -> None:
        if evidence_id not in self._by_id:
            raise UnknownEvidenceIdError(evidence_id)


def parse_clinical_instant(value: str) -> datetime | None:
    """Parse a FHIR-style instant and normalize it to UTC for comparisons."""

    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def clinical_instant_sort_key(value: str) -> tuple[bool, datetime]:
    instant = parse_clinical_instant(value)
    return (instant is not None, instant or datetime.min.replace(tzinfo=UTC))


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def load_snapshot(path: Path) -> PatientSnapshot:
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as handle:
            serialized = json.load(handle)
    except json.JSONDecodeError:
        serialized = None

    if isinstance(serialized, dict) and isinstance(serialized.get("records"), list):
        snapshot = snapshot_from_payloads(
            serialized["records"],
            normalizer_version=str(serialized.get("normalizer_version", LEGACY_NORMALIZER_VERSION)),
            relationship_rules_version=str(
                serialized.get("relationship_rules_version", RELATIONSHIP_RULES_VERSION)
            ),
            terminology_map_version=str(
                serialized.get("terminology_map_version", TERMINOLOGY_MAP_VERSION)
            ),
            logical_group_rules_version=str(
                serialized.get("logical_group_rules_version", LOGICAL_GROUP_RULES_VERSION)
            ),
        )
        expected = serialized.get("snapshot_id")
        if isinstance(expected, str) and expected != snapshot.snapshot_id:
            raise SnapshotIntegrityError(
                f"Snapshot hash mismatch: expected {expected!r}, computed {snapshot.snapshot_id!r}"
            )
        return snapshot

    payloads: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvidenceFormatError(f"Line {line_number} is not valid JSON") from exc
            if not isinstance(payload, dict):
                raise EvidenceFormatError(f"Line {line_number} must contain a JSON object")
            payloads.append(payload)
    return snapshot_from_payloads(payloads)


def snapshot_from_payloads(
    payloads: Iterable[dict[str, Any]],
    *,
    normalizer_version: str = LEGACY_NORMALIZER_VERSION,
    relationship_rules_version: str = RELATIONSHIP_RULES_VERSION,
    terminology_map_version: str = TERMINOLOGY_MAP_VERSION,
    logical_group_rules_version: str = LOGICAL_GROUP_RULES_VERSION,
) -> PatientSnapshot:
    records: list[EvidenceRecord] = []
    evidence_lines: dict[str, int] = {}
    patient_id: str | None = None

    for line_number, payload in enumerate(payloads, start=1):
        if not isinstance(payload, dict):
            raise EvidenceFormatError(f"Line {line_number} must contain a JSON object")
        record = EvidenceRecord.from_payload(payload, line_number)
        first_line = evidence_lines.get(record.evidence_id)
        if first_line is not None:
            raise DuplicateEvidenceIdError(record.evidence_id, first_line, line_number)
        evidence_lines[record.evidence_id] = line_number
        if patient_id is None:
            patient_id = record.patient_id
        elif record.patient_id != patient_id:
            raise PatientBoundaryError(patient_id, record.patient_id, line_number)
        records.append(record)

    if patient_id is None:
        raise EvidenceFormatError("Evidence snapshot is empty")

    ordered = tuple(sorted(records, key=lambda record: record.evidence_id))
    links = _resolve_links(ordered)
    provisional = PatientSnapshot(
        patient_id=patient_id,
        snapshot_id="",
        records=ordered,
        links=links,
        normalizer_version=normalizer_version,
        relationship_rules_version=relationship_rules_version,
        terminology_map_version=terminology_map_version,
        logical_group_rules_version=logical_group_rules_version,
    )
    snapshot_id = _compute_snapshot_id(provisional)
    return _with_indexes(
        PatientSnapshot(
            patient_id=patient_id,
            snapshot_id=snapshot_id,
            records=ordered,
            links=links,
            normalizer_version=normalizer_version,
            relationship_rules_version=relationship_rules_version,
            terminology_map_version=terminology_map_version,
            logical_group_rules_version=logical_group_rules_version,
        )
    )


def dump_snapshot(snapshot: PatientSnapshot, path: Path) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(snapshot.to_payload(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _with_indexes(snapshot: PatientSnapshot) -> PatientSnapshot:
    by_id = {record.evidence_id: record for record in snapshot.records}
    by_type: dict[str, list[EvidenceRecord]] = defaultdict(list)
    by_code: dict[str, list[EvidenceRecord]] = defaultdict(list)
    for record in snapshot.records:
        by_type[record.resource_type].append(record)
        code = record.code.get("code")
        if isinstance(code, str):
            by_code[code].append(record)
        codings = record.code.get("codings")
        if isinstance(codings, list):
            for coding in codings:
                if isinstance(coding, dict) and isinstance(coding.get("code"), str):
                    by_code[coding["code"]].append(record)

    outgoing: dict[str, list[EvidenceLink]] = defaultdict(list)
    incoming: dict[str, list[EvidenceLink]] = defaultdict(list)
    for link in snapshot.links:
        outgoing[link.source_evidence_id].append(link)
        incoming[link.target_evidence_id].append(link)

    return PatientSnapshot(
        patient_id=snapshot.patient_id,
        snapshot_id=snapshot.snapshot_id,
        records=snapshot.records,
        links=snapshot.links,
        schema_version=snapshot.schema_version,
        hash_version=snapshot.hash_version,
        normalizer_version=snapshot.normalizer_version,
        relationship_rules_version=snapshot.relationship_rules_version,
        terminology_map_version=snapshot.terminology_map_version,
        logical_group_rules_version=snapshot.logical_group_rules_version,
        _by_id=MappingProxyType(by_id),
        _by_resource_type=MappingProxyType(
            {
                key: tuple(sorted(value, key=lambda record: record.evidence_id))
                for key, value in by_type.items()
            }
        ),
        _by_code=MappingProxyType(
            {
                key: tuple(
                    sorted(
                        {record.evidence_id: record for record in value}.values(),
                        key=lambda record: record.evidence_id,
                    )
                )
                for key, value in by_code.items()
            }
        ),
        _outgoing=MappingProxyType({key: tuple(sorted(value)) for key, value in outgoing.items()}),
        _incoming=MappingProxyType({key: tuple(sorted(value)) for key, value in incoming.items()}),
    )


def _resolve_links(records: tuple[EvidenceRecord, ...]) -> tuple[EvidenceLink, ...]:
    exact: dict[tuple[str, str], str] = {}
    ambiguous_exact: set[tuple[str, str]] = set()
    versioned: dict[tuple[str, str, str], str] = {}
    full_urls: dict[str, list[str]] = defaultdict(list)
    by_resource_id: dict[str, list[str]] = defaultdict(list)
    for record in records:
        identity = (record.resource_type, record.resource_id)
        if identity in ambiguous_exact:
            pass
        elif identity not in exact:
            exact[identity] = record.evidence_id
        else:
            # An unversioned reference to a history Bundle entry is ambiguous.
            exact.pop(identity, None)
            ambiguous_exact.add(identity)
        attributes = _mapping(record.payload.get("attributes"))
        meta = _mapping(attributes.get("resource_meta"))
        version_id = meta.get("versionId")
        if isinstance(version_id, str) and version_id:
            versioned[(record.resource_type, record.resource_id, version_id)] = record.evidence_id
        by_resource_id[record.resource_id].append(record.evidence_id)
        source = _mapping(record.payload.get("source"))
        full_url = source.get("full_url")
        if isinstance(full_url, str):
            full_urls[full_url].append(record.evidence_id)

    links: set[EvidenceLink] = set()
    for record in records:
        for relation, reference in _iter_references(record.references):
            target = _resolve_reference(reference, exact, versioned, full_urls, by_resource_id)
            if target is not None and target != record.evidence_id:
                links.add(
                    EvidenceLink(
                        source_evidence_id=record.evidence_id,
                        relation=relation,
                        target_evidence_id=target,
                    )
                )
    return tuple(sorted(links))


def _iter_references(
    references: dict[str, Any], relation: str | None = None
) -> Iterable[tuple[str, str]]:
    for key, value in references.items():
        active_relation = relation or key
        if isinstance(value, dict):
            raw = value.get("reference")
            if isinstance(raw, str):
                yield active_relation, raw
            else:
                yield from _iter_references(value, active_relation)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    raw = item.get("reference")
                    if isinstance(raw, str):
                        yield active_relation, raw
                    else:
                        yield from _iter_references(item, active_relation)


def _resolve_reference(
    reference: str,
    exact: dict[tuple[str, str], str],
    versioned: dict[tuple[str, str, str], str],
    full_urls: dict[str, list[str]],
    by_resource_id: dict[str, list[str]],
) -> str | None:
    full_url_matches = full_urls.get(reference, [])
    if len(full_url_matches) == 1:
        return full_url_matches[0]
    clean = reference.rstrip("/")
    if "?" in clean or clean.startswith("#"):
        return None
    version_id: str | None = None
    if "/_history/" in clean:
        clean, version_id = clean.rsplit("/_history/", 1)
        if not version_id:
            return None
    parts = clean.split("/")
    if len(parts) >= 2:
        identity = (parts[-2], parts[-1])
        target = (
            versioned.get((*identity, version_id))
            if version_id is not None
            else exact.get(identity)
        )
        if target is not None:
            return target
    if clean.startswith("urn:uuid:"):
        resource_id = clean.removeprefix("urn:uuid:")
        candidates = by_resource_id.get(resource_id, [])
        if len(candidates) == 1:
            return candidates[0]
    return None


def _compute_snapshot_id(snapshot: PatientSnapshot) -> str:
    body = _canonical_json(_semantic_snapshot_payload(snapshot)).encode("utf-8")
    digest = hashlib.sha256(snapshot.hash_version.encode("utf-8") + b"\0" + body).hexdigest()
    return f"sha256:{digest}"


def _semantic_snapshot_payload(snapshot: PatientSnapshot) -> dict[str, Any]:
    return {
        "schema_version": snapshot.schema_version,
        "hash_version": snapshot.hash_version,
        "normalizer_version": snapshot.normalizer_version,
        "relationship_rules_version": snapshot.relationship_rules_version,
        "terminology_map_version": snapshot.terminology_map_version,
        "logical_group_rules_version": snapshot.logical_group_rules_version,
        "patient_id": snapshot.patient_id,
        "records": [_semantic_record_payload(record.payload) for record in snapshot.records],
        "links": [
            {
                "source_evidence_id": link.source_evidence_id,
                "relation": link.relation,
                "target_evidence_id": link.target_evidence_id,
            }
            for link in snapshot.links
        ],
    }


def _semantic_record_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(payload))
    source = result.get("source")
    if isinstance(source, dict):
        source.pop("file", None)
        source.pop("path", None)
        source.pop("full_url", None)
    result.pop("provenance", None)
    return result


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
