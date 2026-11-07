from __future__ import annotations

import copy
import html
import re
import uuid
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from patient_tool_gym.program import PathPart, StrictModel, content_hash


class RawFHIRPatchError(ValueError):
    """Raised when a raw FHIR patch cannot be applied safely."""


class RawFHIRDistractorUnavailableError(RawFHIRPatchError):
    """Raised when the requested distractor tier has no admissible source record."""

    def __init__(
        self,
        *,
        requested_tier: str,
        target_resource_type: str,
        target_resource_id: str,
    ) -> None:
        self.requested_tier = requested_tier
        self.target_resource_type = target_resource_type
        self.target_resource_id = target_resource_id
        super().__init__(
            f"no distractor candidate for requested tier {requested_tier}: "
            f"{target_resource_type}/{target_resource_id}"
        )


class DanglingFHIRReference(StrictModel):
    source: str
    path: str
    reference: str


class ReferenceClosureReport(StrictModel):
    passed: bool
    resource_count: int
    reference_count: int
    external_conditional_reference_count: int = 0
    allowed_external_conditional_reference_types: tuple[str, ...] = ()
    dangling: tuple[DanglingFHIRReference, ...] = ()


class RawFHIRSet(StrictModel):
    op: Literal["set"] = "set"
    resource_type: str
    resource_id: str
    path: tuple[PathPart, ...] = Field(min_length=1)
    prior_present: bool
    prior_hash: str
    value: Any


class RawFHIRRemove(StrictModel):
    op: Literal["remove"] = "remove"
    resource_type: str
    resource_id: str
    path: tuple[PathPart, ...] = Field(min_length=1)
    prior_hash: str


class RawFHIRInsert(StrictModel):
    op: Literal["insert"] = "insert"
    full_url: str | None = None
    resource: dict[str, Any]


RawFHIRPatchOperation = Annotated[
    RawFHIRSet | RawFHIRRemove | RawFHIRInsert,
    Field(discriminator="op"),
]


class RawFHIRInsertedReferenceRoute(StrictModel):
    """Declare how a parent FHIR resource exposes newly inserted target records."""

    parent_resource_type: str = Field(min_length=1)
    parent_resource_id: str = Field(min_length=1)
    reference_field: str = Field(min_length=1)


class RawFHIRPatch(StrictModel):
    schema_version: Literal["patient-tool-gym.raw-fhir-patch.v1"] = (
        "patient-tool-gym.raw-fhir-patch.v1"
    )
    base_bundle_hash: str
    # A supported baseline is deliberately representable as an authenticated
    # no-op.  This lets the canonical supported/plain chart remain byte-for-byte
    # identical to the source Bundle instead of adding synthetic metadata.
    operations: tuple[RawFHIRPatchOperation, ...] = ()
    intervention_kind: Literal[
        "supported",
        "refuted",
        "insufficient_evidence",
        "conflicting_evidence",
        "answer_preserving_distractor",
    ]
    target_resource_type: str
    target_resource_id: str
    distractor_tier: (
        Literal[
            "cross_resource_routing",
            "same_resource_different_code",
            "same_resource_near_match",
        ]
        | None
    ) = None
    allowed_external_conditional_reference_types: tuple[str, ...] = ()
    patch_hash: str

    @model_validator(mode="after")
    def validate_noop_semantics(self) -> RawFHIRPatch:
        if not self.operations and self.intervention_kind != "supported":
            raise ValueError("only a supported intervention may be an empty no-op patch")
        return self


_VALUE_FIELDS = (
    "valueQuantity",
    "valueCodeableConcept",
    "valueString",
    "valueBoolean",
    "valueInteger",
    "valueDecimal",
    "valueRange",
    "valueRatio",
    "valueSampledData",
    "valueTime",
    "valueDateTime",
    "valuePeriod",
)

_PERFORMED_FIELDS = (
    "performedDateTime",
    "performedPeriod",
    "performedString",
    "performedAge",
    "performedRange",
    "performedTiming",
)

_CLINICAL_EVENT_IDENTIFIER_SYSTEM = "https://patient-tool-gym.dev/fhir/identifier/clinical-event"
_SOURCE_HASH_IDENTIFIER_SYSTEM = (
    "https://patient-tool-gym.dev/fhir/identifier/source-content-sha256"
)
_COMPILER_DEVICE_IDENTIFIER_SYSTEM = "https://patient-tool-gym.dev/fhir/identifier/software-agent"
_COMPILER_DEVICE_IDENTIFIER_VALUE = "patient-tool-gym-fhir-intervention-compiler-v2"

SYNTHEA_EXTERNAL_CONDITIONAL_REFERENCE_TYPES = frozenset(
    {"Location", "Organization", "Practitioner"}
)


def validate_reference_closure(
    bundle: Mapping[str, Any],
    *,
    allowed_external_conditional_reference_types: frozenset[str] = frozenset(),
) -> ReferenceClosureReport:
    entries = _entries(bundle)
    aliases: set[str] = set()
    full_urls: set[str] = set()
    resources: dict[tuple[str, str], dict[str, Any]] = {}
    for index, entry in enumerate(entries):
        resource = _entry_resource(entry, index)
        resource_type = resource.get("resourceType")
        resource_id = resource.get("id")
        if not isinstance(resource_type, str) or not isinstance(resource_id, str):
            raise RawFHIRPatchError(f"Bundle.entry[{index}] resource requires resourceType and id")
        identity = (resource_type, resource_id)
        if identity in resources:
            raise RawFHIRPatchError(
                f"duplicate FHIR resource identity: {resource_type}/{resource_id}"
            )
        resources[identity] = resource
        aliases.update({resource_id, f"{resource_type}/{resource_id}", f"urn:uuid:{resource_id}"})
        full_url = entry.get("fullUrl")
        if isinstance(full_url, str):
            if full_url in full_urls:
                raise RawFHIRPatchError(f"duplicate FHIR fullUrl: {full_url}")
            full_urls.add(full_url)
            aliases.add(full_url)

    dangling: list[DanglingFHIRReference] = []
    reference_count = 0
    external_conditional_reference_count = 0
    for resource_type, resource_id in sorted(resources):
        resource = resources[(resource_type, resource_id)]
        for path, reference in _iter_references(resource):
            reference_count += 1
            if (
                _conditional_reference_type(reference)
                in allowed_external_conditional_reference_types
            ):
                external_conditional_reference_count += 1
                continue
            if _reference_is_closed(reference, aliases):
                continue
            dangling.append(
                DanglingFHIRReference(
                    source=f"{resource_type}/{resource_id}",
                    path=".".join(str(part) for part in path),
                    reference=reference,
                )
            )
    return ReferenceClosureReport(
        passed=not dangling,
        resource_count=len(resources),
        reference_count=reference_count,
        external_conditional_reference_count=external_conditional_reference_count,
        allowed_external_conditional_reference_types=tuple(
            sorted(allowed_external_conditional_reference_types)
        ),
        dangling=tuple(dangling),
    )


def validate_reference_closure_strict(
    bundle: Mapping[str, Any],
    *,
    allowed_external_conditional_reference_types: frozenset[str] = frozenset(),
) -> ReferenceClosureReport:
    """Validate exact Bundle-local reference targets, including contained resources.

    This stricter, versioned release gate intentionally lives beside the legacy
    validator so historical manifests remain reproducible. It rejects ambiguous
    bare IDs, checks ``#contained`` references in their owning resource, follows
    references inside contained resources, and requires an exact history target.
    """

    entries = _entries(bundle)
    resources: dict[tuple[str, str], dict[str, Any]] = {}
    full_urls: set[str] = set()
    typed_aliases: set[str] = set()
    history_aliases: set[str] = set()
    contained_by_owner: dict[tuple[str, str], set[str]] = {}

    transactional = bundle.get("type") in {"batch", "transaction"}
    for index, entry in enumerate(entries):
        resource = _entry_resource(entry, index)
        resource_type = resource.get("resourceType")
        resource_id = resource.get("id")
        if not isinstance(resource_type, str) or not isinstance(resource_id, str):
            raise RawFHIRPatchError(f"Bundle.entry[{index}] resource requires resourceType and id")
        identity = (resource_type, resource_id)
        if identity in resources:
            raise RawFHIRPatchError(
                f"duplicate FHIR resource identity: {resource_type}/{resource_id}"
            )
        resources[identity] = resource
        full_url = entry.get("fullUrl")
        if isinstance(full_url, str):
            if full_url in full_urls:
                raise RawFHIRPatchError(f"duplicate FHIR fullUrl: {full_url}")
            if transactional and full_url.startswith("urn:uuid:"):
                try:
                    uuid.UUID(full_url.removeprefix("urn:uuid:"))
                except ValueError as exc:
                    raise RawFHIRPatchError(
                        f"Bundle.entry[{index}] has an invalid urn:uuid fullUrl"
                    ) from exc
            full_urls.add(full_url)
        if transactional:
            if _transaction_entry_has_resolvable_typed_identity(
                entry,
                index=index,
                resource_type=resource_type,
                resource_id=resource_id,
            ):
                typed_aliases.add(f"{resource_type}/{resource_id}")
        else:
            typed_aliases.add(f"{resource_type}/{resource_id}")
        meta = resource.get("meta")
        version_id = meta.get("versionId") if isinstance(meta, Mapping) else None
        if isinstance(version_id, str) and version_id and not transactional:
            history_aliases.add(f"{resource_type}/{resource_id}/_history/{version_id}")
            if isinstance(full_url, str) and full_url.endswith(f"/{resource_type}/{resource_id}"):
                history_aliases.add(f"{full_url}/_history/{version_id}")

        contained = resource.get("contained", [])
        if not isinstance(contained, list):
            raise RawFHIRPatchError(
                f"FHIR resource {resource_type}/{resource_id} has non-list contained"
            )
        contained_ids: set[str] = set()
        for contained_index, child in enumerate(contained):
            if not isinstance(child, Mapping):
                raise RawFHIRPatchError(
                    f"FHIR resource {resource_type}/{resource_id} contained[{contained_index}] "
                    "is not an object"
                )
            child_id = child.get("id")
            child_type = child.get("resourceType")
            if not isinstance(child_id, str) or not child_id or not isinstance(child_type, str):
                raise RawFHIRPatchError(
                    f"FHIR resource {resource_type}/{resource_id} contained[{contained_index}] "
                    "requires resourceType and id"
                )
            if child_id in contained_ids:
                raise RawFHIRPatchError(
                    f"duplicate contained id #{child_id} in {resource_type}/{resource_id}"
                )
            contained_ids.add(child_id)
        contained_by_owner[identity] = contained_ids

    dangling: list[DanglingFHIRReference] = []
    reference_count = 0
    external_conditional_reference_count = 0
    for identity in sorted(resources):
        resource = resources[identity]
        for path, reference in _iter_references_strict(resource):
            reference_count += 1
            conditional_type = _conditional_reference_type(reference)
            if conditional_type in allowed_external_conditional_reference_types:
                external_conditional_reference_count += 1
                continue
            if _strict_reference_is_closed(
                reference,
                owner_contained_ids=contained_by_owner[identity],
                typed_aliases=typed_aliases,
                full_urls=full_urls,
                history_aliases=history_aliases,
            ):
                continue
            dangling.append(
                DanglingFHIRReference(
                    source=f"{identity[0]}/{identity[1]}",
                    path=".".join(str(part) for part in path),
                    reference=reference,
                )
            )
    return ReferenceClosureReport(
        passed=not dangling,
        resource_count=len(resources),
        reference_count=reference_count,
        external_conditional_reference_count=external_conditional_reference_count,
        allowed_external_conditional_reference_types=tuple(
            sorted(allowed_external_conditional_reference_types)
        ),
        dangling=tuple(dangling),
    )


def apply_raw_fhir_patch(bundle: Mapping[str, Any], patch: RawFHIRPatch) -> dict[str, Any]:
    base = copy.deepcopy(dict(bundle))
    if not patch.operations and patch.intervention_kind != "supported":
        raise RawFHIRPatchError("only a supported intervention may replay as a no-op")
    if content_hash(base) != patch.base_bundle_hash:
        raise RawFHIRPatchError("raw FHIR patch base bundle hash mismatch")
    allowed_conditionals = frozenset(patch.allowed_external_conditional_reference_types)
    base_closure = validate_reference_closure_strict(
        base,
        allowed_external_conditional_reference_types=allowed_conditionals,
    )
    if not base_closure.passed:
        raise RawFHIRPatchError("base FHIR Bundle does not have closed local references")
    expected_hash = _patch_hash(patch, include_hash=False)
    if patch.patch_hash != expected_hash:
        raise RawFHIRPatchError("raw FHIR patch hash mismatch")

    entries = _entries(base)
    for operation in patch.operations:
        if isinstance(operation, RawFHIRInsert):
            resource = copy.deepcopy(operation.resource)
            resource_type = resource.get("resourceType")
            resource_id = resource.get("id")
            if not isinstance(resource_type, str) or not isinstance(resource_id, str):
                raise RawFHIRPatchError("inserted resource requires resourceType and id")
            if _find_resource(base, resource_type, resource_id) is not None:
                raise RawFHIRPatchError(
                    f"insert target already exists: {resource_type}/{resource_id}"
                )
            entry: dict[str, Any] = {"resource": resource}
            if operation.full_url is not None:
                entry["fullUrl"] = operation.full_url
            if base.get("type") in {"batch", "transaction"}:
                if operation.full_url is None:
                    raise RawFHIRPatchError(
                        "transaction inserts require a fullUrl for resolvable references"
                    )
                entry["request"] = {
                    "method": "PUT",
                    "url": f"{resource_type}/{resource_id}",
                }
            entries.append(entry)
            continue

        resource = _find_resource(base, operation.resource_type, operation.resource_id)
        if resource is None:
            raise RawFHIRPatchError(
                f"patch target missing: {operation.resource_type}/{operation.resource_id}"
            )
        present, prior = _path_value(resource, operation.path)
        if isinstance(operation, RawFHIRSet):
            if present != operation.prior_present or content_hash(prior) != operation.prior_hash:
                raise RawFHIRPatchError("raw FHIR set prior value mismatch")
            _set_path(resource, operation.path, copy.deepcopy(operation.value))
        else:
            if not present or content_hash(prior) != operation.prior_hash:
                raise RawFHIRPatchError("raw FHIR remove prior value mismatch")
            _remove_path(resource, operation.path)

    result_closure = validate_reference_closure_strict(
        base,
        allowed_external_conditional_reference_types=allowed_conditionals,
    )
    if not result_closure.passed:
        raise RawFHIRPatchError("patched FHIR Bundle introduced dangling local references")
    return base


def build_truth_state_patch(
    bundle: Mapping[str, Any],
    *,
    resource_type: str,
    resource_id: str,
    target_state: Literal[
        "supported",
        "refuted",
        "insufficient_evidence",
        "conflicting_evidence",
        "answer_preserving_distractor",
    ],
    distractor_tier: Literal[
        "cross_resource_routing",
        "same_resource_different_code",
        "same_resource_near_match",
    ] = "same_resource_near_match",
    provenance_recorded_at: str | None = None,
    inserted_reference_route: RawFHIRInsertedReferenceRoute | None = None,
) -> RawFHIRPatch:
    base = copy.deepcopy(dict(bundle))
    allowed_conditionals = frozenset(
        reference_type
        for _, reference in _iter_references_strict(base)
        if (reference_type := _conditional_reference_type(reference))
        in SYNTHEA_EXTERNAL_CONDITIONAL_REFERENCE_TYPES
    )
    if not validate_reference_closure_strict(
        base,
        allowed_external_conditional_reference_types=allowed_conditionals,
    ).passed:
        raise RawFHIRPatchError("cannot compile a patch from a Bundle with dangling references")
    resource = _find_resource(base, resource_type, resource_id)
    if resource is None:
        raise RawFHIRPatchError(f"FHIR resource not found: {resource_type}/{resource_id}")

    _validate_intervention_target(resource, target_state=target_state)
    if inserted_reference_route is not None:
        _validate_inserted_reference_route(
            base,
            inserted_reference_route,
            target_resource_type=resource_type,
            target_resource_id=resource_id,
        )
    desired = (
        "supported"
        if target_state in {"conflicting_evidence", "answer_preserving_distractor"}
        else target_state
    )
    operations: list[RawFHIRPatchOperation]
    update_sources: dict[tuple[str, str], dict[str, Any]] = {}
    create_sources: list[tuple[str, dict[str, Any]]] = []
    if target_state == "conflicting_evidence":
        supported = copy.deepcopy(resource)
        _mutate_resource_in_place(supported, "supported")
        _ensure_shared_event_identifier(
            supported,
            _shared_event_identifier(base, resource_type=resource_type, resource_id=resource_id),
        )
        operations = list(_resource_diff_operations(resource, supported))
        if operations:
            update_sources[(resource_type, resource_id)] = copy.deepcopy(resource)
        clone = copy.deepcopy(supported)
        clone_id = _opaque_generated_resource_id(
            base,
            resource_type=resource_type,
            source_resource_id=resource_id,
            derivation="paired-assertion",
        )
        clone["id"] = clone_id
        _mutate_resource_in_place(clone, "refuted")
        _sanitize_derived_resource_metadata(clone)
        _harden_conflict_clone(clone)
        _set_generated_narrative(clone)
        clone_full_url = _full_url(resource_type, clone_id)
        operations.append(
            RawFHIRInsert(
                full_url=clone_full_url,
                resource=clone,
            )
        )
        route_operation, route_source = _inserted_reference_operation(
            base,
            inserted_reference_route,
            inserted_full_url=clone_full_url,
        )
        if route_operation is not None and route_source is not None:
            operations.append(route_operation)
            update_sources[
                (
                    inserted_reference_route.parent_resource_type,
                    inserted_reference_route.parent_resource_id,
                )
            ] = route_source
        create_sources.append((_bundle_reference(base, resource_type, resource_id), resource))
    else:
        operations = list(_state_operations(resource, desired))
        if operations:
            update_sources[(resource_type, resource_id)] = copy.deepcopy(resource)
    actual_distractor_tier = None
    if target_state == "answer_preserving_distractor":
        distractor, actual_distractor_tier = _distractor_source(
            base,
            target=resource,
            requested_tier=distractor_tier,
        )
        clone = _materialize_distinct_distractor(
            base,
            source=distractor,
            target=resource,
        )
        distractor_type = str(clone["resourceType"])
        distractor_id = str(clone["id"])
        distractor_full_url = _full_url(distractor_type, distractor_id)
        operations.append(
            RawFHIRInsert(
                full_url=distractor_full_url,
                resource=clone,
            )
        )
        create_sources.append(
            (
                _bundle_reference(
                    base,
                    str(distractor["resourceType"]),
                    str(distractor["id"]),
                ),
                distractor,
            )
        )

    operations = _with_provenance(
        base,
        operations,
        update_sources=update_sources,
        create_sources=create_sources,
        recorded_at=provenance_recorded_at,
    )
    draft = RawFHIRPatch(
        base_bundle_hash=content_hash(base),
        operations=tuple(operations),
        intervention_kind=target_state,
        target_resource_type=resource_type,
        target_resource_id=resource_id,
        distractor_tier=actual_distractor_tier,
        allowed_external_conditional_reference_types=tuple(sorted(allowed_conditionals)),
        patch_hash="pending",
    )
    return draft.model_copy(update={"patch_hash": _patch_hash(draft, include_hash=False)})


def build_distractor_patch(
    bundle: Mapping[str, Any],
    *,
    resource_type: str,
    resource_id: str,
    distractor_tier: Literal[
        "cross_resource_routing",
        "same_resource_different_code",
        "same_resource_near_match",
    ] = "same_resource_near_match",
    forbidden_codes: frozenset[tuple[str, str]] = frozenset(),
    target_date: str | None = None,
    require_exact_tier: bool = False,
    provenance_recorded_at: str | None = None,
    inserted_reference_route: RawFHIRInsertedReferenceRoute | None = None,
) -> RawFHIRPatch:
    """Insert an irrelevant clinical record without changing the target's state."""

    base = copy.deepcopy(dict(bundle))
    allowed_conditionals = frozenset(
        reference_type
        for _, reference in _iter_references_strict(base)
        if (reference_type := _conditional_reference_type(reference))
        in SYNTHEA_EXTERNAL_CONDITIONAL_REFERENCE_TYPES
    )
    if not validate_reference_closure_strict(
        base,
        allowed_external_conditional_reference_types=allowed_conditionals,
    ).passed:
        raise RawFHIRPatchError(
            "cannot compile a distractor patch from a Bundle with unresolved references"
        )
    target = _find_resource(base, resource_type, resource_id)
    if target is None:
        raise RawFHIRPatchError(f"FHIR resource not found: {resource_type}/{resource_id}")
    if inserted_reference_route is not None:
        _validate_inserted_reference_route(
            base,
            inserted_reference_route,
            target_resource_type=resource_type,
            target_resource_id=resource_id,
        )
    distractor, actual_tier = _distractor_source(
        base,
        target=target,
        requested_tier=distractor_tier,
        forbidden_codes=forbidden_codes,
        required_system=(
            target_code[0] if (target_code := _primary_code(target)) is not None else None
        ),
        required_encounter_reference=_encounter_reference(target),
        allow_tier_fallback=not require_exact_tier,
    )
    clone = _materialize_distinct_distractor(
        base,
        source=distractor,
        target=target,
        target_date=target_date,
    )
    distractor_type = str(clone["resourceType"])
    distractor_id = str(clone["id"])
    distractor_full_url = _full_url(distractor_type, distractor_id)
    operations: list[RawFHIRPatchOperation] = [
        RawFHIRInsert(full_url=distractor_full_url, resource=clone)
    ]
    update_sources: dict[tuple[str, str], dict[str, Any]] = {}
    route_operation, route_source = _inserted_reference_operation(
        base,
        inserted_reference_route,
        inserted_full_url=distractor_full_url,
    )
    if route_operation is not None and route_source is not None:
        operations.append(route_operation)
        update_sources[
            (
                inserted_reference_route.parent_resource_type,
                inserted_reference_route.parent_resource_id,
            )
        ] = route_source
    operations = _with_provenance(
        base,
        operations,
        update_sources=update_sources,
        create_sources=[
            (
                _bundle_reference(
                    base,
                    str(distractor["resourceType"]),
                    str(distractor["id"]),
                ),
                distractor,
            )
        ],
        recorded_at=provenance_recorded_at,
    )
    draft = RawFHIRPatch(
        base_bundle_hash=content_hash(base),
        operations=tuple(operations),
        intervention_kind="answer_preserving_distractor",
        target_resource_type=resource_type,
        target_resource_id=resource_id,
        distractor_tier=actual_tier,
        allowed_external_conditional_reference_types=tuple(sorted(allowed_conditionals)),
        patch_hash="pending",
    )
    return draft.model_copy(update={"patch_hash": _patch_hash(draft, include_hash=False)})


def _validate_inserted_reference_route(
    bundle: Mapping[str, Any],
    route: RawFHIRInsertedReferenceRoute,
    *,
    target_resource_type: str,
    target_resource_id: str,
) -> None:
    parent = _find_resource(
        bundle,
        route.parent_resource_type,
        route.parent_resource_id,
    )
    if parent is None:
        raise RawFHIRPatchError(
            "inserted-reference route parent does not exist: "
            f"{route.parent_resource_type}/{route.parent_resource_id}"
        )
    references = parent.get(route.reference_field)
    if not isinstance(references, list):
        raise RawFHIRPatchError(
            "inserted-reference route requires an existing top-level reference list"
        )
    target_reference = _bundle_reference(
        bundle,
        target_resource_type,
        target_resource_id,
    )
    if not any(
        isinstance(item, Mapping) and item.get("reference") == target_reference
        for item in references
    ):
        raise RawFHIRPatchError(
            "inserted-reference route parent does not already reference the intervention target"
        )


def _inserted_reference_operation(
    bundle: Mapping[str, Any],
    route: RawFHIRInsertedReferenceRoute | None,
    *,
    inserted_full_url: str,
) -> tuple[RawFHIRSet | None, dict[str, Any] | None]:
    if route is None:
        return None, None
    parent = _find_resource(
        bundle,
        route.parent_resource_type,
        route.parent_resource_id,
    )
    if parent is None:  # defensive after route validation
        raise RawFHIRPatchError("inserted-reference route parent disappeared")
    prior = parent.get(route.reference_field)
    if not isinstance(prior, list):
        raise RawFHIRPatchError("inserted-reference route field is not a list")
    if any(
        isinstance(item, Mapping) and item.get("reference") == inserted_full_url for item in prior
    ):
        raise RawFHIRPatchError("inserted-reference route would duplicate a reference")
    value = [*copy.deepcopy(prior), {"reference": inserted_full_url}]
    return (
        RawFHIRSet(
            resource_type=route.parent_resource_type,
            resource_id=route.parent_resource_id,
            path=(route.reference_field,),
            prior_present=True,
            prior_hash=content_hash(prior),
            value=value,
        ),
        copy.deepcopy(parent),
    )


def _state_operations(
    resource: dict[str, Any],
    target_state: Literal["supported", "refuted", "insufficient_evidence"],
) -> tuple[RawFHIRPatchOperation, ...]:
    mutated = copy.deepcopy(resource)
    _mutate_resource_in_place(mutated, target_state)
    return _resource_diff_operations(resource, mutated)


def _resource_diff_operations(
    resource: dict[str, Any],
    mutated: dict[str, Any],
) -> tuple[RawFHIRPatchOperation, ...]:
    """Compile deterministic top-level changes with prior-value hash guards."""

    operations: list[RawFHIRPatchOperation] = []
    keys = sorted(set(resource) | set(mutated))
    for key in keys:
        if key in {"resourceType", "id"}:
            continue
        before_present = key in resource
        after_present = key in mutated
        if before_present and not after_present:
            operations.append(
                RawFHIRRemove(
                    resource_type=resource["resourceType"],
                    resource_id=resource["id"],
                    path=(key,),
                    prior_hash=content_hash(resource[key]),
                )
            )
        elif after_present and (not before_present or resource.get(key) != mutated.get(key)):
            operations.append(
                RawFHIRSet(
                    resource_type=resource["resourceType"],
                    resource_id=resource["id"],
                    path=(key,),
                    prior_present=before_present,
                    prior_hash=content_hash(resource.get(key)),
                    value=mutated[key],
                )
            )
    return tuple(operations)


def _validate_intervention_target(
    resource: Mapping[str, Any],
    *,
    target_state: str,
) -> None:
    resource_type = resource.get("resourceType")
    if resource_type not in {
        "Condition",
        "Observation",
        "Procedure",
        "Immunization",
        "MedicationAdministration",
        "MedicationRequest",
    }:
        raise RawFHIRPatchError(f"unsupported intervention resource: {resource_type}")
    if _primary_code(resource) is None:
        raise RawFHIRPatchError(
            f"{resource_type} intervention target requires a coded clinical concept"
        )
    owner_field = "patient" if resource_type == "Immunization" else "subject"
    owner = resource.get(owner_field)
    if not isinstance(owner, Mapping) or not isinstance(owner.get("reference"), str):
        raise RawFHIRPatchError(
            f"{resource_type} intervention target requires {owner_field}.reference"
        )

    if resource_type == "Observation":
        components = resource.get("component")
        if isinstance(components, list) and components:
            raise RawFHIRPatchError(
                "Observation interventions reject component panels; select a scalar "
                "top-level Observation or implement component-scoped mutation"
            )
        if target_state in {"supported", "conflicting_evidence"} and not any(
            key in resource for key in _VALUE_FIELDS
        ):
            raise RawFHIRPatchError(
                "supported/conflicting Observation intervention requires top-level value[x]"
            )
    if resource_type == "Procedure":
        performed = [field for field in _PERFORMED_FIELDS if field in resource]
        if len(performed) > 1:
            raise RawFHIRPatchError("Procedure contains multiple performed[x] choices")
        if target_state in {"supported", "conflicting_evidence"} and not performed:
            raise RawFHIRPatchError(
                "supported/conflicting Procedure intervention requires performed[x]"
            )
    if target_state == "conflicting_evidence":
        _validate_shared_assertion_identity(resource)


def _mutate_resource_in_place(
    resource: dict[str, Any],
    target_state: Literal["supported", "refuted", "insufficient_evidence"],
) -> None:
    resource_type = resource.get("resourceType")
    if resource_type == "Condition":
        if target_state == "supported":
            verification_system = "http://terminology.hl7.org/CodeSystem/condition-ver-status"
            if not _concept_contains_code(
                resource.get("verificationStatus"),
                system=verification_system,
                code="confirmed",
            ):
                resource["verificationStatus"] = _concept(
                    verification_system,
                    "confirmed",
                )
        elif target_state == "refuted":
            resource["verificationStatus"] = _concept(
                "http://terminology.hl7.org/CodeSystem/condition-ver-status", "refuted"
            )
            resource["clinicalStatus"] = _concept(
                "http://terminology.hl7.org/CodeSystem/condition-clinical", "inactive"
            )
            for key in tuple(resource):
                if key.startswith("abatement"):
                    resource.pop(key)
        else:
            resource["verificationStatus"] = _concept(
                "http://terminology.hl7.org/CodeSystem/condition-ver-status", "unconfirmed"
            )
        return

    if resource_type == "Observation":
        if target_state == "supported":
            if not any(key in resource for key in _VALUE_FIELDS):
                raise RawFHIRPatchError("a supported Observation requires an existing value[x]")
            resource["status"] = "final"
            resource.pop("dataAbsentReason", None)
        elif target_state == "refuted":
            resource["status"] = "cancelled"
            for key in _VALUE_FIELDS:
                resource.pop(key, None)
            resource["dataAbsentReason"] = _concept(
                "http://terminology.hl7.org/CodeSystem/data-absent-reason", "not-performed"
            )
        else:
            resource["status"] = "final"
            for key in _VALUE_FIELDS:
                resource.pop(key, None)
            resource["dataAbsentReason"] = _concept(
                "http://terminology.hl7.org/CodeSystem/data-absent-reason", "unknown"
            )
        return

    status_map: dict[str, dict[str, str]] = {
        "Procedure": {
            "supported": "completed",
            "refuted": "not-done",
            "insufficient_evidence": "entered-in-error",
        },
        "Immunization": {
            "supported": "completed",
            "refuted": "not-done",
            "insufficient_evidence": "entered-in-error",
        },
        "MedicationAdministration": {
            "supported": "completed",
            "refuted": "not-done",
            "insufficient_evidence": "entered-in-error",
        },
        "MedicationRequest": {
            "supported": "active",
            "refuted": "cancelled",
            "insufficient_evidence": "draft",
        },
    }
    if resource_type not in status_map:
        raise RawFHIRPatchError(f"unsupported intervention resource: {resource_type}")
    resource["status"] = status_map[resource_type][target_state]
    if resource_type == "Procedure" and target_state != "supported":
        for field in _PERFORMED_FIELDS:
            resource.pop(field, None)


def _with_provenance(
    bundle: Mapping[str, Any],
    operations: list[RawFHIRPatchOperation],
    *,
    update_sources: Mapping[tuple[str, str], dict[str, Any]],
    create_sources: list[tuple[str, dict[str, Any]]],
    recorded_at: str | None,
) -> list[RawFHIRPatchOperation]:
    """Add operation-accurate, source-bound provenance for actual chart changes."""

    update_identities = sorted(
        {
            (operation.resource_type, operation.resource_id)
            for operation in operations
            if isinstance(operation, (RawFHIRSet, RawFHIRRemove))
        }
    )
    created = [operation for operation in operations if isinstance(operation, RawFHIRInsert)]
    if len(created) != len(create_sources):
        raise RawFHIRPatchError("every derived resource requires one source hash binding")
    if not update_identities and not created:
        return operations

    recorded = _intervention_recorded(bundle, explicit=recorded_at)
    device_reference, device_insert = _compiler_device_reference(bundle)
    result = list(operations)
    if device_insert is not None:
        result.append(device_insert)

    if update_identities:
        targets: list[str] = []
        sources: list[tuple[str, dict[str, Any]]] = []
        for identity in update_identities:
            source = update_sources.get(identity)
            if source is None:
                raise RawFHIRPatchError("updated resource lacks its prior source binding")
            reference = _bundle_reference(bundle, *identity)
            targets.append(reference)
            sources.append((reference, source))
        provenance = _provenance_resource(
            operation="UPDATE",
            targets=targets,
            sources=sources,
            device_reference=device_reference,
            recorded=recorded,
        )
        result.append(
            RawFHIRInsert(
                full_url=_full_url("Provenance", str(provenance["id"])),
                resource=provenance,
            )
        )

    if created:
        targets = []
        for operation in created:
            if operation.full_url is None:
                raise RawFHIRPatchError("derived resources require a transaction fullUrl")
            targets.append(operation.full_url)
        provenance = _provenance_resource(
            operation="CREATE",
            targets=targets,
            sources=create_sources,
            device_reference=device_reference,
            recorded=recorded,
        )
        result.append(
            RawFHIRInsert(
                full_url=_full_url("Provenance", str(provenance["id"])),
                resource=provenance,
            )
        )
    return result


def _provenance_resource(
    *,
    operation: Literal["CREATE", "UPDATE"],
    targets: list[str],
    sources: list[tuple[str, dict[str, Any]]],
    device_reference: str,
    recorded: str,
) -> dict[str, Any]:
    source_hashes = [content_hash(source) for _, source in sources]
    resource_id = _opaque_id(
        "Provenance",
        {
            "operation": operation,
            "targets": targets,
            "sources": list(
                zip(
                    (reference for reference, _ in sources),
                    source_hashes,
                    strict=True,
                )
            ),
            "recorded": recorded,
        },
    )
    return {
        "resourceType": "Provenance",
        "id": resource_id,
        "text": {
            "status": "generated",
            "div": (
                '<div xmlns="http://www.w3.org/1999/xhtml">'
                "<p>FHIR dataset transformation provenance.</p></div>"
            ),
        },
        "target": [{"reference": target} for target in targets],
        "recorded": recorded,
        "activity": {
            "coding": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/v3-DataOperation",
                    "code": operation,
                }
            ],
            "text": f"FHIR resource {operation.casefold()}",
        },
        "agent": [
            {
                "type": _concept(
                    "http://terminology.hl7.org/CodeSystem/provenance-participant-type",
                    "assembler",
                ),
                "who": {"reference": device_reference},
            }
        ],
        "entity": [
            {
                "role": "source",
                "what": {
                    "reference": reference,
                    "identifier": {
                        "system": _SOURCE_HASH_IDENTIFIER_SYSTEM,
                        "value": source_hash,
                    },
                },
            }
            for (reference, _), source_hash in zip(sources, source_hashes, strict=True)
        ],
    }


def _compiler_device_reference(
    bundle: Mapping[str, Any],
) -> tuple[str, RawFHIRInsert | None]:
    for entry in _entries(bundle):
        resource = entry.get("resource")
        if not isinstance(resource, Mapping) or resource.get("resourceType") != "Device":
            continue
        identifiers = resource.get("identifier")
        if not isinstance(identifiers, list):
            continue
        if any(
            isinstance(identifier, Mapping)
            and identifier.get("system") == _COMPILER_DEVICE_IDENTIFIER_SYSTEM
            and identifier.get("value") == _COMPILER_DEVICE_IDENTIFIER_VALUE
            for identifier in identifiers
        ):
            resource_id = resource.get("id")
            if not isinstance(resource_id, str):
                raise RawFHIRPatchError("compiler Device is missing id")
            return _bundle_reference(bundle, "Device", resource_id), None

    resource_id = _opaque_id("Device", _COMPILER_DEVICE_IDENTIFIER_VALUE)
    full_url = _full_url("Device", resource_id)
    device = {
        "resourceType": "Device",
        "id": resource_id,
        "identifier": [
            {
                "system": _COMPILER_DEVICE_IDENTIFIER_SYSTEM,
                "value": _COMPILER_DEVICE_IDENTIFIER_VALUE,
            }
        ],
        "status": "active",
        "deviceName": [
            {
                "name": "Patient Tool Gym FHIR Intervention Compiler",
                "type": "model-name",
            }
        ],
    }
    return full_url, RawFHIRInsert(full_url=full_url, resource=device)


def _intervention_recorded(
    bundle: Mapping[str, Any],
    *,
    explicit: str | None,
) -> str:
    """Use an explicit instant no earlier than source ``Bundle.timestamp``."""

    bundle_timestamp = bundle.get("timestamp")
    if explicit is None:
        if not isinstance(bundle_timestamp, str):
            raise RawFHIRPatchError(
                "intervention provenance requires Bundle.timestamp or provenance_recorded_at"
            )
        explicit = bundle_timestamp
    parsed = _parse_fhir_instant(explicit, field="provenance_recorded_at")
    if isinstance(bundle_timestamp, str):
        source_assembled_at = _parse_fhir_instant(bundle_timestamp, field="Bundle.timestamp")
        if parsed < source_assembled_at:
            raise RawFHIRPatchError(
                "provenance_recorded_at cannot be earlier than Bundle.timestamp"
            )
    return parsed.isoformat().replace("+00:00", "Z")


def _parse_fhir_instant(value: str, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RawFHIRPatchError(f"{field} must be an ISO 8601 instant") from exc
    if parsed.tzinfo is None:
        raise RawFHIRPatchError(f"{field} must include a timezone")
    return parsed.astimezone(UTC)


def _concept(system: str, code: str) -> dict[str, Any]:
    return {"coding": [{"system": system, "code": code}]}


def _concept_contains_code(value: Any, *, system: str, code: str) -> bool:
    if not isinstance(value, Mapping):
        return False
    codings = value.get("coding")
    return isinstance(codings, list) and any(
        isinstance(coding, Mapping)
        and coding.get("system") == system
        and coding.get("code") == code
        for coding in codings
    )


def _full_url(resource_type: str, resource_id: str) -> str:
    deterministic = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"https://patient-tool-gym.dev/fhir/{resource_type}/{resource_id}",
    )
    return f"urn:uuid:{deterministic}"


def _bundle_reference(bundle: Mapping[str, Any], resource_type: str, resource_id: str) -> str:
    for entry in _entries(bundle):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == resource_type and resource.get("id") == resource_id:
            full_url = entry.get("fullUrl")
            if isinstance(full_url, str) and full_url:
                return full_url
            return f"{resource_type}/{resource_id}"
    raise RawFHIRPatchError(f"FHIR resource not found: {resource_type}/{resource_id}")


def _patch_hash(patch: RawFHIRPatch, *, include_hash: bool) -> str:
    payload = patch.model_dump(mode="json")
    if not include_hash:
        payload.pop("patch_hash", None)
    return content_hash(payload)


def _opaque_id(resource_type: str, payload: Any) -> str:
    digest = content_hash(
        {
            "resource_type": resource_type,
            "payload": payload,
            "id_schema": "patient-tool-gym.opaque-fhir-id.v1",
        }
    )
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"https://patient-tool-gym.dev/fhir/id/{resource_type}/{digest}",
        )
    )


def _opaque_generated_resource_id(
    bundle: Mapping[str, Any],
    *,
    resource_type: str,
    source_resource_id: str,
    derivation: str,
) -> str:
    # ``derivation`` is intentionally only a hash-domain separator.  It never
    # appears in the emitted FHIR id or any model-visible narrative.
    return _opaque_id(
        resource_type,
        {
            "bundle_hash": content_hash(dict(bundle)),
            "source_resource_id": source_resource_id,
            "derivation": derivation,
        },
    )


def _shared_event_identifier(
    bundle: Mapping[str, Any],
    *,
    resource_type: str,
    resource_id: str,
) -> dict[str, str]:
    value = _opaque_id(
        "ClinicalEvent",
        {
            "bundle_hash": content_hash(dict(bundle)),
            "resource_type": resource_type,
            "source_resource_id": resource_id,
        },
    )
    return {"system": _CLINICAL_EVENT_IDENTIFIER_SYSTEM, "value": value}


def _ensure_shared_event_identifier(
    resource: dict[str, Any],
    identifier: dict[str, str],
) -> None:
    identifiers = resource.get("identifier")
    if identifiers is None:
        identifiers = []
        resource["identifier"] = identifiers
    if not isinstance(identifiers, list):
        raise RawFHIRPatchError("FHIR resource identifier must be a list")
    if identifier not in identifiers:
        identifiers.append(copy.deepcopy(identifier))


def _validate_shared_assertion_identity(resource: Mapping[str, Any]) -> None:
    """Reject records for which a cloned negative could denote another attempt."""

    resource_type = str(resource.get("resourceType"))
    required_time_choices: dict[str, tuple[str, ...]] = {
        "Observation": (
            "effectiveDateTime",
            "effectivePeriod",
            "effectiveTiming",
            "effectiveInstant",
        ),
        "Procedure": _PERFORMED_FIELDS,
        "Immunization": ("occurrenceDateTime", "occurrenceString"),
        "MedicationAdministration": ("effectiveDateTime", "effectivePeriod"),
        "MedicationRequest": ("authoredOn",),
    }
    choices = required_time_choices.get(resource_type)
    if choices is not None and not any(field in resource for field in choices):
        raise RawFHIRPatchError(f"{resource_type} conflict requires a shared clinical event time")
    # A deterministic shared business identifier is added to both records by the
    # compiler.  The required owner/code/time tuple prevents the pair from being
    # interpreted as unrelated same-code events before that identity is examined.


def _sanitize_derived_resource_metadata(resource: dict[str, Any]) -> None:
    resource.pop("text", None)
    meta = resource.get("meta")
    if isinstance(meta, dict):
        for field in ("versionId", "lastUpdated", "source"):
            meta.pop(field, None)
        if not meta:
            resource.pop("meta", None)


def _distractor_source(
    bundle: Mapping[str, Any],
    *,
    target: Mapping[str, Any],
    requested_tier: Literal[
        "cross_resource_routing",
        "same_resource_different_code",
        "same_resource_near_match",
    ],
    forbidden_codes: frozenset[tuple[str, str]] = frozenset(),
    required_system: str | None = None,
    required_encounter_reference: str | None = None,
    allow_tier_fallback: bool = True,
) -> tuple[dict[str, Any], str]:
    """Choose a deterministic distractor and record its actual difficulty tier."""

    target_type = str(target.get("resourceType"))
    target_id = str(target.get("id"))
    target_tokens = _clinical_display_tokens(target)
    candidates = [
        resource
        for index, entry in enumerate(_entries(bundle))
        if (resource := _entry_resource(entry, index)).get("resourceType")
        not in {"Patient", "Provenance"}
        and isinstance(resource.get("id"), str)
        and not (resource.get("resourceType") == target_type and resource.get("id") == target_id)
        and _resource_codes(resource).isdisjoint(forbidden_codes)
        and (
            required_system is None
            or any(system == required_system for system, _ in _resource_codes(resource))
        )
        and (
            required_encounter_reference is None
            or _encounter_reference(resource) == required_encounter_reference
        )
    ]
    same_resource = [
        resource
        for resource in candidates
        if resource.get("resourceType") == target_type
        and _resource_codes(resource).isdisjoint(_resource_codes(target))
    ]
    near_matches = [
        resource for resource in same_resource if target_tokens & _clinical_display_tokens(resource)
    ]
    cross_resource = [
        resource for resource in candidates if resource.get("resourceType") != target_type
    ]
    exact_tiers = {
        "same_resource_near_match": near_matches,
        "same_resource_different_code": same_resource,
        "cross_resource_routing": cross_resource,
    }
    if not exact_tiers[requested_tier]:
        if not allow_tier_fallback:
            raise RawFHIRDistractorUnavailableError(
                requested_tier=requested_tier,
                target_resource_type=target_type,
                target_resource_id=target_id,
            )
    tiers = {
        "same_resource_near_match": (
            (near_matches, "same_resource_near_match"),
            (same_resource, "same_resource_different_code"),
            (cross_resource, "cross_resource_routing"),
        ),
        "same_resource_different_code": (
            (same_resource, "same_resource_different_code"),
            (cross_resource, "cross_resource_routing"),
        ),
        "cross_resource_routing": ((cross_resource, "cross_resource_routing"),),
    }
    if not allow_tier_fallback:
        tiers = {requested_tier: ((exact_tiers[requested_tier], requested_tier),)}
    selected: list[dict[str, Any]] = []
    actual_tier = requested_tier
    for tier_candidates, tier_name in tiers[requested_tier]:
        if tier_candidates:
            selected = tier_candidates
            actual_tier = tier_name
            break
    if not selected:
        raise RawFHIRDistractorUnavailableError(
            requested_tier=requested_tier,
            target_resource_type=target_type,
            target_resource_id=target_id,
        )
    return min(
        selected,
        key=lambda item: (
            -len(target_tokens & _clinical_display_tokens(item)),
            _distractor_warning_risk(item),
            str(item.get("resourceType")),
            str(item.get("id")),
        ),
    ), actual_tier


def _primary_code(resource: Mapping[str, Any]) -> tuple[str, str] | None:
    codes = _resource_codes(resource)
    return min(codes) if codes else None


def _resource_codes(resource: Mapping[str, Any]) -> set[tuple[str, str]]:
    codes: set[tuple[str, str]] = set()
    for field in ("code", "medicationCodeableConcept", "vaccineCode"):
        concept = resource.get(field)
        if not isinstance(concept, Mapping):
            continue
        codings = concept.get("coding")
        if not isinstance(codings, list):
            continue
        for coding in codings:
            if isinstance(coding, Mapping):
                system, code = coding.get("system"), coding.get("code")
                if isinstance(system, str) and isinstance(code, str):
                    codes.add((system, code))
    return codes


def _encounter_reference(resource: Mapping[str, Any]) -> str | None:
    encounter = resource.get("encounter")
    if not isinstance(encounter, Mapping):
        return None
    reference = encounter.get("reference")
    return reference if isinstance(reference, str) and reference else None


def _clinical_display_tokens(resource: Mapping[str, Any]) -> set[str]:
    displays: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            display = value.get("display")
            text = value.get("text")
            if isinstance(display, str):
                displays.append(display)
            if isinstance(text, str):
                displays.append(text)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for field in ("code", "medicationCodeableConcept", "vaccineCode"):
        visit(resource.get(field))
    stop = {"and", "for", "oral", "procedure", "tablet", "the"}
    return {
        token
        for token in re.findall(r"[a-z0-9]+", " ".join(displays).casefold())
        if len(token) > 2 and token not in stop
    }


_TEMPORAL_POINT_FIELDS = (
    "effectiveDateTime",
    "effectiveInstant",
    "occurrenceDateTime",
    "performedDateTime",
    "authoredOn",
    "recordedDate",
    "onsetDateTime",
    "abatementDateTime",
    "issued",
)
_TEMPORAL_PERIOD_FIELDS = (
    "period",
    "effectivePeriod",
    "performedPeriod",
    "onsetPeriod",
    "abatementPeriod",
)


def _materialize_distinct_distractor(
    bundle: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    target_date: str | None = None,
) -> dict[str, Any]:
    """Derive a new same-context event and prove it is not a chart duplicate."""

    resource_type = str(source.get("resourceType"))
    source_id = source.get("id")
    if not isinstance(source_id, str):
        raise RawFHIRPatchError("distractor source requires an id")
    clone = copy.deepcopy(dict(source))
    clone["id"] = _opaque_generated_resource_id(
        bundle,
        resource_type=resource_type,
        source_resource_id=source_id,
        derivation=f"new-event-for-{target.get('resourceType')}/{target.get('id')}",
    )
    _sanitize_derived_resource_metadata(clone)
    clone["identifier"] = [
        {
            "system": _CLINICAL_EVENT_IDENTIFIER_SYSTEM,
            "value": _opaque_id(
                "ClinicalEvent",
                {
                    "bundle_hash": content_hash(dict(bundle)),
                    "source": (resource_type, source_id),
                    "target": (target.get("resourceType"), target.get("id")),
                },
            ),
        }
    ]
    source_anchor = _primary_temporal_instant(clone)
    if source_anchor is None:
        raise RawFHIRPatchError("distractor source requires a clinical event date/time")
    if target_date is not None:
        try:
            desired_date = date.fromisoformat(target_date)
        except ValueError as exc:
            raise RawFHIRPatchError("distractor target_date must be an ISO date") from exc
    else:
        target_anchor = _primary_temporal_instant(target)
        if target_anchor is None:
            raise RawFHIRPatchError("distractor target requires a clinical event date/time")
        desired_date = target_anchor.date()
    _shift_all_clinical_temporals(clone, desired_date - source_anchor.date())

    existing_signatures = {
        _clinical_semantic_signature(resource)
        for index, entry in enumerate(_entries(bundle))
        if (resource := _entry_resource(entry, index)).get("resourceType") == resource_type
    }
    if _clinical_semantic_signature(clone) in existing_signatures:
        aligned = copy.deepcopy(clone)
        aligned_anchor = _primary_temporal_instant(aligned)
        if aligned_anchor is None:
            raise RawFHIRPatchError("aligned distractor lost its clinical event time")
        forward = range(1, 24 * 60)
        backward = range(1, 24 * 60)
        for direction, offsets in ((1, forward), (-1, backward)):
            for minutes in offsets:
                candidate = copy.deepcopy(aligned)
                _shift_all_clinical_temporals(
                    candidate,
                    timedelta(minutes=direction * minutes),
                )
                candidate_anchor = _primary_temporal_instant(candidate)
                if candidate_anchor is None or candidate_anchor.date() != desired_date:
                    break
                if _clinical_semantic_signature(candidate) not in existing_signatures:
                    clone = candidate
                    break
            else:
                continue
            if _clinical_semantic_signature(clone) not in existing_signatures:
                break
        if _clinical_semantic_signature(clone) in existing_signatures:
            raise RawFHIRDistractorUnavailableError(
                requested_tier="same_resource_different_code",
                target_resource_type=str(target.get("resourceType")),
                target_resource_id=str(target.get("id")),
            )
    _set_generated_narrative(clone)
    return clone


def _primary_temporal_instant(resource: Mapping[str, Any]) -> datetime | None:
    for field in _TEMPORAL_POINT_FIELDS:
        value = resource.get(field)
        if isinstance(value, str):
            parsed = _parse_fhir_temporal(value)
            if parsed is not None:
                return parsed
    for field in _TEMPORAL_PERIOD_FIELDS:
        period = resource.get(field)
        if not isinstance(period, Mapping):
            continue
        start = period.get("start")
        if isinstance(start, str):
            parsed = _parse_fhir_temporal(start)
            if parsed is not None:
                return parsed
    return None


def _parse_fhir_temporal(value: str) -> datetime | None:
    try:
        if len(value) == 10:
            parsed_date = date.fromisoformat(value)
            return datetime(
                parsed_date.year,
                parsed_date.month,
                parsed_date.day,
                tzinfo=UTC,
            )
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _shift_all_clinical_temporals(
    resource: dict[str, Any],
    delta: timedelta,
) -> None:
    for field in _TEMPORAL_POINT_FIELDS:
        value = resource.get(field)
        if isinstance(value, str):
            shifted = _shift_fhir_temporal(value, delta)
            if shifted is not None:
                resource[field] = shifted
    for field in _TEMPORAL_PERIOD_FIELDS:
        period = resource.get(field)
        if not isinstance(period, dict):
            continue
        for endpoint in ("start", "end"):
            value = period.get(endpoint)
            if isinstance(value, str):
                shifted = _shift_fhir_temporal(value, delta)
                if shifted is not None:
                    period[endpoint] = shifted


def _shift_fhir_temporal(value: str, delta: timedelta) -> str | None:
    parsed = _parse_fhir_temporal(value)
    if parsed is None:
        return None
    shifted = parsed + delta
    if len(value) == 10:
        return shifted.date().isoformat()
    return shifted.isoformat().replace("+00:00", "Z")


def _clinical_semantic_signature(resource: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(resource))
    for field in ("id", "identifier", "meta", "text"):
        payload.pop(field, None)
    return content_hash(payload)


def _distractor_warning_risk(resource: Mapping[str, Any]) -> tuple[bool, int]:
    """Sort by concrete warning features, without arbitrary numeric weights."""

    missing_observation_performer = resource.get(
        "resourceType"
    ) == "Observation" and not resource.get("performer")
    explicit_version_count = 0

    def visit(value: Any) -> None:
        nonlocal explicit_version_count
        if isinstance(value, Mapping):
            if value.get("version") is not None:
                explicit_version_count += 1
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(resource)
    return missing_observation_performer, explicit_version_count


def _harden_conflict_clone(resource: dict[str, Any]) -> None:
    """Avoid multiplying source-only best-practice warnings in inserted records."""

    resource_type = resource.get("resourceType")
    if resource_type == "MedicationRequest":
        # The generated source asserts an old US Core profile whose 2017 medication
        # value set rejects otherwise valid current RxNorm concepts. The inserted
        # synthetic record remains base-R4-valid without making that false assertion.
        meta = resource.get("meta")
        if isinstance(meta, dict) and isinstance(meta.get("profile"), list):
            meta["profile"] = [
                profile
                for profile in meta["profile"]
                if "us-core-medicationrequest" not in str(profile)
            ]
            if not meta["profile"]:
                meta.pop("profile")
            if not meta:
                resource.pop("meta")


def _set_generated_narrative(resource: dict[str, Any]) -> None:
    resource_type = str(resource.get("resourceType", "Clinical resource"))
    display = _resource_display(resource)
    status = resource.get("status")
    parts = [display or resource_type]
    if isinstance(status, str) and status:
        parts.append(f"Status: {status}")
    summary = ". ".join(parts) + "."
    resource["text"] = {
        "status": "generated",
        "div": (f'<div xmlns="http://www.w3.org/1999/xhtml"><p>{html.escape(summary)}</p></div>'),
    }


def _resource_display(resource: Mapping[str, Any]) -> str | None:
    for field in (
        "code",
        "vaccineCode",
        "medicationCodeableConcept",
        "type",
    ):
        concept = resource.get(field)
        if not isinstance(concept, Mapping):
            continue
        text = concept.get("text")
        if isinstance(text, str) and text:
            return text
        codings = concept.get("coding")
        if isinstance(codings, list):
            for coding in codings:
                if isinstance(coding, Mapping) and isinstance(coding.get("display"), str):
                    return str(coding["display"])
    return None


def _entries(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    if bundle.get("resourceType") != "Bundle" or not isinstance(bundle.get("entry"), list):
        raise RawFHIRPatchError("raw FHIR input must be a Bundle with an entry list")
    return bundle["entry"]  # type: ignore[return-value]


def _entry_resource(entry: dict[str, Any], index: int) -> dict[str, Any]:
    resource = entry.get("resource")
    if not isinstance(resource, dict):
        raise RawFHIRPatchError(f"Bundle.entry[{index}] has no resource object")
    return resource


def _find_resource(
    bundle: Mapping[str, Any], resource_type: str, resource_id: str
) -> dict[str, Any] | None:
    matches = [
        resource
        for index, entry in enumerate(_entries(bundle))
        if (resource := _entry_resource(entry, index)).get("resourceType") == resource_type
        and resource.get("id") == resource_id
    ]
    if len(matches) > 1:
        raise RawFHIRPatchError(f"ambiguous FHIR resource: {resource_type}/{resource_id}")
    return matches[0] if matches else None


def _iter_references(value: Any, path: tuple[PathPart, ...] = ()):
    if isinstance(value, dict):
        reference = value.get("reference")
        if isinstance(reference, str):
            yield (*path, "reference"), reference
        for key, child in value.items():
            if key not in {"reference", "contained"}:
                yield from _iter_references(child, (*path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_references(child, (*path, index))


def _iter_references_strict(value: Any, path: tuple[PathPart, ...] = ()):
    """Traverse every reference, including references inside contained resources."""

    if isinstance(value, Mapping):
        reference = value.get("reference")
        if isinstance(reference, str):
            yield (*path, "reference"), reference
        for key, child in value.items():
            if key != "reference":
                yield from _iter_references_strict(child, (*path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_references_strict(child, (*path, index))


def _strict_reference_is_closed(
    reference: str,
    *,
    owner_contained_ids: set[str],
    typed_aliases: set[str],
    full_urls: set[str],
    history_aliases: set[str],
) -> bool:
    if reference.startswith("#"):
        return len(reference) > 1 and reference[1:] in owner_contained_ids
    if reference in full_urls or reference in typed_aliases or reference in history_aliases:
        return True
    if "/_history/" in reference:
        return False
    if "/" not in reference and not reference.startswith(("urn:", "http://", "https://")):
        return False
    return False


def _transaction_entry_has_resolvable_typed_identity(
    entry: Mapping[str, Any],
    *,
    index: int,
    resource_type: str,
    resource_id: str,
) -> bool:
    """Validate a transaction request and report whether ``Type/id`` is resolvable.

    A POST entry is assigned its server id only when the transaction executes, so
    other entries must reference its ``fullUrl`` URN.  A PUT to an exact logical
    identity may safely be referenced as ``ResourceType/id``.
    """

    request = entry.get("request")
    if not isinstance(request, Mapping):
        raise RawFHIRPatchError(f"transaction Bundle.entry[{index}] requires a request object")
    method = request.get("method")
    url = request.get("url")
    if not isinstance(method, str) or not isinstance(url, str) or not url:
        raise RawFHIRPatchError(
            f"transaction Bundle.entry[{index}] request requires method and url"
        )
    method = method.upper()
    if method == "POST":
        if url != resource_type:
            raise RawFHIRPatchError(
                f"transaction POST entry[{index}] url must equal {resource_type}"
            )
        full_url = entry.get("fullUrl")
        if not isinstance(full_url, str) or not full_url:
            raise RawFHIRPatchError(
                f"transaction POST entry[{index}] requires a fullUrl for references"
            )
        return False
    if method == "PUT":
        expected = f"{resource_type}/{resource_id}"
        if url == expected:
            return True
        if url.startswith(f"{resource_type}?") and url.partition("?")[2]:
            # A conditional update is not guaranteed to resolve to the in-memory
            # resource id, so references must still use the entry fullUrl.
            full_url = entry.get("fullUrl")
            if not isinstance(full_url, str) or not full_url:
                raise RawFHIRPatchError(
                    f"conditional transaction PUT entry[{index}] requires a fullUrl"
                )
            return False
        raise RawFHIRPatchError(f"transaction PUT entry[{index}] url does not identify {expected}")
    raise RawFHIRPatchError(
        f"transaction Bundle.entry[{index}] uses unsupported {method} with a resource"
    )


def _reference_is_closed(reference: str, aliases: set[str]) -> bool:
    if reference.startswith("#"):
        return True
    if reference in aliases:
        return True
    if "/_history/" in reference and reference.split("/_history/", 1)[0] in aliases:
        return True
    # An absolute URL is still a Reference, not proof that its target exists. Bundle-local
    # absolute references are already covered by ``aliases`` above. External references
    # must be admitted by the caller's explicit conditional-reference allowlist or supplied
    # as Bundle entries. This fails closed instead of silently accepting a dead local URL.
    return False


def _conditional_reference_type(reference: str) -> str | None:
    if "?" not in reference:
        return None
    resource_type, query = reference.split("?", 1)
    if not query or not resource_type or not resource_type.replace("_", "").isalnum():
        return None
    return resource_type


def _path_value(payload: Any, path: tuple[PathPart, ...]) -> tuple[bool, Any]:
    current = payload
    for part in path:
        if isinstance(part, int):
            if not isinstance(current, list) or part < 0 or part >= len(current):
                return False, None
            current = current[part]
        else:
            if not isinstance(current, dict) or part not in current:
                return False, None
            current = current[part]
    return True, current


def _set_path(payload: Any, path: tuple[PathPart, ...], value: Any) -> None:
    current = payload
    for part in path[:-1]:
        if isinstance(part, int):
            if not isinstance(current, list) or part < 0 or part >= len(current):
                raise RawFHIRPatchError("raw FHIR set path is unavailable")
            current = current[part]
        else:
            if not isinstance(current, dict):
                raise RawFHIRPatchError("raw FHIR set path crosses a non-object")
            current = current.setdefault(part, {})
    final = path[-1]
    if isinstance(final, int):
        if not isinstance(current, list) or final < 0 or final >= len(current):
            raise RawFHIRPatchError("raw FHIR set list index is unavailable")
        current[final] = value
    else:
        if not isinstance(current, dict):
            raise RawFHIRPatchError("raw FHIR set target is not an object")
        current[final] = value


def _remove_path(payload: Any, path: tuple[PathPart, ...]) -> None:
    current = payload
    for part in path[:-1]:
        present, current = _path_value(current, (part,))
        if not present:
            raise RawFHIRPatchError("raw FHIR remove path is unavailable")
    final = path[-1]
    if isinstance(final, int):
        if not isinstance(current, list) or final < 0 or final >= len(current):
            raise RawFHIRPatchError("raw FHIR remove list index is unavailable")
        current.pop(final)
    else:
        if not isinstance(current, dict) or final not in current:
            raise RawFHIRPatchError("raw FHIR remove target is unavailable")
        current.pop(final)
