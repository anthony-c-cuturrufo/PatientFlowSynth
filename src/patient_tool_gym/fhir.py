from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from patient_tool_gym.evidence import (
    FHIR_NORMALIZER_VERSION,
    EvidenceFormatError,
    PatientBoundaryError,
    PatientSnapshot,
    snapshot_from_payloads,
)

SELECTED_RESOURCE_TYPES = frozenset(
    {
        "Patient",
        "Encounter",
        "Condition",
        "Observation",
        "Procedure",
        "MedicationRequest",
        "MedicationAdministration",
        "DiagnosticReport",
        "DocumentReference",
        "ServiceRequest",
        "Immunization",
        "CarePlan",
        "Provenance",
    }
)


class FHIRNormalizationError(EvidenceFormatError):
    """Raised when a FHIR Bundle cannot be normalized safely."""


def load_fhir_bundle(path: Path) -> dict[str, Any]:
    try:
        with Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise FHIRNormalizationError("FHIR input is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise FHIRNormalizationError("FHIR input must be a JSON object")
    return payload


def normalize_fhir_file(path: Path, *, dataset: str = "synthea") -> PatientSnapshot:
    return normalize_fhir_bundle(load_fhir_bundle(path), dataset=dataset)


def normalize_fhir_bundle(
    bundle: Mapping[str, Any], *, dataset: str = "synthea"
) -> PatientSnapshot:
    if bundle.get("resourceType") != "Bundle":
        raise FHIRNormalizationError("FHIR input must have resourceType Bundle")
    entries = bundle.get("entry")
    if not isinstance(entries, list):
        raise FHIRNormalizationError("FHIR Bundle.entry must be a list")

    version_aware = bundle.get("type") == "history"
    bundle_resources: list[tuple[str | None, dict[str, Any]]] = []
    selected: list[tuple[str | None, dict[str, Any]]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not isinstance(entry.get("resource"), dict):
            raise FHIRNormalizationError(f"FHIR Bundle.entry[{index}] has no resource object")
        resource = dict(entry["resource"])
        full_url = entry.get("fullUrl")
        normalized_full_url = full_url if isinstance(full_url, str) else None
        bundle_resources.append((normalized_full_url, resource))
        resource_type = resource.get("resourceType")
        if resource_type not in SELECTED_RESOURCE_TYPES:
            continue
        resource_id = resource.get("id")
        if not isinstance(resource_id, str) or not resource_id:
            raise FHIRNormalizationError(f"selected FHIR resource at entry[{index}] is missing id")
        selected.append((normalized_full_url, resource))

    patients = [resource for _, resource in selected if resource["resourceType"] == "Patient"]
    if len(patients) != 1:
        raise FHIRNormalizationError("FHIR Bundle must contain exactly one selected Patient")
    patient_id = patients[0]["id"]

    patient_aliases = {f"Patient/{patient_id}", patient_id}
    for full_url, resource in selected:
        if resource["resourceType"] == "Patient" and full_url:
            patient_aliases.add(full_url)

    medication_aliases = _index_medications(bundle_resources)
    logical_groups = _clinical_logical_groups(
        selected,
        dataset=dataset,
        patient_id=patient_id,
    )
    normalized: list[dict[str, Any]] = []
    for full_url, resource in selected:
        resource_type = resource["resourceType"]
        if resource_type not in {"Patient", "Provenance"}:
            ownership_field = "patient" if resource_type == "Immunization" else "subject"
            owner = resource.get(ownership_field)
            owner_reference = owner.get("reference") if isinstance(owner, dict) else None
            if not isinstance(owner_reference, str):
                raise FHIRNormalizationError(
                    f"{resource_type}/{resource['id']} is missing {ownership_field}.reference"
                )
            clean_owner = owner_reference.rstrip("/")
            if clean_owner not in patient_aliases:
                raise PatientBoundaryError(patient_id, clean_owner, len(normalized) + 1)
        elif resource_type == "Provenance":
            targets = resource.get("target")
            target_references = (
                {
                    item.get("reference")
                    for item in targets
                    if isinstance(item, dict) and isinstance(item.get("reference"), str)
                }
                if isinstance(targets, list)
                else set()
            )
            selected_aliases = {
                alias
                for selected_full_url, selected_resource in selected
                for alias in _resource_reference_aliases(
                    selected_resource["resourceType"],
                    selected_resource["id"],
                    full_url=selected_full_url,
                )
            }
            if not target_references or target_references.isdisjoint(selected_aliases):
                logical_targets = {
                    reference.rsplit("/_history/", 1)[0]
                    for reference in target_references
                    if "/_history/" in reference
                }
                if logical_targets.isdisjoint(selected_aliases):
                    raise FHIRNormalizationError(
                        f"Provenance/{resource['id']} has no target in the patient bundle"
                    )
        medication_resolution = _resolve_medication_reference(resource, medication_aliases)
        payload = _normalize_resource(
            resource,
            patient_id=patient_id,
            dataset=dataset,
            full_url=full_url,
            logical_evidence_group=logical_groups.get((resource_type, resource["id"])),
            medication_resolution=medication_resolution,
            version_aware=version_aware,
        )
        normalized.append(payload)
        if resource_type == "Observation":
            normalized.extend(
                _normalize_observation_components(
                    resource,
                    parent=payload,
                    patient_id=patient_id,
                    dataset=dataset,
                    version_aware=version_aware,
                )
            )

    return snapshot_from_payloads(
        normalized,
        normalizer_version=FHIR_NORMALIZER_VERSION,
    )


def _index_medications(
    bundle_resources: list[tuple[str | None, dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    aliases: dict[str, list[dict[str, Any]]] = {}
    for full_url, resource in bundle_resources:
        if resource.get("resourceType") != "Medication":
            continue
        resource_id = resource.get("id")
        if not isinstance(resource_id, str) or not resource_id:
            continue
        for alias in _resource_reference_aliases("Medication", resource_id, full_url=full_url):
            aliases.setdefault(alias, []).append(resource)
    return aliases


def _resolve_medication_reference(
    resource: dict[str, Any],
    medication_aliases: dict[str, list[dict[str, Any]]],
) -> tuple[str, dict[str, Any]] | None:
    if resource.get("resourceType") != "MedicationRequest":
        return None
    reference_payload = resource.get("medicationReference")
    reference = reference_payload.get("reference") if isinstance(reference_payload, dict) else None
    if not isinstance(reference, str) or not reference:
        return None
    candidates = medication_aliases.get(reference, [])
    if len(candidates) != 1:
        return None
    return reference, candidates[0]


def _document_logical_groups(
    selected: list[tuple[str | None, dict[str, Any]]],
    *,
    dataset: str,
    patient_id: str,
) -> dict[tuple[str, str], str]:
    report_aliases: dict[str, list[dict[str, Any]]] = {}
    for full_url, resource in selected:
        if resource.get("resourceType") != "DiagnosticReport":
            continue
        resource_id = resource["id"]
        for alias in _resource_reference_aliases(
            "DiagnosticReport", resource_id, full_url=full_url
        ):
            report_aliases.setdefault(alias, []).append(resource)

    groups: dict[tuple[str, str], str] = {}
    for _, document in selected:
        if document.get("resourceType") != "DocumentReference":
            continue
        identifiers = document.get("identifier")
        if not isinstance(identifiers, list):
            continue
        matched_reports: dict[str, dict[str, Any]] = {}
        for identifier in identifiers:
            value = identifier.get("value") if isinstance(identifier, dict) else None
            if not isinstance(value, str):
                continue
            candidates = report_aliases.get(value, [])
            if len(candidates) == 1:
                report = candidates[0]
                matched_reports[report["id"]] = report
        if len(matched_reports) != 1:
            continue
        report = next(iter(matched_reports.values()))
        report_evidence_id = f"{dataset}:{patient_id}:DiagnosticReport:{report['id']}"
        group = f"record:{report_evidence_id}"
        groups[("DiagnosticReport", report["id"])] = group
        groups[("DocumentReference", document["id"])] = group
    return groups


def _clinical_logical_groups(
    selected: list[tuple[str | None, dict[str, Any]]],
    *,
    dataset: str,
    patient_id: str,
) -> dict[tuple[str, str], str]:
    """Group records only when FHIR business identifiers assert one event/claim.

    Same code, patient, encounter, and date are not sufficient: repeated tests or
    vaccination attempts can legitimately share all four.  A shared Identifier is
    the explicit FHIR assertion that two logical resources describe one event.
    """

    groups = _document_logical_groups(selected, dataset=dataset, patient_id=patient_id)
    eligible_types = {
        "Condition",
        "Observation",
        "Procedure",
        "Immunization",
        "MedicationRequest",
        "MedicationAdministration",
    }
    token_members: dict[tuple[str, str, str], set[tuple[str, str]]] = {}
    for _, resource in selected:
        resource_type = resource.get("resourceType")
        resource_id = resource.get("id")
        if resource_type not in eligible_types or not isinstance(resource_id, str):
            continue
        identifiers = resource.get("identifier")
        if not isinstance(identifiers, list):
            continue
        for identifier in identifiers:
            if not isinstance(identifier, dict):
                continue
            system = identifier.get("system")
            value = identifier.get("value")
            if not isinstance(system, str) or not system:
                continue
            if not isinstance(value, str) or not value:
                continue
            token_members.setdefault((resource_type, system, value), set()).add(
                (resource_type, resource_id)
            )

    parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(item: tuple[str, str]) -> tuple[str, str]:
        root = parent.setdefault(item, item)
        if root != item:
            parent[item] = find(root)
        return parent[item]

    def union(left: tuple[str, str], right: tuple[str, str]) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    shared_tokens: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
    for token, members in sorted(token_members.items()):
        ordered = sorted(members)
        if len(ordered) < 2:
            continue
        for member in ordered[1:]:
            union(ordered[0], member)
        for member in ordered:
            shared_tokens.setdefault(member, []).append(token)

    components: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for member in parent:
        components.setdefault(find(member), []).append(member)
    for members in components.values():
        identity_tokens = sorted(
            {
                token
                for member in members
                for token in shared_tokens.get(member, [])
            }
        )
        if not identity_tokens:
            continue
        group = "clinical-identifier:" + "|".join(
            f"{resource_type}:{system}:{value}"
            for resource_type, system, value in identity_tokens
        )
        for member in members:
            groups[member] = group
    return groups


def _resource_reference_aliases(
    resource_type: str, resource_id: str, *, full_url: str | None
) -> set[str]:
    aliases = {
        resource_id,
        f"{resource_type}/{resource_id}",
        f"urn:uuid:{resource_id}",
    }
    if full_url:
        aliases.add(full_url)
    return aliases


def _normalize_resource(
    resource: dict[str, Any],
    *,
    patient_id: str,
    dataset: str,
    full_url: str | None,
    logical_evidence_group: str | None = None,
    medication_resolution: tuple[str, dict[str, Any]] | None = None,
    version_aware: bool = False,
) -> dict[str, Any]:
    resource_type = resource["resourceType"]
    resource_id = resource["id"]
    evidence_id = f"{dataset}:{patient_id}:{resource_type}:{resource_id}"
    version_id = resource.get("meta", {}).get("versionId")
    if version_aware:
        if not isinstance(version_id, str) or not version_id:
            raise FHIRNormalizationError(
                f"history Bundle resource {resource_type}/{resource_id} is missing meta.versionId"
            )
        evidence_id += f":version:{version_id}"
    resolved_medication = medication_resolution[1] if medication_resolution else None
    code = _resource_code(resource, resolved_medication=resolved_medication)
    display = _display(resource, code)
    value = _resource_value(resource)
    time = _resource_time(resource)
    references = _extract_references(resource)
    source: dict[str, Any] = {
        "format": "fhir.Bundle.entry.resource",
        "resource_id": resource_id,
        "resource_type": resource_type,
    }
    if full_url:
        source["full_url"] = full_url
    if medication_resolution is not None:
        medication_reference, medication = medication_resolution
        source["resolved_medication"] = {
            "reference": medication_reference,
            "resource_id": medication["id"],
            "resource_type": "Medication",
        }
    return {
        "schema_version": "patient-tool-gym.evidence.v1",
        "normalizer_version": FHIR_NORMALIZER_VERSION,
        "category": _resource_category(resource),
        "code": code,
        "display": display,
        "evidence_id": evidence_id,
        "logical_evidence_group": logical_evidence_group or f"record:{evidence_id}",
        "patient_id": patient_id,
        "resource_id": resource_id,
        "references": references,
        "resource_type": resource_type,
        "source": source,
        "source_dataset": dataset,
        "source_format": "fhir-r4",
        "status": resource.get("status"),
        "text": _search_text(display, value),
        "time": time,
        "value": value,
        "attributes": _resource_attributes(resource),
    }


def _normalize_observation_components(
    resource: dict[str, Any],
    *,
    parent: dict[str, Any],
    patient_id: str,
    dataset: str,
    version_aware: bool = False,
) -> list[dict[str, Any]]:
    components = resource.get("component")
    if not isinstance(components, list):
        return []
    payloads: list[dict[str, Any]] = []
    for index, component in enumerate(components):
        if not isinstance(component, dict):
            continue
        component_code = _normalize_concept(component.get("code"))
        display = component_code.get("display")
        if not isinstance(display, str) or not display:
            display = f"{parent['display']} component {index + 1}"
        value = _component_value(component)
        component_resource_id = f"{resource['id']}.component.{index:04d}"
        evidence_id = f"{dataset}:{patient_id}:Observation:{resource['id']}:component:{index:04d}"
        if version_aware:
            version_id = resource.get("meta", {}).get("versionId")
            if not isinstance(version_id, str) or not version_id:
                raise FHIRNormalizationError("history Observation component lacks versionId")
            evidence_id += f":version:{version_id}"
        references = dict(parent["references"])
        references["component_of"] = {"reference": f"Observation/{resource['id']}"}
        inherited_attributes = {
            key: item for key, item in parent.get("attributes", {}).items() if key != "components"
        }
        inherited_attributes.update(
            {
                "component_index": index,
                "derived_from": "Observation.component",
                "parent_observation_code": parent["code"],
            }
        )
        component_absent_reason = component.get("dataAbsentReason")
        if isinstance(component_absent_reason, dict):
            inherited_attributes["data_absent_reason"] = _normalize_concept(component_absent_reason)
        payloads.append(
            {
                "schema_version": "patient-tool-gym.evidence.v1",
                "normalizer_version": FHIR_NORMALIZER_VERSION,
                "category": parent["category"],
                "code": component_code,
                "display": display,
                "evidence_id": evidence_id,
                "logical_evidence_group": parent["logical_evidence_group"],
                "patient_id": patient_id,
                "resource_id": component_resource_id,
                "references": references,
                "resource_type": "Observation",
                "source": {
                    "format": "fhir.Bundle.entry.resource.component",
                    "resource_id": component_resource_id,
                    "resource_type": "Observation",
                    "parent_evidence_id": parent["evidence_id"],
                    "parent_resource_id": resource["id"],
                    "component_index": index,
                },
                "source_dataset": dataset,
                "source_format": "fhir-r4",
                "status": parent.get("status"),
                "text": _search_text(display, value),
                "time": dict(parent["time"]),
                "value": value,
                "attributes": inherited_attributes,
            }
        )
    return payloads


def _resource_code(
    resource: dict[str, Any], *, resolved_medication: dict[str, Any] | None = None
) -> dict[str, Any]:
    resource_type = resource["resourceType"]
    candidate: Any = resource.get("code")
    if resource_type == "Encounter":
        types = resource.get("type")
        candidate = types[0] if isinstance(types, list) and types else resource.get("class")
    elif resource_type in {"MedicationRequest", "MedicationAdministration"}:
        candidate = resource.get("medicationCodeableConcept")
        if (
            candidate is None
            and resource_type == "MedicationRequest"
            and resolved_medication is not None
        ):
            candidate = resolved_medication.get("code")
    elif resource_type == "DocumentReference":
        candidate = resource.get("type")
    elif resource_type == "Immunization":
        candidate = resource.get("vaccineCode")
    elif resource_type == "CarePlan":
        categories = resource.get("category")
        candidate = _care_plan_concept(categories)
    elif resource_type == "Provenance":
        candidate = resource.get("activity")
    return _normalize_concept(candidate)


def _care_plan_concept(categories: Any) -> dict[str, Any] | None:
    if not isinstance(categories, list):
        return None
    concepts = [item for item in categories if isinstance(item, dict)]
    for concept in concepts:
        normalized = _normalize_concept(concept)
        if normalized.get("code") not in {None, "", "assess-plan"}:
            return concept
    return concepts[0] if concepts else None


def _normalize_concept(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    codings = value.get("coding")
    if isinstance(codings, list):
        normalized = sorted(
            {
                (
                    coding.get("system"),
                    coding.get("code"),
                    coding.get("display"),
                )
                for coding in codings
                if isinstance(coding, dict)
            },
            key=lambda item: tuple(part or "" for part in item),
        )
    elif isinstance(value.get("code"), str):
        normalized = [(value.get("system"), value.get("code"), value.get("display"))]
    else:
        normalized = []
    first = normalized[0] if normalized else (None, None, None)
    return {
        "system": first[0],
        "code": first[1],
        "display": value.get("text") or first[2],
        "codings": [
            {"system": system, "code": code, "display": display}
            for system, code, display in normalized
        ],
    }


def _display(resource: dict[str, Any], code: dict[str, Any]) -> str:
    if isinstance(code.get("display"), str) and code["display"]:
        return code["display"]
    if resource["resourceType"] == "Patient":
        names = resource.get("name")
        if isinstance(names, list) and names and isinstance(names[0], dict):
            family = names[0].get("family", "")
            given = names[0].get("given", [])
            return " ".join([*(str(item) for item in given), str(family)]).strip()
    for field in ("title", "description", "conclusion"):
        if isinstance(resource.get(field), str) and resource[field]:
            return resource[field]
    return f"{resource['resourceType']}/{resource['id']}"


def _resource_category(resource: dict[str, Any]) -> str:
    category = resource.get("category")
    if isinstance(category, list) and category:
        concept = _normalize_concept(category[0])
        return str(concept.get("code") or concept.get("display") or "").casefold()
    if isinstance(category, dict):
        concept = _normalize_concept(category)
        return str(concept.get("code") or concept.get("display") or "").casefold()
    return {
        "MedicationRequest": "medication_request",
        "MedicationAdministration": "medication_administration",
        "DocumentReference": "note",
        "Provenance": "provenance",
    }.get(resource["resourceType"], resource["resourceType"].casefold())


def _resource_value(resource: dict[str, Any]) -> Any:
    resource_type = resource["resourceType"]
    if resource_type == "Observation":
        quantity = resource.get("valueQuantity")
        if isinstance(quantity, dict):
            number = quantity.get("value")
            if not isinstance(number, (int, float)) or isinstance(number, bool):
                raise FHIRNormalizationError(
                    f"Observation/{resource['id']} valueQuantity.value must be numeric"
                )
            return {
                "value": number,
                "unit": quantity.get("unit") or quantity.get("code"),
                "system": quantity.get("system"),
                "code": quantity.get("code"),
                "comparator": quantity.get("comparator"),
            }
        for field in ("valueString", "valueBoolean", "valueInteger", "valueDecimal"):
            if field in resource:
                return resource[field]
        concept = resource.get("valueCodeableConcept")
        if isinstance(concept, dict):
            return _normalize_concept(concept)
        components = _normalized_components(resource)
        if components is not None:
            return {"components": components}
        return None
    if resource_type == "Patient":
        return {
            "active": resource.get("active"),
            "gender": resource.get("gender"),
            "birth_date": resource.get("birthDate"),
            "deceased_boolean": resource.get("deceasedBoolean"),
            "deceased_date_time": resource.get("deceasedDateTime"),
        }
    if resource_type == "DocumentReference":
        return _document_text(resource)
    if resource_type == "DiagnosticReport":
        conclusion = resource.get("conclusion")
        if isinstance(conclusion, str) and conclusion:
            return conclusion
        return _attachment_text(resource, field="presentedForm")
    if resource_type == "Provenance":
        activity = _normalize_concept(resource.get("activity"))
        return activity.get("code") or activity.get("display")
    return resource.get("status")


def _document_text(resource: dict[str, Any]) -> str | None:
    return _attachment_text(resource, field="content", nested=True)


def _attachment_text(resource: dict[str, Any], *, field: str, nested: bool = False) -> str | None:
    attachments = resource.get(field)
    if not isinstance(attachments, list):
        return None
    for item in attachments:
        if not isinstance(item, dict):
            continue
        attachment = item.get("attachment") if nested else item
        if not isinstance(attachment, dict):
            continue
        content_type = attachment.get("contentType")
        encoded = attachment.get("data")
        media_type = (
            content_type.split(";", 1)[0].strip().casefold()
            if isinstance(content_type, str)
            else ""
        )
        if not isinstance(encoded, str) or not (
            media_type.startswith("text/") or media_type == "application/json"
        ):
            continue
        try:
            return base64.b64decode(encoded, validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError) as exc:
            raise FHIRNormalizationError(
                f"{resource['resourceType']}/{resource['id']} has invalid inline text data"
            ) from exc
    return None


def _component_value(component: dict[str, Any]) -> Any:
    quantity = component.get("valueQuantity")
    if isinstance(quantity, dict):
        return {
            "value": quantity.get("value"),
            "unit": quantity.get("unit") or quantity.get("code"),
            "system": quantity.get("system"),
            "code": quantity.get("code"),
        }
    for field in ("valueString", "valueBoolean", "valueInteger", "valueDecimal"):
        if field in component:
            return component[field]
    concept = component.get("valueCodeableConcept")
    if isinstance(concept, dict):
        return _normalize_concept(concept)
    return None


def _normalized_components(resource: dict[str, Any]) -> list[dict[str, Any]] | None:
    components = resource.get("component")
    if not isinstance(components, list):
        return None
    return [
        {
            "code": _normalize_concept(component.get("code")),
            "value": _component_value(component),
        }
        for component in components
        if isinstance(component, dict)
    ]


def _resource_time(resource: dict[str, Any]) -> dict[str, Any]:
    if resource["resourceType"] == "Condition":
        time: dict[str, Any] = {}
        recorded = resource.get("recordedDate")
        onset = resource.get("onsetDateTime")
        onset_period = resource.get("onsetPeriod")
        abatement = resource.get("abatementDateTime")
        abatement_period = resource.get("abatementPeriod")
        if isinstance(recorded, str):
            time["recorded"] = recorded
        if isinstance(onset, str):
            time["onset"] = onset
        if isinstance(onset_period, dict):
            if isinstance(onset_period.get("start"), str):
                time["onset_start"] = onset_period["start"]
            if isinstance(onset_period.get("end"), str):
                time["onset_end"] = onset_period["end"]
        if isinstance(abatement, str):
            time["abatement"] = abatement
        if isinstance(abatement_period, dict):
            if isinstance(abatement_period.get("start"), str):
                time["abatement_start"] = abatement_period["start"]
            if isinstance(abatement_period.get("end"), str):
                time["abatement_end"] = abatement_period["end"]
        if time:
            return time
    if resource["resourceType"] == "Observation":
        time = {}
        effective_period = resource.get("effectivePeriod")
        if isinstance(effective_period, dict):
            if isinstance(effective_period.get("start"), str):
                time["start"] = effective_period["start"]
            if isinstance(effective_period.get("end"), str):
                time["end"] = effective_period["end"]
        effective = resource.get("effectiveDateTime")
        if isinstance(effective, str):
            time["start"] = effective
        issued = resource.get("issued")
        if isinstance(issued, str):
            time["issued"] = issued
        if time:
            return time
    for field in ("effectivePeriod", "performedPeriod", "period"):
        period = resource.get(field)
        if isinstance(period, dict):
            return {
                key: value
                for key, value in {"start": period.get("start"), "end": period.get("end")}.items()
                if isinstance(value, str)
            }
    for field in (
        "effectiveDateTime",
        "performedDateTime",
        "occurrenceDateTime",
        "authoredOn",
        "onsetDateTime",
        "recordedDate",
        "date",
        "created",
        "issued",
        "recorded",
    ):
        value = resource.get(field)
        if isinstance(value, str):
            key = "recorded" if field in {"recordedDate", "date", "issued", "recorded"} else "start"
            return {key: value}
    return {}


def _extract_references(resource: dict[str, Any]) -> dict[str, Any]:
    collected: dict[str, list[dict[str, Any]]] = {}

    def walk(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            reference = value.get("reference")
            if isinstance(reference, str):
                relation = ".".join(path)
                collected.setdefault(relation, []).append(
                    {
                        "reference": reference,
                        **(
                            {"display": value["display"]}
                            if isinstance(value.get("display"), str)
                            else {}
                        ),
                    }
                )
                return
            for key, child in value.items():
                if key not in {"text", "contained"}:
                    walk(child, (*path, key))
        elif isinstance(value, list):
            for child in value:
                walk(child, path)

    for key, value in resource.items():
        if key not in {"resourceType", "id", "text", "contained"}:
            walk(value, (key,))
    return {
        relation: values[0] if len(values) == 1 else values
        for relation, values in sorted(collected.items())
    }


def _resource_attributes(resource: dict[str, Any]) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    meta = resource.get("meta")
    if isinstance(meta, dict):
        normalized_meta = {
            key: meta[key]
            for key in ("versionId", "lastUpdated", "source", "profile", "security", "tag")
            if key in meta
        }
        if normalized_meta:
            attributes["resource_meta"] = normalized_meta
    for field in (
        "intent",
        "priority",
        "verificationStatus",
        "clinicalStatus",
        "docStatus",
    ):
        value = resource.get(field)
        if value is not None:
            attributes[field] = _normalize_concept(value) if isinstance(value, dict) else value
    if resource["resourceType"] == "Observation" and isinstance(resource.get("component"), list):
        attributes["components"] = _normalized_components(resource) or []
    if resource["resourceType"] == "Observation":
        absent_reason = resource.get("dataAbsentReason")
        if isinstance(absent_reason, dict):
            attributes["data_absent_reason"] = _normalize_concept(absent_reason)
    if resource["resourceType"] in {"Encounter", "Procedure"}:
        reason_codes = _normalize_concept_list(resource.get("reasonCode"))
        if reason_codes:
            attributes["reason_codes"] = reason_codes
    if resource["resourceType"] == "Encounter":
        encounter_class = resource.get("class")
        if isinstance(encounter_class, dict):
            attributes["encounter_class"] = _normalize_concept(encounter_class)
        hospitalization = resource.get("hospitalization")
        discharge = (
            hospitalization.get("dischargeDisposition")
            if isinstance(hospitalization, dict)
            else None
        )
        if isinstance(discharge, dict):
            attributes["discharge_disposition"] = _normalize_concept(discharge)
    if resource["resourceType"] == "CarePlan":
        categories = _normalize_concept_list(resource.get("category"))
        if categories:
            attributes["categories"] = categories
        activities = _care_plan_activities(resource)
        if activities:
            attributes["activities"] = activities
    if resource["resourceType"] == "Immunization" and isinstance(
        resource.get("primarySource"), bool
    ):
        attributes["primary_source"] = resource["primarySource"]
    if resource["resourceType"] == "DocumentReference":
        description = resource.get("description")
        if isinstance(description, str) and description:
            attributes["description"] = description
        content = resource.get("content")
        if isinstance(content, list):
            attachments = []
            for item in content:
                attachment = item.get("attachment") if isinstance(item, dict) else None
                if not isinstance(attachment, dict):
                    continue
                attachments.append(
                    {
                        key: attachment[key]
                        for key in ("contentType", "language", "title", "creation")
                        if isinstance(attachment.get(key), str)
                    }
                )
            if attachments:
                attributes["attachments"] = attachments
    if resource["resourceType"] == "Provenance":
        activity = resource.get("activity")
        if isinstance(activity, dict):
            attributes["activity"] = _normalize_concept(activity)
        agents = resource.get("agent")
        if isinstance(agents, list):
            attributes["agents"] = [
                {key: item[key] for key in ("type", "role", "who", "onBehalfOf") if key in item}
                for item in agents
                if isinstance(item, dict)
            ]
        entities = resource.get("entity")
        if isinstance(entities, list):
            attributes["entities"] = [
                {key: item[key] for key in ("role", "what", "agent") if key in item}
                for item in entities
                if isinstance(item, dict)
            ]
    return attributes


def _normalize_concept_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [_normalize_concept(item) for item in value if isinstance(item, dict)]


def _care_plan_activities(resource: dict[str, Any]) -> list[dict[str, Any]]:
    activities = resource.get("activity")
    if not isinstance(activities, list):
        return []
    normalized: list[dict[str, Any]] = []
    for index, activity in enumerate(activities):
        detail = activity.get("detail") if isinstance(activity, dict) else None
        if not isinstance(detail, dict):
            continue
        item: dict[str, Any] = {"index": index}
        code = detail.get("code")
        if isinstance(code, dict):
            item["code"] = _normalize_concept(code)
        if isinstance(detail.get("status"), str):
            item["status"] = detail["status"]
        if isinstance(detail.get("description"), str):
            item["description"] = detail["description"]
        location = detail.get("location")
        if isinstance(location, dict) and isinstance(location.get("display"), str):
            item["location_display"] = location["display"]
        normalized.append(item)
    return normalized


def _search_text(display: str, value: Any) -> str:
    if isinstance(value, dict) and set(value).issuperset({"value", "unit"}):
        rendered = f"{value['value']} {value['unit']}"
    elif isinstance(value, dict):
        rendered = next(
            (
                str(value[key])
                for key in ("display", "text", "code")
                if isinstance(value.get(key), (str, int, float, bool))
            ),
            "",
        )
    elif isinstance(value, (str, int, float, bool)):
        rendered = str(value)
    else:
        rendered = ""
    return " ".join(part for part in (display, rendered) if part)


def iter_selected_counts(bundle: Mapping[str, Any]) -> Iterable[tuple[str, int]]:
    counts: dict[str, int] = {}
    for entry in bundle.get("entry", []):
        if isinstance(entry, dict) and isinstance(entry.get("resource"), dict):
            resource_type = entry["resource"].get("resourceType")
            if resource_type in SELECTED_RESOURCE_TYPES:
                counts[resource_type] = counts.get(resource_type, 0) + 1
    return sorted(counts.items())
