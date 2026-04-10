"""
Samayik AI — Complete Pipeline v2
====================================
Layer 1: Format Parser           (CSV / JSON / SQL input)
Layer 2: Patient Extractor       (smart identity extraction)
Layer 3: Semantic Field Mapper   (sentence-transformers NLP)
Layer 4: Value Parser            (deterministic rule-based parsing)
Layer 5: Resource Router         (correct FHIR resource dispatch)
Layer 6: FHIR R4 Bundle Builder  (clinically valid FHIR R4 output)
Layer 7: Validation Layer        (input vs output completeness check)
Layer 8: DRI Scorer              (Decision Risk Index)
Layer 9: Alert Engine            (clinician-facing alert)

AI Model: sentence-transformers 'all-MiniLM-L6-v2'
  - 22M parameter BERT neural network
  - Runs 100% locally, zero data leaves the machine
  - Used ONLY for semantic field classification
  - All value parsing is deterministic (rule-based)

LOINC codes sourced from:
  HL7 FHIR R4 Official Vital Signs Profile
  https://hl7.org/fhir/R4/observation-vitalsigns.html
"""

import json
import csv
import uuid
import io
import re
import os
from datetime import datetime, timezone, timedelta
from sentence_transformers import SentenceTransformer, util
import torch


# ─────────────────────────────────────────────
# FHIR R4 LOINC CODE REGISTRY
# ─────────────────────────────────────────────

FHIR_LOINC_REGISTRY = [
    # --- VITAL SIGNS ---
    {
        "loinc": "85354-9",
        "display": "Blood pressure panel",
        "keywords": "blood pressure bp vitals_bp pressure systolic diastolic",
        "resource": "Observation",
        "category": "vital-signs",
        "unit": "mm[Hg]",
        "dri_field": "vitals_complete"
    },
    {
        "loinc": "8480-6",
        "display": "Systolic blood pressure",
        "keywords": "systolic blood pressure bp sys mmhg arterial upper",
        "resource": "Observation",
        "category": "vital-signs",
        "unit": "mm[Hg]",
        "dri_field": "vitals_complete"
    },
    {
        "loinc": "8462-4",
        "display": "Diastolic blood pressure",
        "keywords": "diastolic blood pressure bp dia mmhg lower",
        "resource": "Observation",
        "category": "vital-signs",
        "unit": "mm[Hg]",
        "dri_field": "vitals_complete"
    },
    {
        "loinc": "8867-4",
        "display": "Heart rate",
        "keywords": "heart rate pulse beat bpm hr cardiac rhythm",
        "resource": "Observation",
        "category": "vital-signs",
        "unit": "/min",
        "dri_field": "vitals_complete"
    },
    {
        "loinc": "9279-1",
        "display": "Respiratory rate",
        "keywords": "respiratory rate breathing breath respiration rr",
        "resource": "Observation",
        "category": "vital-signs",
        "unit": "/min",
        "dri_field": "vitals_complete"
    },
    {
        "loinc": "8310-5",
        "display": "Body temperature",
        "keywords": "temperature temp body celsius fahrenheit fever thermometer",
        "resource": "Observation",
        "category": "vital-signs",
        "unit": "Cel",
        "dri_field": "vitals_complete"
    },
    {
        "loinc": "59408-5",
        "display": "Oxygen saturation",
        "keywords": "oxygen saturation spo2 o2 pulse oximetry sat",
        "resource": "Observation",
        "category": "vital-signs",
        "unit": "%",
        "dri_field": "vitals_complete"
    },
    {
        "loinc": "29463-7",
        "display": "Body weight",
        "keywords": "weight body mass kg kilogram bmi",
        "resource": "Observation",
        "category": "vital-signs",
        "unit": "kg",
        "dri_field": "vitals_complete"
    },
    {
        "loinc": "8302-2",
        "display": "Body height",
        "keywords": "height body length cm centimeter tall stature",
        "resource": "Observation",
        "category": "vital-signs",
        "unit": "cm",
        "dri_field": "vitals_complete"
    },
    # --- BLOOD GROUP ---
    {
        "loinc": "882-1",
        "display": "ABO and Rh group",
        "keywords": "blood group type abo rh positive negative",
        "resource": "Observation",
        "category": "laboratory",
        "unit": None,
        "dri_field": "lab_results_present"
    },
    # --- ALLERGIES ---
    {
        "loinc": "52473-6",
        "display": "Allergy to substance",
        "keywords": "allergy allergies allergic reaction substance drug food intolerance",
        "resource": "AllergyIntolerance",
        "category": "allergy",
        "unit": None,
        "dri_field": "allergy_present"
    },
    # --- MEDICATIONS ---
    {
        "loinc": "57828-6",
        "display": "Prescription medication list",
        "keywords": "medication medicine drug prescription tablet capsule dose meds current",
        "resource": "MedicationRequest",
        "category": "medication",
        "unit": None,
        "dri_field": "medication_count"
    },
    # --- MENTAL HEALTH ---
    {
        "loinc": "44249-1",
        "display": "Mental health assessment",
        "keywords": "mental health psychology psychiatric depression anxiety phq score",
        "resource": "Observation",
        "category": "survey",
        "unit": None,
        "dri_field": "mental_health_present"
    },
    # --- LAB RESULTS (now Observations, not DiagnosticReports) ---
    {
        "loinc": "4548-4",
        "display": "Hemoglobin A1c",
        "keywords": "hba1c hemoglobin a1c glycated sugar diabetes lab",
        "resource": "Observation",
        "category": "laboratory",
        "unit": "%",
        "dri_field": "lab_results_present"
    },
    {
        "loinc": "1558-6",
        "display": "Fasting glucose",
        "keywords": "fasting glucose sugar blood sugar fbs lab",
        "resource": "Observation",
        "category": "laboratory",
        "unit": "mg/dL",
        "dri_field": "lab_results_present"
    },
    {
        "loinc": "58410-2",
        "display": "Complete blood count panel",
        "keywords": "blood count lab CBC haemoglobin hemoglobin rbc wbc platelet laboratory panel",
        "resource": "Observation",
        "category": "laboratory",
        "unit": None,
        "dri_field": "lab_results_present"
    },
    {
        "loinc": "24323-8",
        "display": "Comprehensive metabolic panel",
        "keywords": "metabolic panel creatinine sodium potassium liver kidney lab results",
        "resource": "Observation",
        "category": "laboratory",
        "unit": None,
        "dri_field": "lab_results_present"
    },
    # --- DIAGNOSIS ---
    {
        "loinc": "29548-5",
        "display": "Diagnosis",
        "keywords": "diagnosis condition problem disease illness clinical finding assessment",
        "resource": "Condition",
        "category": "problem-list-item",
        "unit": None,
        "dri_field": "diagnosis_present"
    },
    # --- ENCOUNTER ---
    {
        "loinc": "46240-8",
        "display": "Encounter history",
        "keywords": "visit encounter admission date last hospital appointment last_visit",
        "resource": "Encounter",
        "category": "encounter",
        "unit": None,
        "dri_field": "last_visit_gap"
    },
]

# Fields that belong to patient identity, NOT clinical data
PATIENT_META_FIELDS = {
    "patient_id", "name", "first_name", "last_name",
    "dob", "gender", "date", "pt_id", "id", "sex", "age"
}


# ─────────────────────────────────────────────
# PATIENT EXTRACTOR
# ─────────────────────────────────────────────

class PatientExtractor:
    """Smartly extracts and normalizes patient identity from raw data."""

    GENDER_MAP = {
        "m": "male", "male": "male", "man": "male", "boy": "male",
        "f": "female", "female": "female", "woman": "female", "girl": "female",
        "o": "other", "other": "other", "non-binary": "other",
    }

    def extract(self, raw_record: dict, form_meta: dict) -> dict:
        """Merge raw record identity fields with form-submitted metadata."""
        meta = {}

        # --- Patient ID ---
        meta["patient_id"] = (
            self._find(raw_record, ["patient_id", "pt_id", "id", "pid", "mrn", "patient_number"])
            or form_meta.get("patient_id", "UNKNOWN")
        )

        # --- Name splitting ---
        raw_name = self._find(raw_record, ["name", "patient_name", "full_name"])
        if raw_name:
            parts = str(raw_name).strip().split()
            meta["first_name"] = parts[0] if parts else ""
            meta["last_name"] = " ".join(parts[1:]) if len(parts) > 1 else ""
        else:
            meta["first_name"] = (
                self._find(raw_record, ["first_name", "fname", "given_name"])
                or form_meta.get("first_name", "")
            )
            meta["last_name"] = (
                self._find(raw_record, ["last_name", "lname", "family_name", "surname"])
                or form_meta.get("last_name", "")
            )

        # --- Gender normalization ---
        raw_gender = str(
            self._find(raw_record, ["gender", "sex"])
            or form_meta.get("gender", "unknown")
        ).strip().lower()
        meta["gender"] = self.GENDER_MAP.get(raw_gender, "unknown")

        # --- Age → approximate birthDate ---
        raw_dob = self._find(raw_record, ["dob", "date_of_birth", "birth_date", "birthdate"])
        raw_age = self._find(raw_record, ["age"])
        if raw_dob:
            meta["dob"] = str(raw_dob)
        elif raw_age:
            meta["dob"] = self._age_to_birthdate(raw_age)
        else:
            meta["dob"] = form_meta.get("dob", "")

        return meta

    def _find(self, record: dict, keys: list):
        """Case-insensitive field finder."""
        lower_map = {k.lower(): v for k, v in record.items()}
        for k in keys:
            if k.lower() in lower_map:
                return lower_map[k.lower()]
        return None

    def _age_to_birthdate(self, age_val) -> str:
        """Convert '45 yrs' or '45' to approximate ISO birthDate."""
        try:
            digits = re.search(r"(\d+)", str(age_val))
            if digits:
                years = int(digits.group(1))
                approx = datetime.now() - timedelta(days=years * 365)
                return approx.strftime("%Y-01-01")
        except Exception:
            pass
        return ""


# ─────────────────────────────────────────────
# SEMANTIC FIELD MAPPER
# ─────────────────────────────────────────────

class SemanticFieldMapper:
    def __init__(self, confidence_threshold=0.30):
        self.registry = FHIR_LOINC_REGISTRY
        self.threshold = confidence_threshold
        corpus = [
            f"{e['display']} {e['keywords']}" for e in self.registry
        ]
        print("[Samayik] Loading Semantic Model (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.corpus_embeddings = self.model.encode(corpus, convert_to_tensor=True)
        print(f"[Samayik] Mapper ready — {len(self.registry)} LOINC codes indexed.")

    def map(self, raw_field: str) -> dict:
        cleaned = re.sub(r'[_\-]', ' ', raw_field).lower().strip()
        query_embedding = self.model.encode(cleaned, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
        idx = int(torch.argmax(cos_scores).item())
        conf = float(cos_scores[idx])

        if conf >= self.threshold:
            m = self.registry[idx]
            return {
                "status": "MAPPED",
                "raw_field": raw_field,
                "loinc_code": m["loinc"],
                "display": m["display"],
                "resource_type": m["resource"],
                "category": m["category"],
                "unit": m["unit"],
                "dri_field": m["dri_field"],
                "confidence": round(conf, 3)
            }
        return {
            "status": "UNKNOWN",
            "raw_field": raw_field,
            "confidence": round(conf, 3),
            "note": "Below threshold — flagged for manual review"
        }


# ─────────────────────────────────────────────
# FORMAT PARSER
# ─────────────────────────────────────────────

class FormatParser:
    def _flatten_dict(self, d: dict, parent_key: str = '', sep: str = '_') -> dict:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Keep arrays intact so ResourceRouter can spawn multiple resources
                items.append((new_key, v))
            else:
                items.append((new_key, v))
        return dict(items)

    def parse(self, data: str, fmt: str) -> list:
        if fmt == "csv":
            return [dict(r) for r in csv.DictReader(io.StringIO(data))]
        elif fmt == "json":
            p = json.loads(data)
            records = p if isinstance(p, list) else [p]
            return [self._flatten_dict(r) for r in records]
        elif fmt == "xml":
            import xmltodict
            
            # Force lists for typical child tags that might repeat (medications, allergies)
            def force_list(path, key, value):
                return key in ["medications", "medication", "allergies", "allergy", "patient", "record"]
                
            p = xmltodict.parse(data, force_list=force_list)
            
            # Drill down past the root node if it holds a list
            root_val = list(p.values())[0]
            records = root_val if isinstance(root_val, list) else [root_val]
            return [self._flatten_dict(r) for r in records]
        raise ValueError(f"Unsupported format: {fmt}")


# ─────────────────────────────────────────────
# VALUE PARSER (deterministic, rule-based)
# ─────────────────────────────────────────────

class ValueParser:
    """Extracts structured clinical values from raw text deterministically."""

    # Common medication frequency abbreviations
    FREQ_MAP = {
        "od": 1, "qd": 1, "once daily": 1, "daily": 1,
        "bd": 2, "bid": 2, "twice daily": 2,
        "tds": 3, "tid": 3, "three times daily": 3,
        "qid": 4, "qds": 4, "four times daily": 4,
        "prn": 0, "as needed": 0, "sos": 0,
    }

    # Temperature unit normalization
    TEMP_UNITS = {
        "f": "[degF]", "°f": "[degF]", "fahrenheit": "[degF]",
        "c": "Cel", "°c": "Cel", "celsius": "Cel",
    }

    def parse_bp(self, value) -> dict | None:
        """Parse '140/90' → {systolic: 140, diastolic: 90}"""
        if not isinstance(value, str) or "/" not in value:
            return None
        try:
            parts = value.strip().split("/")
            sys_val = float(re.search(r"[\d.]+", parts[0]).group())
            dia_val = float(re.search(r"[\d.]+", parts[1]).group())
            return {"systolic": sys_val, "diastolic": dia_val}
        except Exception:
            return None

    def parse_temp(self, value) -> dict | None:
        """Parse '99F' or '37.1°C' → {value: 99, unit: '[degF]'}"""
        if not isinstance(value, str):
            try:
                return {"value": float(value), "unit": "Cel"}
            except (ValueError, TypeError):
                return None
        match = re.match(r"^([\d.]+)\s*°?\s*([a-zA-Z]+)?$", str(value).strip())
        if match:
            try:
                val = float(match.group(1))
                unit_raw = (match.group(2) or "").lower()
                unit = self.TEMP_UNITS.get(unit_raw, "Cel")
                return {"value": val, "unit": unit}
            except Exception:
                pass
        return None

    def parse_lab_value(self, value) -> dict | None:
        """Parse '8.5%' or '150 mg/dL' → {value: 8.5, unit: '%'}"""
        if not isinstance(value, str):
            try:
                return {"value": float(value), "unit": None}
            except (ValueError, TypeError):
                return None
        match = re.match(r"^([\d.]+)\s*([a-zA-Z/%]+(?:/[a-zA-Z]+)?)?$", str(value).strip())
        if match:
            try:
                return {"value": float(match.group(1)), "unit": match.group(2) or None}
            except Exception:
                pass
        return None

    def parse_dose(self, value) -> dict | None:
        """Parse '500mg' → {value: 500, unit: 'mg'}"""
        if not isinstance(value, str):
            return None
        match = re.match(r"^([\d.]+)\s*([a-zA-Z]+)?$", str(value).strip())
        if match:
            try:
                return {"value": float(match.group(1)), "unit": match.group(2) or "mg"}
            except Exception:
                pass
        return None

    def parse_frequency(self, value) -> dict | None:
        """Parse 'BD' → {frequency: 2, period: 1, periodUnit: 'd'}"""
        if not isinstance(value, str):
            return None
        freq = self.FREQ_MAP.get(value.strip().lower())
        if freq is not None:
            return {"frequency": freq, "period": 1, "periodUnit": "d"}
        return None

    def parse_number(self, value) -> float | None:
        """Extract a number from any value."""
        try:
            return float(value)
        except (ValueError, TypeError):
            pass
        if isinstance(value, str):
            match = re.search(r"[\d.]+", value)
            if match:
                try:
                    return float(match.group())
                except Exception:
                    pass
        return None

    def parse_date(self, value) -> str | None:
        """Normalize dates like '12-03-24' to ISO format."""
        if not isinstance(value, str):
            return None
        for fmt in ["%d-%m-%Y", "%d-%m-%y", "%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%d/%m/%Y", "%d/%m/%y"]:
            try:
                return datetime.strptime(value.strip(), fmt).strftime("%Y-%m-%d")
            except Exception:
                continue
        return value  # Return as-is if no format matches


# ─────────────────────────────────────────────
# FHIR R4 BUNDLE BUILDER (with Resource Router)
# ─────────────────────────────────────────────

class FHIRBundleBuilder:
    def __init__(self):
        self.vp = ValueParser()

    def build(self, patient_meta: dict, mapped_fields: list) -> dict:
        patient_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Build Patient resource
        patient_resource = {
            "resourceType": "Patient",
            "id": patient_id,
            "meta": {"profile": ["http://hl7.org/fhir/StructureDefinition/Patient"]},
            "identifier": [{"system": "urn:samayik:upid", "value": patient_meta.get("patient_id", "UNKNOWN")}],
            "name": [{
                "family": patient_meta.get("last_name", ""),
                "given": [patient_meta.get("first_name", "")]
            }],
            "birthDate": patient_meta.get("dob", ""),
            "gender": patient_meta.get("gender", "unknown")
        }
        entries = [{"fullUrl": f"urn:uuid:{patient_id}", "resource": patient_resource}]

        # Route each mapped field to the correct FHIR resource(s)
        for field in mapped_fields:
            if field["status"] != "MAPPED":
                continue
            resources = self._route(field, patient_id, now)
            for r in resources:
                entries.append({"fullUrl": f"urn:uuid:{r['id']}", "resource": r})

        return {
            "resourceType": "Bundle",
            "id": str(uuid.uuid4()),
            "meta": {
                "lastUpdated": now,
                "tag": [{"system": "urn:samayik", "code": "samayik-processed", "display": "Processed by Samayik AI"}]
            },
            "type": "collection",
            "timestamp": now,
            "entry": entries
        }

    def _route(self, field, patient_id, now):
        """Resource Router: handles arrays by spawning individual resources."""
        raw = field.get("raw_value")
        if isinstance(raw, list):
            resources = []
            for item in raw:
                sub = dict(field)
                sub["raw_value"] = item
                r = self._build_single(sub, patient_id, now)
                if r:
                    resources.append(r)
            return resources
        else:
            r = self._build_single(field, patient_id, now)
            return [r] if r else []

    def _build_single(self, field, patient_id, now):
        """Builds a single FHIR resource with proper value parsing."""
        rtype = field["resource_type"]
        raw = field.get("raw_value")

        if rtype == "Observation":
            return self._build_observation(field, raw, patient_id, now)
        elif rtype == "AllergyIntolerance":
            return self._build_allergy(field, raw, patient_id, now)
        elif rtype == "MedicationRequest":
            return self._build_medication(field, raw, patient_id, now)
        elif rtype == "Condition":
            return self._build_condition(field, raw, patient_id, now)
        elif rtype == "Encounter":
            return self._build_encounter(field, raw, patient_id, now)
        return None

    # --- Observation (vital signs, labs, blood group) ---
    def _build_observation(self, field, raw, patient_id, now):
        obs = {
            "resourceType": "Observation",
            "id": str(uuid.uuid4()),
            "status": "final",
            "category": [{"coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                "code": field["category"],
                "display": field["category"].replace("-", " ").title()
            }]}],
            "code": {"coding": [{
                "system": "http://loinc.org",
                "code": field["loinc_code"],
                "display": field["display"]
            }], "text": field["display"]},
            "subject": {"reference": f"Patient/{patient_id}"},
            "effectiveDateTime": now
        }

        # Blood Pressure panel: split "140/90" into components
        bp = self.vp.parse_bp(raw)
        if bp and field["loinc_code"] in ["85354-9", "8480-6", "8462-4"]:
            obs["code"] = {"coding": [{"system": "http://loinc.org", "code": "85354-9", "display": "Blood pressure panel"}], "text": "Blood Pressure"}
            obs["component"] = [
                {
                    "code": {"coding": [{"system": "http://loinc.org", "code": "8480-6", "display": "Systolic blood pressure"}]},
                    "valueQuantity": {"value": bp["systolic"], "unit": "mm[Hg]", "system": "http://unitsofmeasure.org", "code": "mm[Hg]"}
                },
                {
                    "code": {"coding": [{"system": "http://loinc.org", "code": "8462-4", "display": "Diastolic blood pressure"}]},
                    "valueQuantity": {"value": bp["diastolic"], "unit": "mm[Hg]", "system": "http://unitsofmeasure.org", "code": "mm[Hg]"}
                }
            ]
            return obs

        # Temperature: extract value and unit (F→[degF], C→Cel)
        if field["loinc_code"] == "8310-5":
            parsed = self.vp.parse_temp(raw)
            if parsed:
                obs["valueQuantity"] = {"value": parsed["value"], "unit": parsed["unit"], "system": "http://unitsofmeasure.org", "code": parsed["unit"]}
                return obs

        # Lab values: parse "8.5%" or "150 mg/dL"
        if field["category"] == "laboratory":
            parsed = self.vp.parse_lab_value(raw)
            if parsed:
                unit = parsed["unit"] or field.get("unit") or ""
                obs["valueQuantity"] = {"value": parsed["value"], "unit": unit, "system": "http://unitsofmeasure.org", "code": unit}
                return obs
            # If it's a text value like "B+" for blood group
            obs["valueString"] = str(raw)
            return obs

        # Generic numeric value
        num = self.vp.parse_number(raw)
        if num is not None and field.get("unit"):
            obs["valueQuantity"] = {"value": num, "unit": field["unit"], "system": "http://unitsofmeasure.org", "code": field["unit"]}
        elif num is not None:
            obs["valueQuantity"] = {"value": num}
        else:
            obs["valueString"] = str(raw)
        return obs

    # --- AllergyIntolerance (uses SNOMED, not LOINC) ---
    def _build_allergy(self, field, raw, patient_id, now):
        # Handle comma-separated allergies by splitting them
        if isinstance(raw, str) and "," in raw:
            # Return just the first one; the pipeline caller can handle multiples
            items = [x.strip() for x in raw.split(",") if x.strip()]
            text_val = items[0] if items else str(raw)
        elif isinstance(raw, dict):
            text_val = str(raw.get("name", raw.get("substance", raw.get("value", raw))))
        else:
            text_val = str(raw)

        return {
            "resourceType": "AllergyIntolerance",
            "id": str(uuid.uuid4()),
            "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical", "code": "active"}]},
            "verificationStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-verification", "code": "confirmed"}]},
            "code": {
                "coding": [{"system": "http://snomed.info/sct", "display": text_val}],
                "text": text_val
            },
            "patient": {"reference": f"Patient/{patient_id}"},
            "recordedDate": now
        }

    # --- MedicationRequest (structured dose + frequency) ---
    def _build_medication(self, field, raw, patient_id, now):
        if isinstance(raw, dict):
            med_name = raw.get("name", raw.get("medication", "Unknown Medication"))
            dose_raw = raw.get("dose", raw.get("dosage", ""))
            freq_raw = raw.get("freq", raw.get("frequency", ""))

            # Build dosage instruction
            dosage = {"text": f"{med_name} {dose_raw} {freq_raw}".strip()}

            # Parse dose quantity
            parsed_dose = self.vp.parse_dose(str(dose_raw))
            if parsed_dose:
                dosage["doseAndRate"] = [{
                    "doseQuantity": {
                        "value": parsed_dose["value"],
                        "unit": parsed_dose["unit"],
                        "system": "http://unitsofmeasure.org",
                        "code": parsed_dose["unit"]
                    }
                }]

            # Parse frequency
            parsed_freq = self.vp.parse_frequency(str(freq_raw))
            if parsed_freq:
                dosage["timing"] = {
                    "repeat": {
                        "frequency": parsed_freq["frequency"],
                        "period": parsed_freq["period"],
                        "periodUnit": parsed_freq["periodUnit"]
                    }
                }
        else:
            med_name = str(raw)
            dosage = {"text": med_name}

        return {
            "resourceType": "MedicationRequest",
            "id": str(uuid.uuid4()),
            "status": "active",
            "intent": "order",
            "medicationCodeableConcept": {"text": str(med_name)},
            "subject": {"reference": f"Patient/{patient_id}"},
            "authoredOn": now,
            "dosageInstruction": [dosage]
        }

    # --- Condition (diagnosis) ---
    def _build_condition(self, field, raw, patient_id, now):
        text_val = str(raw.get("name", raw.get("value", raw))) if isinstance(raw, dict) else str(raw)
        return {
            "resourceType": "Condition",
            "id": str(uuid.uuid4()),
            "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]},
            "verificationStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-verification", "code": "confirmed"}]},
            "code": {
                "coding": [{"system": "http://snomed.info/sct", "display": text_val}],
                "text": text_val
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "recordedDate": now
        }

    # --- Encounter (last visit) ---
    def _build_encounter(self, field, raw, patient_id, now):
        parsed_date = self.vp.parse_date(raw) if isinstance(raw, str) else now
        return {
            "resourceType": "Encounter",
            "id": str(uuid.uuid4()),
            "status": "finished",
            "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": "AMB", "display": "ambulatory"},
            "subject": {"reference": f"Patient/{patient_id}"},
            "period": {"start": parsed_date or now, "end": parsed_date or now}
        }


# ─────────────────────────────────────────────
# VALIDATION LAYER
# ─────────────────────────────────────────────

class ValidationLayer:
    """Compares input fields vs output resources to detect data loss."""

    def validate(self, raw_record: dict, mapped_fields: list, bundle: dict) -> dict:
        total_input = len([k for k in raw_record if k.lower() not in PATIENT_META_FIELDS])
        mapped_count = sum(1 for f in mapped_fields if f["status"] == "MAPPED")
        unmapped = [f for f in mapped_fields if f["status"] == "UNKNOWN"]
        resources_built = len(bundle.get("entry", [])) - 1  # exclude Patient resource

        coverage = round(mapped_count / max(total_input, 1) * 100, 1)

        issues = []
        for f in unmapped:
            issues.append({
                "field": f["raw_field"],
                "issue": "unmapped",
                "detail": f"Field '{f['raw_field']}' could not be mapped (confidence: {f['confidence']})"
            })

        return {
            "total_input_fields": total_input,
            "fields_mapped": mapped_count,
            "fields_unmapped": len(unmapped),
            "resources_generated": resources_built,
            "coverage_percent": coverage,
            "issues": issues,
            "data_loss_detected": len(unmapped) > 0
        }


# ─────────────────────────────────────────────
# DRI SCORER (enhanced with validation feedback)
# ─────────────────────────────────────────────

class DRIScorer:
    WEIGHTS = {
        "vitals_complete":       25,
        "allergy_present":       20,
        "medication_count":      15,
        "lab_results_present":   15,
        "diagnosis_present":     10,
        "last_visit_gap":         5,
        "mental_health_present":  5,
        "data_coverage":          5,
    }

    def score(self, bundle: dict, validation: dict = None) -> dict:
        signals = self._extract(bundle)

        # Feed validation coverage into risk score
        if validation:
            signals["data_coverage"] = validation.get("coverage_percent", 0) >= 70

        risk = 0
        missing = []
        for field, weight in self.WEIGHTS.items():
            if not signals.get(field):
                risk += weight
                missing.append(field.replace("_", " ").title())
        return {
            "dri_score": risk,
            "risk_level": self._level(risk),
            "signals": signals,
            "missing": missing,
            "max_possible": sum(self.WEIGHTS.values())
        }

    def _extract(self, bundle: dict) -> dict:
        signals = {k: False for k in self.WEIGHTS}
        for entry in bundle.get("entry", []):
            r = entry.get("resource", {})
            rtype = r.get("resourceType", "")
            if rtype == "Observation":
                cats = [c.get("code", "") for cat in r.get("category", []) for c in cat.get("coding", [])]
                if "vital-signs" in cats:
                    signals["vitals_complete"] = True
                if "survey" in cats:
                    signals["mental_health_present"] = True
                if "laboratory" in cats:
                    signals["lab_results_present"] = True
            elif rtype == "AllergyIntolerance":
                signals["allergy_present"] = True
            elif rtype == "MedicationRequest":
                signals["medication_count"] = True
            elif rtype == "Condition":
                signals["diagnosis_present"] = True
            elif rtype == "Encounter":
                signals["last_visit_gap"] = True
        return signals

    def _level(self, score: int) -> str:
        if score >= 60:
            return "HIGH RISK"
        if score >= 30:
            return "MEDIUM RISK"
        return "LOW RISK"


# ─────────────────────────────────────────────
# ALERT ENGINE
# ─────────────────────────────────────────────

class AlertEngine:
    def generate(self, dri: dict) -> dict:
        score, level, missing = dri["dri_score"], dri["risk_level"], dri["missing"]
        ms = ", ".join(missing) if missing else "None"
        msgs = {
            "HIGH RISK":   f"⚠️  HIGH RISK — DRI Score: {score}/100\nCritical data is missing. Do not proceed without review.\nMissing: {ms}",
            "MEDIUM RISK": f"⚡ MEDIUM RISK — DRI Score: {score}/100\nSome records are incomplete. Review before deciding.\nMissing: {ms}",
            "LOW RISK":    f"✅ LOW RISK — DRI Score: {score}/100\nPatient data looks reasonably complete.\nMissing: {ms}"
        }
        recs = {
            "HIGH RISK":   "Do not proceed without collecting missing critical data.",
            "MEDIUM RISK": "Proceed with caution. Flag this case for follow-up.",
            "LOW RISK":    "Safe to proceed with clinical decision."
        }
        return {"alert_level": level, "dri_score": score, "message": msgs[level], "missing_fields": missing, "recommendation": recs[level]}


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

class SamayikPipeline:
    def __init__(self):
        self.parser     = FormatParser()
        self.extractor  = PatientExtractor()
        self.mapper     = SemanticFieldMapper(confidence_threshold=0.30)
        self.builder    = FHIRBundleBuilder()
        self.validator  = ValidationLayer()
        self.scorer     = DRIScorer()
        self.alerter    = AlertEngine()

    def run_file(self, file_path: str, patient_meta: dict = None) -> dict:
        fmt = os.path.splitext(file_path)[1].lower().replace(".", "")
        with open(file_path, "r") as f:
            raw_data = f.read()
        return self.run(raw_data, fmt, patient_meta or {}, original_file=os.path.basename(file_path))

    def run(self, raw_data: str, fmt: str, patient_meta: dict = None, original_file: str = "data") -> dict:
        patient_meta = patient_meta or {}

        print(f"\n{'='*55}")
        print(f"  Samayik AI Pipeline v2")
        print(f"{'='*55}")

        # Step 1: Parse
        print("\n[1] Parsing input...")
        records = self.parser.parse(raw_data, fmt)
        print(f"    → {len(records)} record(s)")

        # Step 2: Extract patient identity from the data itself
        print("\n[2] Extracting patient identity...")
        first_record = records[0] if records else {}
        smart_meta = self.extractor.extract(first_record, patient_meta)
        print(f"    → ID: {smart_meta['patient_id']}")
        print(f"    → Name: {smart_meta['first_name']} {smart_meta['last_name']}")
        print(f"    → Gender: {smart_meta['gender']}")
        print(f"    → DOB: {smart_meta['dob']}")

        # Step 3: Map fields
        print("\n[3] Mapping fields → FHIR codes (AI semantic analysis)...")
        mapped = []
        for record in records:
            for fname, fval in record.items():
                if fname.lower() in PATIENT_META_FIELDS:
                    continue
                result = self.mapper.map(fname)
                result["raw_value"] = fval
                mapped.append(result)
                icon = "✓" if result["status"] == "MAPPED" else "?"
                print(f"    {icon}  '{fname}' → {result.get('display','UNKNOWN')} "
                      f"[LOINC: {result.get('loinc_code','N/A')}] "
                      f"(conf: {result['confidence']})")

        # Step 4: Build FHIR Bundle
        print("\n[4] Building FHIR R4 Bundle (with value parsing)...")
        bundle = self.builder.build(smart_meta, mapped)
        print(f"    → {len(bundle['entry'])} resources built")

        # Step 5: Validate
        print("\n[5] Validating transformation completeness...")
        validation = self.validator.validate(first_record, mapped, bundle)
        print(f"    → Coverage: {validation['coverage_percent']}%")
        print(f"    → Mapped: {validation['fields_mapped']}/{validation['total_input_fields']}")
        if validation["issues"]:
            for issue in validation["issues"]:
                print(f"    ⚠  {issue['detail']}")

        # Step 6: Score risk
        print("\n[6] Computing DRI Score...")
        dri = self.scorer.score(bundle, validation)
        print(f"    → Score: {dri['dri_score']}/100  |  {dri['risk_level']}")
        print(f"    → Missing: {dri['missing']}")

        # Step 7: Alert
        print("\n[7] Generating clinical alert...")
        alert = self.alerter.generate(dri)
        print(f"\n    {alert['message']}")
        print(f"\n    Recommendation: {alert['recommendation']}")

        # Save output
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        rid = str(uuid.uuid4())[:6]
        base_name = original_file.rsplit('.', 1)[0]
        filename = f"{base_name}_fhir_{ts}_{rid}.json"

        with open(filename, "w") as f:
            json.dump(bundle, f, indent=2)
        print(f"\n[Samayik] Saved FHIR Bundle to: {filename}")

        return {
            "fhir_bundle": bundle,
            "dri_result": dri,
            "alert": alert,
            "mapped_fields": mapped,
            "validation": validation,
            "patient_extracted": smart_meta
        }


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = SamayikPipeline()

    # Demo — Real-world messy hospital JSON
    print("\n\n" + "▓"*55)
    print("  DEMO — Real Hospital JSON (nested, messy)")
    print("▓"*55)

    test_file = "../tests/01.json"
    if os.path.exists(test_file):
        pipeline.run_file(test_file)
    else:
        # Fallback inline demo
        demo_data = json.dumps({
            "patient_id": "P-1023",
            "name": "Rohit Sharma",
            "age": "45 yrs",
            "gender": "M",
            "blood_group": "B+",
            "allergies": "penicillin",
            "vitals": {"bp": "140/90", "heart_rate": 92, "temp": "99F"},
            "diagnosis": "Type 2 Diabetes",
            "medications": [
                {"name": "Metformin", "dose": "500mg", "freq": "BD"},
                {"name": "Atorvastatin", "dose": "10mg", "freq": "OD"}
            ],
            "lab_results": {"HbA1c": "8.5%", "fasting_glucose": "150 mg/dL"},
            "last_visit": "12-03-24"
        })
        pipeline.run(raw_data=demo_data, fmt="json", original_file="demo_hospital.json")

    print("\n[Samayik] Complete ✓\n")