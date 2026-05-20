"""
Samayik AI — Complete Pipeline v3
====================================
Layer 1:  Format Parser            (CSV / JSON / SQL / XML input)
Layer 2:  Patient Extractor        (strict identity extraction from patient_info)
Layer 3:  Intermediate Normalizer  (raw EMR → clean normalized schema)
Layer 4:  Semantic Field Mapper    (sentence-transformers NLP)
Layer 5:  Schema Guard             (prevents misclassification)
Layer 6:  Value Parser             (deterministic rule-based parsing)
Layer 7:  Resource Router          (correct FHIR resource dispatch)
Layer 8:  FHIR R4 Bundle Builder   (clinically valid FHIR R4 output)
Layer 9:  Validation Layer         (input vs output completeness check)
Layer 10: DRI Scorer               (Decision Risk Index → Quality Score)
Layer 11: Alert Engine             (clinician-facing alert)

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
    # --- LAB RESULTS ---
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
        "keywords": "metabolic panel creatinine sodium potassium liver kidney lab results serum",
        "resource": "Observation",
        "category": "laboratory",
        "unit": None,
        "dri_field": "lab_results_present"
    },
    {
        "loinc": "718-7",
        "display": "Hemoglobin",
        "keywords": "hemoglobin haemoglobin hgb hb blood lab",
        "resource": "Observation",
        "category": "laboratory",
        "unit": "g/dL",
        "dri_field": "lab_results_present"
    },
    {
        "loinc": "6690-2",
        "display": "White blood cell count",
        "keywords": "white blood cell wbc leukocyte count lab",
        "resource": "Observation",
        "category": "laboratory",
        "unit": "/uL",
        "dri_field": "lab_results_present"
    },
    {
        "loinc": "777-3",
        "display": "Platelet count",
        "keywords": "platelet plt thrombocyte count lab",
        "resource": "Observation",
        "category": "laboratory",
        "unit": "/uL",
        "dri_field": "lab_results_present"
    },
    {
        "loinc": "787-2",
        "display": "Mean corpuscular volume",
        "keywords": "mcv mean corpuscular volume red blood cell size lab",
        "resource": "Observation",
        "category": "laboratory",
        "unit": "fL",
        "dri_field": "lab_results_present"
    },
    {
        "loinc": "2498-4",
        "display": "Serum iron",
        "keywords": "serum iron fe iron level lab",
        "resource": "Observation",
        "category": "laboratory",
        "unit": "ug/dL",
        "dri_field": "lab_results_present"
    },
    {
        "loinc": "2276-4",
        "display": "Ferritin",
        "keywords": "ferritin iron storage lab",
        "resource": "Observation",
        "category": "laboratory",
        "unit": "ng/mL",
        "dri_field": "lab_results_present"
    },
    {
        "loinc": "2160-0",
        "display": "Serum creatinine",
        "keywords": "creatinine serum renal kidney function lab",
        "resource": "Observation",
        "category": "laboratory",
        "unit": "mg/dL",
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
    # --- SYMPTOMS ---
    {
        "loinc": "75325-1",
        "display": "Symptom",
        "keywords": "symptom symptoms complaint presenting chief complaint sign",
        "resource": "Observation",
        "category": "exam",
        "unit": None,
        "dri_field": "diagnosis_present"
    },
    # --- CLINICAL NOTES ---
    {
        "loinc": "11506-3",
        "display": "Clinical note",
        "keywords": "note notes clinical impression summary comment remark finding observation text",
        "resource": "ClinicalNote",
        "category": "note",
        "unit": None,
        "dri_field": None
    },
    # --- PRACTITIONER ---
    {
        "loinc": "N/A",
        "display": "Practitioner",
        "keywords": "doctor physician practitioner provider attending consultant dr surgeon specialist",
        "resource": "Practitioner",
        "category": "practitioner",
        "unit": None,
        "dri_field": None
    },
]


# ─────────────────────────────────────────────
# SCHEMA GUARD — prevents misclassification
# ─────────────────────────────────────────────

# Fields that belong to patient identity — NEVER send to semantic mapper
PATIENT_IDENTITY_FIELDS = {
    "patient_id", "name", "first_name", "last_name", "given_name", "family_name",
    "dob", "date_of_birth", "birthdate", "birth_date",
    "gender", "sex",
    "age",
    "pt_id", "id", "pid", "mrn", "patient_number",
    "phone", "telephone", "mobile", "contact",
    "address", "city", "state", "zip", "pincode", "postal_code",
    "email", "email_address",
    "patient_name", "full_name",
    "patient_info",  # whole nested block
}

# Fields that should be explicitly routed by schema, not AI
SCHEMA_FIELD_MAP = {
    # Exact field name → forced FHIR resource type
    "doctor":            "Practitioner",
    "physician":         "Practitioner",
    "attending":         "Practitioner",
    "consultant":        "Practitioner",
    "provider":          "Practitioner",
    "notes":             "ClinicalNote",
    "clinical_notes":    "ClinicalNote",
    "remarks":           "ClinicalNote",
    "comment":           "ClinicalNote",
    "summary":           "ClinicalNote",
    "symptoms":          "Symptom",
    "complaints":        "Symptom",
    "chief_complaint":   "Symptom",
    "diagnosis":         "Condition",
    "diagnoses":         "Condition",
    "visit_details":     "Encounter",
    "visit":             "Encounter",
    "last_visit":        "Encounter",
    "encounter":         "Encounter",
}


# ─────────────────────────────────────────────
# PATIENT EXTRACTOR (Fix #1: strict extraction)
# ─────────────────────────────────────────────

class PatientExtractor:
    """Strictly extracts patient identity from known sections only."""

    GENDER_MAP = {
        "m": "male", "male": "male", "man": "male", "boy": "male",
        "f": "female", "female": "female", "woman": "female", "girl": "female",
        "o": "other", "other": "other", "non-binary": "other",
    }

    def extract(self, raw_record: dict, form_meta: dict) -> dict:
        """
        Extract patient identity from:
          1. A nested 'patient_info' block (highest priority)
          2. Top-level identity fields
          3. Form-submitted metadata (fallback)
        """
        # Check for nested patient_info block first (Fix #1)
        patient_block = None
        for key in ["patient_info", "patient_details", "patient_data", "demographics"]:
            if key in raw_record and isinstance(raw_record[key], dict):
                patient_block = raw_record[key]
                break

        # Merge: patient_block overrides top-level, top-level overrides form
        sources = [form_meta, raw_record, patient_block] if patient_block else [form_meta, raw_record]

        meta = {}

        # --- Patient ID ---
        meta["patient_id"] = self._find_cascade(
            sources, ["patient_id", "pt_id", "id", "pid", "mrn", "patient_number"]
        ) or "UNKNOWN"

        # --- Name splitting ---
        raw_name = self._find_cascade(sources, ["name", "patient_name", "full_name"])
        if raw_name:
            parts = str(raw_name).strip().split()
            meta["first_name"] = parts[0] if parts else ""
            meta["last_name"] = " ".join(parts[1:]) if len(parts) > 1 else ""
        else:
            meta["first_name"] = (
                self._find_cascade(sources, ["first_name", "fname", "given_name"])
                or ""
            )
            meta["last_name"] = (
                self._find_cascade(sources, ["last_name", "lname", "family_name", "surname"])
                or ""
            )

        # --- Gender normalization ---
        raw_gender = str(
            self._find_cascade(sources, ["gender", "sex"]) or "unknown"
        ).strip().lower()
        meta["gender"] = self.GENDER_MAP.get(raw_gender, "unknown")

        # --- Age → approximate birthDate ---
        raw_dob = self._find_cascade(sources, ["dob", "date_of_birth", "birth_date", "birthdate"])
        raw_age = self._find_cascade(sources, ["age"])
        if raw_dob:
            meta["dob"] = str(raw_dob)
        elif raw_age:
            meta["dob"] = self._age_to_birthdate(raw_age)
        else:
            meta["dob"] = ""

        # --- Contact (non-clinical, stored in Patient resource) ---
        meta["phone"] = str(self._find_cascade(sources, ["phone", "telephone", "mobile", "contact"]) or "")
        meta["address"] = str(self._find_cascade(sources, ["address"]) or "")

        return meta

    def _find_cascade(self, sources: list, keys: list):
        """Search multiple source dicts in priority order (last wins)."""
        found = None
        for source in sources:
            if not source:
                continue
            lower_map = {k.lower(): v for k, v in source.items()}
            for k in keys:
                if k.lower() in lower_map:
                    val = lower_map[k.lower()]
                    if val is not None and str(val).strip():
                        found = val
        return found

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
        
        # --- ENTERPRISE LOCAL CACHE LOGIC ---
        import os
        from pathlib import Path
        
        # We store the model inside the project for 100% portability
        base_dir = Path(__file__).parent
        model_path = base_dir / "model_cache" / "all-MiniLM-L6-v2"
        
        if model_path.exists():
            print(f"[Samayik] Loading local brain from: {model_path}")
            # Loading from a local path is ALWAYS 100% offline and fast
            self.model = SentenceTransformer(str(model_path), device='cpu')
        else:
            print("[Samayik] First run: Downloading AI model to local cache...")
            # This only runs ONCE ever to bundle the model into the app
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            model_path.parent.mkdir(exist_ok=True)
            self.model.save(str(model_path))
            print(f"[Samayik] Brain archived to: {model_path}")
            
        self.corpus_embeddings = self.model.encode(corpus, convert_to_tensor=True)
        print(f"[Samayik] Mapper ready — 100% Portable & Offline | {len(self.registry)} LOINC codes indexed.")

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
# INTERMEDIATE NORMALIZER (Fix #6)
# ─────────────────────────────────────────────

class IntermediateNormalizer:
    """
    Cleans, validates, and restructures raw EMR data before FHIR conversion.
    This prevents raw messy data from directly hitting the builder.
    """

    def normalize(self, raw_record: dict) -> dict:
        """Return a clean intermediate schema from raw flattened data."""
        normalized = {}
        for key, value in raw_record.items():
            clean_key = key.strip()

            # Strip whitespace from string values
            if isinstance(value, str):
                value = value.strip()
                # Skip empty strings
                if not value:
                    continue

            # Validate date-like fields don't contain non-date junk
            if self._is_date_field(clean_key) and isinstance(value, str):
                validated = self._validate_date(value)
                if validated:
                    normalized[clean_key] = validated
                else:
                    # Not a valid date — still include but flag type
                    normalized[clean_key] = value
                continue

            # Normalize list values — strip inner whitespace
            if isinstance(value, list):
                cleaned_list = []
                for item in value:
                    if isinstance(item, str):
                        cleaned_list.append(item.strip())
                    elif isinstance(item, dict):
                        cleaned_list.append({k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in item.items()})
                    else:
                        cleaned_list.append(item)
                normalized[clean_key] = cleaned_list
                continue

            normalized[clean_key] = value

        return normalized

    def _is_date_field(self, key: str) -> bool:
        date_keywords = ["date", "visit", "admission", "discharge", "dob", "birth"]
        return any(kw in key.lower() for kw in date_keywords)

    def _validate_date(self, value: str) -> str | None:
        """Try to parse a date string; return ISO format or None."""
        for fmt in ["%d-%m-%Y", "%d-%m-%y", "%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%d/%m/%Y", "%d/%m/%y"]:
            try:
                return datetime.strptime(value.strip(), fmt).strftime("%Y-%m-%d")
            except Exception:
                continue
        return None


# ─────────────────────────────────────────────
# VALUE PARSER (deterministic, rule-based)
# ─────────────────────────────────────────────

class ValueParser:
    """Extracts structured clinical values from raw text deterministically."""

    # Common medication frequency abbreviations (Fix #5: SOS handling)
    FREQ_MAP = {
        "od": {"frequency": 1, "period": 1, "periodUnit": "d", "asNeeded": False},
        "qd": {"frequency": 1, "period": 1, "periodUnit": "d", "asNeeded": False},
        "once daily": {"frequency": 1, "period": 1, "periodUnit": "d", "asNeeded": False},
        "daily": {"frequency": 1, "period": 1, "periodUnit": "d", "asNeeded": False},
        "bd": {"frequency": 2, "period": 1, "periodUnit": "d", "asNeeded": False},
        "bid": {"frequency": 2, "period": 1, "periodUnit": "d", "asNeeded": False},
        "twice daily": {"frequency": 2, "period": 1, "periodUnit": "d", "asNeeded": False},
        "tds": {"frequency": 3, "period": 1, "periodUnit": "d", "asNeeded": False},
        "tid": {"frequency": 3, "period": 1, "periodUnit": "d", "asNeeded": False},
        "three times daily": {"frequency": 3, "period": 1, "periodUnit": "d", "asNeeded": False},
        "qid": {"frequency": 4, "period": 1, "periodUnit": "d", "asNeeded": False},
        "qds": {"frequency": 4, "period": 1, "periodUnit": "d", "asNeeded": False},
        "four times daily": {"frequency": 4, "period": 1, "periodUnit": "d", "asNeeded": False},
        # SOS / PRN → asNeededBoolean = true (Fix #5)
        "prn": {"asNeeded": True},
        "as needed": {"asNeeded": True},
        "sos": {"asNeeded": True},
        "when required": {"asNeeded": True},
        "if needed": {"asNeeded": True},
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
        """Parse '99F' or '37.1C' → {value: 99, unit: '[degF]'}"""
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
        """Parse '8.5%' or '150 mg/dL' or '220000 /uL' → {value, unit}"""
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
        """Parse 'BD' → scheduled timing, 'SOS' → asNeededBoolean=true"""
        if not isinstance(value, str):
            return None
        entry = self.FREQ_MAP.get(value.strip().lower())
        if entry:
            return dict(entry)  # return a copy
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

        # Build Patient resource (with contact info from Fix #1)
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

        # Add telecom if available
        phone = patient_meta.get("phone", "")
        if phone:
            patient_resource["telecom"] = [{"system": "phone", "value": phone, "use": "mobile"}]

        # Add address if available
        address = patient_meta.get("address", "")
        if address:
            patient_resource["address"] = [{"text": address, "use": "home"}]

        entries = [{"fullUrl": f"urn:uuid:{patient_id}", "resource": patient_resource}]

        # Fix #3: Track encounter — only ONE per visit
        encounter_id = None
        encounter_built = False

        # Route each mapped field to the correct FHIR resource(s)
        for field in mapped_fields:
            if field["status"] != "MAPPED":
                continue

            rtype = field["resource_type"]

            # Fix #3: Only one Encounter per visit
            if rtype == "Encounter":
                if not encounter_built:
                    enc = self._build_encounter(field, patient_id, now)
                    encounter_id = enc["id"]
                    entries.append({"fullUrl": f"urn:uuid:{enc['id']}", "resource": enc})
                    encounter_built = True
                continue  # Skip duplicate encounters

            resources = self._route(field, patient_id, now)
            for r in resources:
                # Link to encounter if we have one
                if encounter_id and r["resourceType"] in ("Observation", "MedicationRequest", "Condition", "AllergyIntolerance"):
                    r["encounter"] = {"reference": f"Encounter/{encounter_id}"}
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
                    if isinstance(r, list):
                        resources.extend(r)
                    else:
                        resources.append(r)
            return resources
        else:
            r = self._build_single(field, patient_id, now)
            if r is None:
                return []
            return r if isinstance(r, list) else [r]

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
        elif rtype == "Practitioner":
            return self._build_practitioner(field, raw, now)
        elif rtype == "ClinicalNote":
            return self._build_clinical_note(field, raw, patient_id, now)
        elif rtype == "Symptom":
            return self._build_symptom(field, raw, patient_id, now)
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

        # Temperature: extract value and unit (F->[degF], C->Cel)
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

    # --- Symptom → Observation with valueString (Fix #4) ---
    def _build_symptom(self, field, raw, patient_id, now):
        """Symptoms are stored as qualitative Observations, not Conditions."""
        if isinstance(raw, list):
            # Multiple symptoms → multiple Observations
            results = []
            for symptom in raw:
                obs = {
                    "resourceType": "Observation",
                    "id": str(uuid.uuid4()),
                    "status": "final",
                    "category": [{"coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "exam",
                        "display": "Exam"
                    }]}],
                    "code": {"coding": [{
                        "system": "http://loinc.org",
                        "code": "75325-1",
                        "display": "Symptom"
                    }], "text": str(symptom)},
                    "subject": {"reference": f"Patient/{patient_id}"},
                    "effectiveDateTime": now,
                    "valueString": str(symptom)
                }
                results.append(obs)
            return results
        else:
            return {
                "resourceType": "Observation",
                "id": str(uuid.uuid4()),
                "status": "final",
                "category": [{"coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                    "code": "exam",
                    "display": "Exam"
                }]}],
                "code": {"coding": [{
                    "system": "http://loinc.org",
                    "code": "75325-1",
                    "display": "Symptom"
                }], "text": str(raw)},
                "subject": {"reference": f"Patient/{patient_id}"},
                "effectiveDateTime": now,
                "valueString": str(raw)
            }

    # --- AllergyIntolerance (uses SNOMED, not LOINC) ---
    def _build_allergy(self, field, raw, patient_id, now):
        # Handle comma-separated allergies by splitting them
        if isinstance(raw, str) and "," in raw:
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

    # --- MedicationRequest (Fix #5: SOS → asNeededBoolean) ---
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

            # Parse frequency (Fix #5: handle SOS/PRN correctly)
            parsed_freq = self.vp.parse_frequency(str(freq_raw))
            if parsed_freq:
                if parsed_freq.get("asNeeded"):
                    # SOS/PRN: use asNeededBoolean instead of frequency=0
                    dosage["asNeededBoolean"] = True
                else:
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

    # --- Condition (diagnosis) — Fix #2: ONLY actual diagnoses ---
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

    # --- Encounter (Fix #3: ONE per visit, with validated dates) ---
    def _build_encounter(self, field, raw_or_patient_id=None, patient_id_or_now=None, now_maybe=None):
        """Handles both direct call and routed call signatures."""
        # Handle the case when called directly from build() with (field, patient_id, now)
        if isinstance(raw_or_patient_id, str) and now_maybe is not None:
            patient_id = raw_or_patient_id
            now = now_maybe
            raw = field.get("raw_value")
        else:
            raw = field.get("raw_value")
            patient_id = raw_or_patient_id
            now = patient_id_or_now

        enc = {
            "resourceType": "Encounter",
            "id": str(uuid.uuid4()),
            "status": "finished",
            "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": "AMB", "display": "ambulatory"},
            "subject": {"reference": f"Patient/{patient_id}"},
        }

        # Extract date from visit_details dict or raw string
        if isinstance(raw, dict):
            # visit_details: { date: "...", type: "follow-up", department: "..." }
            date_raw = raw.get("date", raw.get("visit_date", ""))
            parsed_date = self.vp.parse_date(str(date_raw)) if date_raw else None

            visit_type = raw.get("type", "")
            dept = raw.get("department", "")

            if parsed_date:
                enc["period"] = {"start": parsed_date, "end": parsed_date}
            else:
                enc["period"] = {"start": now, "end": now}

            # Add visit type as reason
            if visit_type:
                enc["type"] = [{"text": visit_type}]

            # Add department as serviceType
            if dept:
                enc["serviceType"] = {"text": dept}

        elif isinstance(raw, str):
            parsed_date = self.vp.parse_date(raw)
            if parsed_date:
                enc["period"] = {"start": parsed_date, "end": parsed_date}
            else:
                # Fix #5: If it's not a date, don't put it in period
                enc["period"] = {"start": now, "end": now}
        else:
            enc["period"] = {"start": now, "end": now}

        return enc

    # --- Practitioner (Fix #2: doctor → Practitioner, NOT Condition) ---
    def _build_practitioner(self, field, raw, now):
        text_val = str(raw).strip()
        # Strip "Dr." prefix for structured name
        name_clean = re.sub(r"^(Dr\.?\s*)", "", text_val, flags=re.IGNORECASE).strip()
        parts = name_clean.split()

        return {
            "resourceType": "Practitioner",
            "id": str(uuid.uuid4()),
            "name": [{
                "text": text_val,
                "given": [parts[0]] if parts else [],
                "family": " ".join(parts[1:]) if len(parts) > 1 else ""
            }]
        }

    # --- Clinical Note → DocumentReference (Fix #2) ---
    def _build_clinical_note(self, field, raw, patient_id, now):
        text_val = str(raw)
        return {
            "resourceType": "DocumentReference",
            "id": str(uuid.uuid4()),
            "status": "current",
            "type": {"coding": [{"system": "http://loinc.org", "code": "11506-3", "display": "Clinical note"}]},
            "subject": {"reference": f"Patient/{patient_id}"},
            "date": now,
            "content": [{"attachment": {"contentType": "text/plain", "data": text_val}}]
        }


# ─────────────────────────────────────────────
# VALIDATION LAYER
# ─────────────────────────────────────────────

class ValidationLayer:
    """Compares input fields vs output resources to detect data loss."""

    def validate(self, raw_record: dict, mapped_fields: list, bundle: dict) -> dict:
        total_input = len([k for k in raw_record if k.lower() not in PATIENT_IDENTITY_FIELDS])
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
# DRI SCORER (Quality Score — higher is better)
# ─────────────────────────────────────────────

class DRIScorer:
    """
    Granular Data Quality Scorer.
    
    Instead of binary yes/no checks, measures DEPTH of each category:
      - Vitals: how many of the 5 core vitals (BP, HR, Temp, SpO2, RR) are present
      - Labs: how many individual lab observations exist
      - Medications: how many have structured dosage info
      - Allergies: are they listed, and how many
      - Diagnosis: confirmed conditions present
      - Encounter: does it have a valid date
      - Clinical context: practitioner, notes, symptoms present
      - Data coverage: what % of input fields were successfully mapped
    """

    # Maximum points per category
    CATEGORY_WEIGHTS = {
        "vitals":           20,   # 5 core vitals, 4 pts each
        "labs":             18,   # up to 6 labs, 3 pts each (capped at 18)
        "medications":      14,   # up to 4 meds with dose info, 3.5 pts each
        "allergies":        10,   # at least 1 = 5pts, 2+ = 10pts
        "diagnosis":        10,   # at least 1 = 5pts, 2+ = 10pts
        "encounter":         8,   # present=4, has valid date=8
        "clinical_context":  10,  # practitioner=4, notes=3, symptoms=3
        "data_coverage":     10,  # proportional to validation coverage %
    }

    # The 5 core vitals by LOINC code
    CORE_VITALS = {"85354-9", "8480-6", "8462-4", "8867-4", "8310-5", "59408-5", "9279-1"}

    def score(self, bundle: dict, validation: dict = None) -> dict:
        counts = self._count_resources(bundle)
        breakdown = {}
        quality_score = 0

        # --- Vitals: 4 pts per unique vital type, max 20 ---
        unique_vitals = counts["vital_loincs"]
        # BP panel (85354-9) counts as 1 even though it has 2 components
        vital_count = len(unique_vitals)
        vital_pts = min(vital_count * 4, self.CATEGORY_WEIGHTS["vitals"])
        breakdown["vitals"] = {"score": vital_pts, "max": self.CATEGORY_WEIGHTS["vitals"],
                               "detail": f"{vital_count} vital sign(s) recorded"}
        quality_score += vital_pts

        # --- Labs: 3 pts per lab, max 18 ---
        lab_count = counts["lab_count"]
        lab_pts = min(lab_count * 3, self.CATEGORY_WEIGHTS["labs"])
        breakdown["labs"] = {"score": lab_pts, "max": self.CATEGORY_WEIGHTS["labs"],
                             "detail": f"{lab_count} lab result(s)"}
        quality_score += lab_pts

        # --- Medications: 3.5 pts per med with structured dose, max 14 ---
        med_total = counts["med_count"]
        med_structured = counts["med_with_dose"]
        med_pts = min(round(med_structured * 3.5), self.CATEGORY_WEIGHTS["medications"])
        breakdown["medications"] = {"score": med_pts, "max": self.CATEGORY_WEIGHTS["medications"],
                                     "detail": f"{med_structured}/{med_total} medication(s) with dosage"}
        quality_score += med_pts

        # --- Allergies: 1=5pts, 2+=10pts ---
        allergy_count = counts["allergy_count"]
        allergy_pts = 0
        if allergy_count >= 2: allergy_pts = 10
        elif allergy_count == 1: allergy_pts = 5
        breakdown["allergies"] = {"score": allergy_pts, "max": self.CATEGORY_WEIGHTS["allergies"],
                                   "detail": f"{allergy_count} allergy(ies)"}
        quality_score += allergy_pts

        # --- Diagnosis: 1=5pts, 2+=10pts ---
        dx_count = counts["condition_count"]
        dx_pts = 0
        if dx_count >= 2: dx_pts = 10
        elif dx_count == 1: dx_pts = 5
        breakdown["diagnosis"] = {"score": dx_pts, "max": self.CATEGORY_WEIGHTS["diagnosis"],
                                   "detail": f"{dx_count} diagnosis(es)"}
        quality_score += dx_pts

        # --- Encounter: present=4, with valid date=8 ---
        enc_pts = 0
        if counts["encounter_present"]:
            enc_pts = 8 if counts["encounter_has_date"] else 4
        breakdown["encounter"] = {"score": enc_pts, "max": self.CATEGORY_WEIGHTS["encounter"],
                                   "detail": "with date" if counts["encounter_has_date"] else ("present" if counts["encounter_present"] else "missing")}
        quality_score += enc_pts

        # --- Clinical context: practitioner=4, notes=3, symptoms=3 ---
        ctx_pts = 0
        ctx_details = []
        if counts["has_practitioner"]: ctx_pts += 4; ctx_details.append("practitioner")
        if counts["has_notes"]: ctx_pts += 3; ctx_details.append("notes")
        if counts["has_symptoms"]: ctx_pts += 3; ctx_details.append("symptoms")
        breakdown["clinical_context"] = {"score": ctx_pts, "max": self.CATEGORY_WEIGHTS["clinical_context"],
                                          "detail": ", ".join(ctx_details) if ctx_details else "none"}
        quality_score += ctx_pts

        # --- Data coverage: proportional to validation % ---
        coverage_pct = validation.get("coverage_percent", 0) if validation else 0
        coverage_pts = round(coverage_pct / 100 * self.CATEGORY_WEIGHTS["data_coverage"])
        breakdown["data_coverage"] = {"score": coverage_pts, "max": self.CATEGORY_WEIGHTS["data_coverage"],
                                       "detail": f"{coverage_pct}% fields mapped"}
        quality_score += coverage_pts

        max_possible = sum(self.CATEGORY_WEIGHTS.values())
        level = self._level(quality_score)

        # Build signals dict for frontend compatibility
        signals = {
            "vitals_complete": vital_count >= 3,
            "allergy_present": allergy_count > 0,
            "medication_count": med_total > 0,
            "lab_results_present": lab_count > 0,
            "diagnosis_present": dx_count > 0,
            "last_visit_gap": counts["encounter_present"],
            "mental_health_present": counts.get("has_mental_health", False),
            "data_coverage": coverage_pct >= 70,
        }

        # Missing = categories scoring 0
        missing = [cat.replace("_", " ").title() for cat, info in breakdown.items() if info["score"] == 0]

        return {
            "dri_score": quality_score,
            "risk_level": level,
            "signals": signals,
            "missing": missing,
            "max_possible": max_possible,
            "breakdown": breakdown,
        }

    def _count_resources(self, bundle: dict) -> dict:
        """Walk the bundle and count everything granularly."""
        counts = {
            "vital_loincs": set(),
            "lab_count": 0,
            "med_count": 0,
            "med_with_dose": 0,
            "allergy_count": 0,
            "condition_count": 0,
            "encounter_present": False,
            "encounter_has_date": False,
            "has_practitioner": False,
            "has_notes": False,
            "has_symptoms": False,
            "has_mental_health": False,
        }
        for entry in bundle.get("entry", []):
            r = entry.get("resource", {})
            rtype = r.get("resourceType", "")

            if rtype == "Observation":
                cats = [c.get("code", "") for cat in r.get("category", []) for c in cat.get("coding", [])]
                loinc = ""
                for coding in r.get("code", {}).get("coding", []):
                    if coding.get("system") == "http://loinc.org":
                        loinc = coding.get("code", "")

                if "vital-signs" in cats:
                    counts["vital_loincs"].add(loinc)
                elif "laboratory" in cats:
                    counts["lab_count"] += 1
                elif "exam" in cats:
                    counts["has_symptoms"] = True
                elif "survey" in cats:
                    counts["has_mental_health"] = True

            elif rtype == "AllergyIntolerance":
                counts["allergy_count"] += 1

            elif rtype == "MedicationRequest":
                counts["med_count"] += 1
                dosage = (r.get("dosageInstruction") or [{}])[0]
                if dosage.get("doseAndRate") or dosage.get("asNeededBoolean"):
                    counts["med_with_dose"] += 1

            elif rtype == "Condition":
                counts["condition_count"] += 1

            elif rtype == "Encounter":
                counts["encounter_present"] = True
                period = r.get("period", {})
                start = period.get("start", "")
                # Check if the date is an actual calendar date (not an ISO timestamp from "now")
                if start and not start.startswith("20") or (len(start) == 10):
                    counts["encounter_has_date"] = True

            elif rtype == "Practitioner":
                counts["has_practitioner"] = True

            elif rtype == "DocumentReference":
                counts["has_notes"] = True

        return counts

    def _level(self, score: int) -> str:
        if score >= 75: return "EXCELLENT"
        if score >= 45: return "ACCEPTABLE"
        return "INCOMPLETE"


# ─────────────────────────────────────────────
# ALERT ENGINE
# ─────────────────────────────────────────────

class AlertEngine:
    def generate(self, dri: dict) -> dict:
        score, level, missing = dri["dri_score"], dri["risk_level"], dri["missing"]
        ms = ", ".join(missing) if missing else "None"
        msgs = {
            "INCOMPLETE":  f"INCOMPLETE DATA — Quality Score: {score}/100\nCritical data is missing. Review required.\nMissing: {ms}",
            "ACCEPTABLE":  f"ACCEPTABLE DATA — Quality Score: {score}/100\nSome clinical information is absent. Review advised.\nMissing: {ms}",
             "EXCELLENT":  f"EXCELLENT DATA — Quality Score: {score}/100\nPatient data is highly complete.\nMissing: {ms}"
        }
        recs = {
            "INCOMPLETE": "Do not proceed without collecting missing critical data.",
            "ACCEPTABLE": "Proceed with caution. Flag this case for follow-up.",
             "EXCELLENT": "Safe to proceed with clinical integration."
        }
        return {"alert_level": level, "dri_score": score, "message": msgs[level], "missing_fields": missing, "recommendation": recs[level]}


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

class SamayikPipeline:
    def __init__(self):
        self.parser      = FormatParser()
        self.extractor   = PatientExtractor()
        self.normalizer  = IntermediateNormalizer()
        self.mapper      = SemanticFieldMapper(confidence_threshold=0.30)
        self.builder     = FHIRBundleBuilder()
        self.validator   = ValidationLayer()
        self.scorer      = DRIScorer()
        self.alerter     = AlertEngine()

    def run_file(self, file_path: str, patient_meta: dict = None) -> dict:
        fmt = os.path.splitext(file_path)[1].lower().replace(".", "")
        with open(file_path, "r") as f:
            raw_data = f.read()
        return self.run(raw_data, fmt, patient_meta or {}, original_file=os.path.basename(file_path))

    def run(self, raw_data: str, fmt: str, patient_meta: dict = None, original_file: str = "data") -> dict:
        patient_meta = patient_meta or {}

        print(f"\n{'='*55}")
        print(f"  Samayik AI Pipeline v3")
        print(f"{'='*55}")

        # Step 1: Parse
        print("\n[1] Parsing input...")
        records = self.parser.parse(raw_data, fmt)
        print(f"    -> {len(records)} record(s)")

        # Step 2: Extract patient identity from the data itself (Fix #1)
        print("\n[2] Extracting patient identity (strict mode)...")
        # For patient extraction, use the ORIGINAL non-flattened record
        original_record = json.loads(raw_data) if fmt == "json" else records[0] if records else {}
        if isinstance(original_record, list):
            original_record = original_record[0] if original_record else {}
        smart_meta = self.extractor.extract(original_record, patient_meta)
        print(f"    -> ID: {smart_meta['patient_id']}")
        print(f"    -> Name: {smart_meta['first_name']} {smart_meta['last_name']}")
        print(f"    -> Gender: {smart_meta['gender']}")
        print(f"    -> DOB: {smart_meta['dob']}")

        # Step 3: Normalize (Fix #6)
        print("\n[3] Normalizing raw data...")
        first_record = records[0] if records else {}
        normalized = self.normalizer.normalize(first_record)
        print(f"    -> {len(normalized)} clean fields")

        # Step 4: Map fields with Schema Guard (Fix #2)
        print("\n[4] Mapping fields -> FHIR codes (Schema Guard + AI)...")
        mapped = []
        for fname, fval in normalized.items():
            # Skip patient identity fields entirely
            base_name = fname.split("_")[-1].lower() if "_" in fname else fname.lower()
            if fname.lower() in PATIENT_IDENTITY_FIELDS or base_name in PATIENT_IDENTITY_FIELDS:
                continue

            # Fix #2: Check schema guard first (deterministic route)
            schema_route = SCHEMA_FIELD_MAP.get(base_name) or SCHEMA_FIELD_MAP.get(fname.lower())
            if schema_route:
                # Find the matching registry entry for this resource type
                registry_match = next(
                    (e for e in FHIR_LOINC_REGISTRY if e["resource"] == schema_route),
                    None
                )
                if registry_match:
                    result = {
                        "status": "MAPPED",
                        "raw_field": fname,
                        "loinc_code": registry_match["loinc"],
                        "display": registry_match["display"],
                        "resource_type": schema_route,
                        "category": registry_match["category"],
                        "unit": registry_match.get("unit"),
                        "dri_field": registry_match.get("dri_field"),
                        "confidence": 1.0,  # Schema-routed = 100% confidence
                        "raw_value": fval
                    }
                    mapped.append(result)
                    print(f"    [SCHEMA] '{fname}' -> {registry_match['display']} [{schema_route}] (deterministic)")
                    continue

            # Fall through to AI semantic mapping
            result = self.mapper.map(fname)
            result["raw_value"] = fval
            mapped.append(result)
            icon = "+" if result["status"] == "MAPPED" else "?"
            print(f"    {icon}  '{fname}' -> {result.get('display','UNKNOWN')} "
                  f"[LOINC: {result.get('loinc_code','N/A')}] "
                  f"(conf: {result['confidence']})")

        # Step 5: Build FHIR Bundle
        print("\n[5] Building FHIR R4 Bundle (with value parsing)...")
        bundle = self.builder.build(smart_meta, mapped)
        print(f"    -> {len(bundle['entry'])} resources built")

        # Step 6: Validate
        print("\n[6] Validating transformation completeness...")
        validation = self.validator.validate(normalized, mapped, bundle)
        print(f"    -> Coverage: {validation['coverage_percent']}%")
        print(f"    -> Mapped: {validation['fields_mapped']}/{validation['total_input_fields']}")
        if validation["issues"]:
            for issue in validation["issues"]:
                print(f"    !  {issue['detail']}")

        # Step 7: Score
        print("\n[7] Computing Data Quality Score...")
        dri = self.scorer.score(bundle, validation)
        print(f"    -> Score: {dri['dri_score']}/100  |  {dri['risk_level']}")
        print(f"    -> Missing: {dri['missing']}")

        # Step 8: Alert
        print("\n[8] Generating clinical alert...")
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

    # Demo: test all 3 test files
    for test_file in ["../tests/01.json", "../tests/02.json", "../tests/03.json"]:
        if os.path.exists(test_file):
            print("\n\n" + "="*55)
            print(f"  TEST: {test_file}")
            print("="*55)
            pipeline.run_file(test_file)

    print("\n[Samayik] Complete\n")