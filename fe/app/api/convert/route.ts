/**
 * Samayik AI — FHIR R4 Conversion API
 * =====================================
 * POST /api/convert
 *
 * Accepts: multipart/form-data with a file (CSV, JSON, or SQL)
 *          + optional patient metadata fields
 *
 * Returns: { fhir_bundle, dri_result, alert, mapped_fields }
 *
 * Pipeline:
 *   1. Format Parser     → reads CSV / JSON / SQL
 *   2. Semantic Mapper   → TF-IDF cosine similarity → LOINC codes
 *   3. FHIR R4 Builder   → valid FHIR R4 Bundle
 *   4. DRI Scorer        → Decision Risk Index
 *   5. Alert Engine      → clinician-facing alert
 */

// ─────────────────────────────────────────────
// FHIR R4 LOINC CODE REGISTRY
// Source: HL7 FHIR R4 Vital Signs Profile
// ─────────────────────────────────────────────

interface LoincEntry {
  loinc: string;
  display: string;
  keywords: string;
  resource: string;
  category: string;
  unit: string | null;
  dri_field: string;
}

const FHIR_LOINC_REGISTRY: LoincEntry[] = [
  {
    loinc: "8480-6",
    display: "Systolic blood pressure",
    keywords: "systolic blood pressure bp sys mmhg arterial",
    resource: "Observation",
    category: "vital-signs",
    unit: "mm[Hg]",
    dri_field: "vitals_complete",
  },
  {
    loinc: "8462-4",
    display: "Diastolic blood pressure",
    keywords: "diastolic blood pressure bp dia mmhg",
    resource: "Observation",
    category: "vital-signs",
    unit: "mm[Hg]",
    dri_field: "vitals_complete",
  },
  {
    loinc: "8867-4",
    display: "Heart rate",
    keywords: "heart rate pulse beat bpm cardiac rhythm",
    resource: "Observation",
    category: "vital-signs",
    unit: "/min",
    dri_field: "vitals_complete",
  },
  {
    loinc: "9279-1",
    display: "Respiratory rate",
    keywords: "respiratory rate breathing breath respiration rr",
    resource: "Observation",
    category: "vital-signs",
    unit: "/min",
    dri_field: "vitals_complete",
  },
  {
    loinc: "8310-5",
    display: "Body temperature",
    keywords: "temperature temp body celsius fahrenheit fever",
    resource: "Observation",
    category: "vital-signs",
    unit: "Cel",
    dri_field: "vitals_complete",
  },
  {
    loinc: "59408-5",
    display: "Oxygen saturation",
    keywords: "oxygen saturation spo2 o2 pulse oximetry sat level",
    resource: "Observation",
    category: "vital-signs",
    unit: "%",
    dri_field: "vitals_complete",
  },
  {
    loinc: "29463-7",
    display: "Body weight",
    keywords: "weight body mass kg kilogram bmi",
    resource: "Observation",
    category: "vital-signs",
    unit: "kg",
    dri_field: "vitals_complete",
  },
  {
    loinc: "8302-2",
    display: "Body height",
    keywords: "height body length cm centimeter tall stature",
    resource: "Observation",
    category: "vital-signs",
    unit: "cm",
    dri_field: "vitals_complete",
  },
  {
    loinc: "52473-6",
    display: "Allergy to substance",
    keywords: "allergy allergies allergic reaction substance drug food info",
    resource: "AllergyIntolerance",
    category: "allergy",
    unit: null,
    dri_field: "allergy_present",
  },
  {
    loinc: "57828-6",
    display: "Prescription medication list",
    keywords:
      "medication medicine drug prescription tablet capsule dose prescribed drugs meds",
    resource: "MedicationRequest",
    category: "medication",
    unit: null,
    dri_field: "medication_count",
  },
  {
    loinc: "44249-1",
    display: "Mental health assessment",
    keywords:
      "mental health psychology psychiatric depression anxiety phq score assessment",
    resource: "Observation",
    category: "survey",
    unit: null,
    dri_field: "mental_health_present",
  },
  {
    loinc: "58410-2",
    display: "Complete blood count panel",
    keywords:
      "blood count lab CBC haemoglobin hemoglobin rbc wbc platelet laboratory",
    resource: "DiagnosticReport",
    category: "laboratory",
    unit: null,
    dri_field: "lab_results_present",
  },
  {
    loinc: "24323-8",
    display: "Comprehensive metabolic panel",
    keywords:
      "metabolic panel glucose creatinine sodium potassium liver kidney lab",
    resource: "DiagnosticReport",
    category: "laboratory",
    unit: null,
    dri_field: "lab_results_present",
  },
  {
    loinc: "46240-8",
    display: "Encounter history",
    keywords: "visit encounter admission date last hospital appointment",
    resource: "Encounter",
    category: "encounter",
    unit: null,
    dri_field: "last_visit_gap",
  },
];

const PATIENT_META_FIELDS = new Set([
  "patient_id",
  "name",
  "first_name",
  "last_name",
  "dob",
  "gender",
  "date",
  "pt_id",
  "id",
]);

// ─────────────────────────────────────────────
// TF-IDF VECTORIZER (pure TypeScript)
// ─────────────────────────────────────────────

function tokenize(text: string): string[] {
  const cleaned = text
    .toLowerCase()
    .replace(/[_\-]/g, " ")
    .trim();
  const words = cleaned.split(/\s+/).filter((w) => w.length > 0);
  const tokens: string[] = [...words];
  for (let i = 0; i < words.length - 1; i++) {
    tokens.push(`${words[i]} ${words[i + 1]}`);
  }
  return tokens;
}

class TfidfVectorizer {
  private vocabulary: Map<string, number> = new Map();
  private idf: number[] = [];
  private corpusVectors: number[][] = [];

  fit(corpus: string[]): void {
    const docCount = corpus.length;
    const tokenizedDocs = corpus.map((doc) => tokenize(doc));

    // Build vocabulary from all tokens
    const allTokens = new Set<string>();
    for (const tokens of tokenizedDocs) {
      for (const token of tokens) {
        allTokens.add(token);
      }
    }
    let idx = 0;
    for (const token of allTokens) {
      this.vocabulary.set(token, idx++);
    }

    // Compute IDF (smooth variant matching sklearn)
    const df = new Array(this.vocabulary.size).fill(0);
    for (const tokens of tokenizedDocs) {
      const unique = new Set(tokens);
      for (const token of unique) {
        const vi = this.vocabulary.get(token);
        if (vi !== undefined) df[vi]++;
      }
    }
    this.idf = df.map((d) => Math.log((docCount + 1) / (d + 1)) + 1);

    // Pre-compute corpus vectors
    this.corpusVectors = tokenizedDocs.map((tokens) =>
      this.vectorize(tokens)
    );
  }

  private vectorize(tokens: string[]): number[] {
    const vec = new Array(this.vocabulary.size).fill(0);
    const counts = new Map<string, number>();
    for (const t of tokens) {
      counts.set(t, (counts.get(t) || 0) + 1);
    }
    for (const [token, count] of counts) {
      const vi = this.vocabulary.get(token);
      if (vi !== undefined) {
        // sublinear TF: 1 + log(tf)
        vec[vi] = (1 + Math.log(count)) * this.idf[vi];
      }
    }
    // L2 normalize
    const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
    if (norm > 0) for (let i = 0; i < vec.length; i++) vec[i] /= norm;
    return vec;
  }

  findBestMatch(query: string): { index: number; similarity: number } {
    const qVec = this.vectorize(tokenize(query));
    let bestIdx = 0;
    let bestSim = -1;
    for (let i = 0; i < this.corpusVectors.length; i++) {
      let dot = 0;
      for (let j = 0; j < qVec.length; j++) {
        dot += qVec[j] * this.corpusVectors[i][j];
      }
      if (dot > bestSim) {
        bestSim = dot;
        bestIdx = i;
      }
    }
    return { index: bestIdx, similarity: bestSim };
  }
}

// ─────────────────────────────────────────────
// SEMANTIC FIELD MAPPER
// ─────────────────────────────────────────────

interface MappedField {
  status: "MAPPED" | "UNKNOWN";
  raw_field: string;
  raw_value: string;
  loinc_code?: string;
  display?: string;
  resource_type?: string;
  category?: string;
  unit?: string | null;
  dri_field?: string;
  confidence: number;
  note?: string;
}

const CONFIDENCE_THRESHOLD = 0.1;

// Initialize vectorizer once at module level
const vectorizer = new TfidfVectorizer();
const corpus = FHIR_LOINC_REGISTRY.map(
  (e) => `${e.display} ${e.keywords}`
);
vectorizer.fit(corpus);

function mapField(rawField: string): Omit<MappedField, "raw_value"> {
  const cleaned = rawField
    .replace(/[_\-]/g, " ")
    .toLowerCase()
    .trim();
  const { index, similarity } = vectorizer.findBestMatch(cleaned);
  const conf = Math.round(similarity * 1000) / 1000;

  if (conf >= CONFIDENCE_THRESHOLD) {
    const m = FHIR_LOINC_REGISTRY[index];
    return {
      status: "MAPPED",
      raw_field: rawField,
      loinc_code: m.loinc,
      display: m.display,
      resource_type: m.resource,
      category: m.category,
      unit: m.unit,
      dri_field: m.dri_field,
      confidence: conf,
    };
  }
  return {
    status: "UNKNOWN",
    raw_field: rawField,
    confidence: conf,
    note: "Below threshold — flagged for manual review",
  };
}

// ─────────────────────────────────────────────
// FORMAT PARSERS
// ─────────────────────────────────────────────

function parseCSVLine(line: string): string[] {
  const result: string[] = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (ch === "," && !inQuotes) {
      result.push(current);
      current = "";
    } else {
      current += ch;
    }
  }
  result.push(current);
  return result;
}

function parseCSV(text: string): Record<string, string>[] {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return [];
  const headers = parseCSVLine(lines[0]);
  const records: Record<string, string>[] = [];
  for (let i = 1; i < lines.length; i++) {
    if (!lines[i].trim()) continue;
    const values = parseCSVLine(lines[i]);
    const record: Record<string, string> = {};
    headers.forEach((h, idx) => {
      record[h.trim()] = (values[idx] ?? "").trim();
    });
    records.push(record);
  }
  return records;
}

function parseJSON(text: string): Record<string, string>[] {
  const parsed = JSON.parse(text);
  const arr = Array.isArray(parsed) ? parsed : [parsed];
  return arr.map((obj: Record<string, unknown>) => {
    const record: Record<string, string> = {};
    for (const [k, v] of Object.entries(obj)) {
      record[k] = v == null ? "" : String(v);
    }
    return record;
  });
}

function parseSQLValues(valuesStr: string): string[] {
  const values: string[] = [];
  let current = "";
  let inQuotes = false;
  let quoteChar = "";

  for (let i = 0; i < valuesStr.length; i++) {
    const ch = valuesStr[i];
    if (!inQuotes && (ch === "'" || ch === '"')) {
      inQuotes = true;
      quoteChar = ch;
    } else if (inQuotes && ch === quoteChar) {
      if (valuesStr[i + 1] === quoteChar) {
        current += ch;
        i++;
      } else {
        inQuotes = false;
      }
    } else if (ch === "," && !inQuotes) {
      values.push(current.trim());
      current = "";
    } else {
      current += ch;
    }
  }
  values.push(current.trim());
  return values.map((v) => {
    if (v.toUpperCase() === "NULL") return "";
    return v;
  });
}

function parseSQL(text: string): Record<string, string>[] {
  const records: Record<string, string>[] = [];
  const insertRegex =
    /INSERT\s+INTO\s+\w+\s*\(([^)]+)\)\s*VALUES\s*([\s\S]*?)(?:;|$)/gi;

  let match;
  while ((match = insertRegex.exec(text)) !== null) {
    const columns = match[1]
      .split(",")
      .map((c) => c.trim().replace(/[`"'[\]]/g, ""));
    const valuesBlock = match[2];

    const valueGroupRegex = /\(([^)]*)\)/g;
    let vMatch;
    while ((vMatch = valueGroupRegex.exec(valuesBlock)) !== null) {
      const values = parseSQLValues(vMatch[1]);
      const record: Record<string, string> = {};
      columns.forEach((col, idx) => {
        record[col] = values[idx] ?? "";
      });
      records.push(record);
    }
  }
  return records;
}

function detectFormat(filename: string, content: string): "csv" | "json" | "sql" {
  const ext = filename.split(".").pop()?.toLowerCase();
  if (ext === "csv") return "csv";
  if (ext === "json") return "json";
  if (ext === "sql") return "sql";

  // Auto-detect from content
  const trimmed = content.trim();
  if (trimmed.startsWith("{") || trimmed.startsWith("[")) return "json";
  if (/INSERT\s+INTO/i.test(trimmed)) return "sql";
  return "csv";
}

function parseData(
  content: string,
  format: "csv" | "json" | "sql"
): Record<string, string>[] {
  switch (format) {
    case "csv":
      return parseCSV(content);
    case "json":
      return parseJSON(content);
    case "sql":
      return parseSQL(content);
  }
}

// ─────────────────────────────────────────────
// FHIR R4 BUNDLE BUILDER
// ─────────────────────────────────────────────

function generateUUID(): string {
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

interface PatientMeta {
  patient_id: string;
  first_name: string;
  last_name: string;
  dob: string;
  gender: string;
}

/* eslint-disable @typescript-eslint/no-explicit-any */

function buildResource(
  field: MappedField,
  patientId: string,
  now: string
): Record<string, any> | null {
  const rtype = field.resource_type;

  if (rtype === "Observation") {
    const obs: Record<string, any> = {
      resourceType: "Observation",
      id: generateUUID(),
      status: "final",
      category: [
        {
          coding: [
            {
              system:
                "http://terminology.hl7.org/CodeSystem/observation-category",
              code: field.category,
              display: (field.category || "")
                .replace(/-/g, " ")
                .replace(/\b\w/g, (c) => c.toUpperCase()),
            },
          ],
        },
      ],
      code: {
        coding: [
          {
            system: "http://loinc.org",
            code: field.loinc_code,
            display: field.display,
          },
        ],
        text: field.display,
      },
      subject: { reference: `Patient/${patientId}` },
      effectiveDateTime: now,
    };

    if (field.raw_value && field.unit) {
      const num = parseFloat(field.raw_value);
      if (!isNaN(num)) {
        obs.valueQuantity = {
          value: num,
          unit: field.unit,
          system: "http://unitsofmeasure.org",
          code: field.unit,
        };
      } else {
        obs.valueString = field.raw_value;
      }
    } else if (field.raw_value) {
      obs.valueString = field.raw_value;
    }
    return obs;
  }

  if (rtype === "AllergyIntolerance") {
    return {
      resourceType: "AllergyIntolerance",
      id: generateUUID(),
      clinicalStatus: {
        coding: [
          {
            system:
              "http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical",
            code: "active",
          },
        ],
      },
      code: {
        coding: [
          {
            system: "http://loinc.org",
            code: field.loinc_code,
            display: field.display,
          },
        ],
        text: field.raw_value || "",
      },
      patient: { reference: `Patient/${patientId}` },
      recordedDate: now,
    };
  }

  if (rtype === "MedicationRequest") {
    return {
      resourceType: "MedicationRequest",
      id: generateUUID(),
      status: "active",
      intent: "order",
      medicationCodeableConcept: { text: field.raw_value || "" },
      subject: { reference: `Patient/${patientId}` },
      authoredOn: now,
    };
  }

  if (rtype === "DiagnosticReport") {
    return {
      resourceType: "DiagnosticReport",
      id: generateUUID(),
      status: "final",
      code: {
        coding: [
          {
            system: "http://loinc.org",
            code: field.loinc_code,
            display: field.display,
          },
        ],
      },
      subject: { reference: `Patient/${patientId}` },
      effectiveDateTime: now,
    };
  }

  return null;
}

function buildFHIRBundle(
  patientMeta: PatientMeta,
  mappedFields: MappedField[]
): Record<string, any> {
  const patientId = generateUUID();
  const now = new Date().toISOString();

  const entries: Record<string, any>[] = [
    {
      fullUrl: `urn:uuid:${patientId}`,
      resource: {
        resourceType: "Patient",
        id: patientId,
        meta: {
          profile: ["http://hl7.org/fhir/StructureDefinition/Patient"],
        },
        identifier: [
          {
            system: "urn:samayik:upid",
            value: patientMeta.patient_id || "UNKNOWN",
          },
        ],
        name: [
          {
            family: patientMeta.last_name || "",
            given: [patientMeta.first_name || ""],
          },
        ],
        birthDate: patientMeta.dob || "",
        gender: (patientMeta.gender || "unknown").toLowerCase(),
      },
    },
  ];

  for (const field of mappedFields) {
    if (field.status !== "MAPPED") continue;
    const resource = buildResource(field, patientId, now);
    if (resource) {
      entries.push({
        fullUrl: `urn:uuid:${generateUUID()}`,
        resource,
      });
    }
  }

  return {
    resourceType: "Bundle",
    id: generateUUID(),
    meta: {
      lastUpdated: now,
      tag: [
        {
          system: "urn:samayik",
          code: "samayik-processed",
          display: "Processed by Samayik AI",
        },
      ],
    },
    type: "collection",
    timestamp: now,
    entry: entries,
  };
}

/* eslint-enable @typescript-eslint/no-explicit-any */

// ─────────────────────────────────────────────
// DRI SCORER
// ─────────────────────────────────────────────

const DRI_WEIGHTS: Record<string, number> = {
  vitals_complete: 30,
  allergy_present: 25,
  medication_count: 20,
  lab_results_present: 15,
  mental_health_present: 10,
};

interface DRIResult {
  dri_score: number;
  risk_level: string;
  signals: Record<string, boolean>;
  missing: string[];
  max_possible: number;
}

function scoreDRI(bundle: Record<string, unknown>): DRIResult {
  const signals: Record<string, boolean> = {};
  for (const k of Object.keys(DRI_WEIGHTS)) signals[k] = false;

  const entries = (bundle.entry as Array<Record<string, unknown>>) || [];
  for (const entry of entries) {
    const r = (entry.resource as Record<string, unknown>) || {};
    const rtype = r.resourceType as string;

    if (rtype === "Observation") {
      const cats = (
        (r.category as Array<Record<string, unknown>>) || []
      ).flatMap((cat) =>
        ((cat.coding as Array<Record<string, string>>) || []).map(
          (c) => c.code
        )
      );
      if (cats.includes("vital-signs")) signals.vitals_complete = true;
      if (cats.includes("survey")) signals.mental_health_present = true;
      if (cats.includes("laboratory")) signals.lab_results_present = true;
    } else if (rtype === "AllergyIntolerance") {
      signals.allergy_present = true;
    } else if (rtype === "MedicationRequest") {
      signals.medication_count = true;
    } else if (rtype === "DiagnosticReport") {
      signals.lab_results_present = true;
    }
  }

  let risk = 0;
  const missing: string[] = [];
  for (const [field, weight] of Object.entries(DRI_WEIGHTS)) {
    if (!signals[field]) {
      risk += weight;
      missing.push(
        field
          .replace(/_/g, " ")
          .replace(/\b\w/g, (c) => c.toUpperCase())
      );
    }
  }

  let riskLevel: string;
  if (risk >= 60) riskLevel = "HIGH RISK";
  else if (risk >= 30) riskLevel = "MEDIUM RISK";
  else riskLevel = "LOW RISK";

  return {
    dri_score: risk,
    risk_level: riskLevel,
    signals,
    missing,
    max_possible: Object.values(DRI_WEIGHTS).reduce((a, b) => a + b, 0),
  };
}

// ─────────────────────────────────────────────
// ALERT ENGINE
// ─────────────────────────────────────────────

interface AlertResult {
  alert_level: string;
  dri_score: number;
  message: string;
  missing_fields: string[];
  recommendation: string;
}

function generateAlert(dri: DRIResult): AlertResult {
  const { dri_score: score, risk_level: level, missing } = dri;
  const ms = missing.length > 0 ? missing.join(", ") : "None";

  const msgs: Record<string, string> = {
    "HIGH RISK": `⚠️  HIGH RISK — DRI Score: ${score}/100\nCritical data is missing. Do not proceed without review.\nMissing: ${ms}`,
    "MEDIUM RISK": `⚡ MEDIUM RISK — DRI Score: ${score}/100\nSome records are incomplete. Review before deciding.\nMissing: ${ms}`,
    "LOW RISK": `✅ LOW RISK — DRI Score: ${score}/100\nPatient data looks reasonably complete.\nMissing: ${ms}`,
  };

  const recs: Record<string, string> = {
    "HIGH RISK":
      "Do not proceed without collecting missing critical data.",
    "MEDIUM RISK":
      "Proceed with caution. Flag this case for follow-up.",
    "LOW RISK": "Safe to proceed with clinical decision.",
  };

  return {
    alert_level: level,
    dri_score: score,
    message: msgs[level],
    missing_fields: missing,
    recommendation: recs[level],
  };
}

// ─────────────────────────────────────────────
// MAIN PIPELINE
// ─────────────────────────────────────────────

function runPipeline(
  content: string,
  filename: string,
  patientMeta: PatientMeta
) {
  // 1. Detect format & parse
  const format = detectFormat(filename, content);
  const records = parseData(content, format);

  if (records.length === 0) {
    throw new Error(
      "No records found. Check your file format and content."
    );
  }

  // 2. Map fields → LOINC
  const mappedFields: MappedField[] = [];
  for (const record of records) {
    for (const [fieldName, fieldValue] of Object.entries(record)) {
      if (PATIENT_META_FIELDS.has(fieldName.toLowerCase())) continue;
      const mapping = mapField(fieldName);
      mappedFields.push({ ...mapping, raw_value: fieldValue });
    }
  }

  // 3. Build FHIR Bundle
  const bundle = buildFHIRBundle(patientMeta, mappedFields);

  // 4. Score DRI
  const dri = scoreDRI(bundle);

  // 5. Generate alert
  const alert = generateAlert(dri);

  return {
    success: true,
    format_detected: format,
    records_parsed: records.length,
    fhir_bundle: bundle,
    dri_result: dri,
    alert,
    mapped_fields: mappedFields,
  };
}

// ─────────────────────────────────────────────
// ROUTE HANDLER
// ─────────────────────────────────────────────

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const file = formData.get("file") as File | null;

    if (!file) {
      return Response.json(
        { success: false, error: "No file provided." },
        { status: 400 }
      );
    }

    const content = await file.text();
    const filename = file.name || "data.csv";

    const patientMeta: PatientMeta = {
      patient_id: (formData.get("patient_id") as string) || "UNKNOWN",
      first_name: (formData.get("first_name") as string) || "",
      last_name: (formData.get("last_name") as string) || "",
      dob: (formData.get("dob") as string) || "",
      gender: (formData.get("gender") as string) || "unknown",
    };

    const result = runPipeline(content, filename, patientMeta);

    return Response.json(result);
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return Response.json(
      { success: false, error: message },
      { status: 500 }
    );
  }
}
