"use client";

import { useState, useRef, useCallback } from "react";

// ─────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────

interface MappedField {
  status: "MAPPED" | "UNKNOWN";
  raw_field: string;
  /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
  raw_value: any;
  loinc_code?: string;
  display?: string;
  resource_type?: string;
  category?: string;
  unit?: string | null;
  confidence: number;
  note?: string;
}

interface DRIResult {
  dri_score: number;
  risk_level: string;
  signals: Record<string, boolean>;
  missing: string[];
  max_possible: number;
}

interface AlertResult {
  alert_level: string;
  dri_score: number;
  message: string;
  missing_fields: string[];
  recommendation: string;
}

interface ConvertResult {
  success: boolean;
  format_detected: string;
  records_parsed: number;
  fhir_bundle: Record<string, unknown>;
  dri_result: DRIResult;
  alert: AlertResult;
  mapped_fields: MappedField[];
  error?: string;
}

// ─────────────────────────────────────────────
// Main Page
// ─────────────────────────────────────────────

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<ConvertResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showBundle, setShowBundle] = useState(false);

  // Patient metadata
  const [patientId, setPatientId] = useState("");
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [dob, setDob] = useState("");
  const [gender, setGender] = useState("");

  const fileInputRef = useRef<HTMLInputElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);

  // Drag & Drop handlers
  const handleDragOver = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      if (!isDragging) setIsDragging(true);
    },
    [isDragging]
  );

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files?.[0];
    if (droppedFile) {
      setFile(droppedFile);
      setResult(null);
      setError(null);
    }
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setResult(null);
      setError(null);
    }
  };

  // Convert
  const handleConvert = async () => {
    if (!file) return;

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("patient_id", patientId || "UNKNOWN");
      formData.append("first_name", firstName);
      formData.append("last_name", lastName);
      formData.append("dob", dob);
      formData.append("gender", gender);

      const apiBase = process.env.NEXT_PUBLIC_API_URL || "";
      const res = await fetch(`${apiBase}/convert`, {
        method: "POST",
        body: formData,
      });

      const data: ConvertResult = await res.json();

      if (!data.success) {
        setError(data.error || "Conversion failed.");
      } else {
        setResult(data);
        setTimeout(() => {
          resultsRef.current?.scrollIntoView({ behavior: "smooth" });
        }, 100);
      }
    } catch {
      setError("Failed to connect to conversion service.");
    } finally {
      setIsLoading(false);
    }
  };

  // Download
  const handleDownload = () => {
    if (!result?.fhir_bundle) return;
    const blob = new Blob(
      [JSON.stringify(result.fhir_bundle, null, 2)],
      { type: "application/json" }
    );
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `fhir_bundle_${patientId || "patient"}_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const getFileIcon = () => {
    if (!file) return "📁";
    const ext = file.name.split(".").pop()?.toLowerCase();
    if (ext === "csv") return "📊";
    if (ext === "json") return "📋";
    if (ext === "sql") return "🗃️";
    return "📄";
  };

  const getRiskColor = (level: string) => {
    if (level === "HIGH RISK") return "text-danger";
    if (level === "MEDIUM RISK") return "text-warning";
    return "text-success";
  };

  const getRiskBg = (level: string) => {
    if (level === "HIGH RISK") return "bg-danger/10 border-danger/20";
    if (level === "MEDIUM RISK") return "bg-warning/10 border-warning/20";
    return "bg-success/10 border-success/20";
  };

  const getConfidenceColor = (conf: number) => {
    if (conf >= 0.4) return "bg-success";
    if (conf >= 0.2) return "bg-warning";
    return "bg-danger";
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b border-card-border bg-card/50 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-accent to-cyan-600 flex items-center justify-center text-lg font-bold text-black">
              S
            </div>
            <div>
              <h1 className="text-lg font-semibold tracking-tight">
                Samayik AI
              </h1>
              <p className="text-xs text-muted">FHIR R4 Converter</p>
            </div>
          </div>
          <div className="flex gap-2">
            {["CSV", "JSON", "SQL", "XML"].map((fmt) => (
              <span
                key={fmt}
                className="text-[11px] px-2.5 py-1 rounded-full bg-card border border-card-border text-muted font-mono"
              >
                {fmt}
              </span>
            ))}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-6xl mx-auto px-6 py-12 w-full">
        {/* Hero */}
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold tracking-tight mb-4 bg-gradient-to-r from-white via-foreground to-muted bg-clip-text text-transparent">
            Transform Hospital Data to
            <br />
            <span className="bg-gradient-to-r from-accent to-cyan-400 bg-clip-text text-transparent">
              FHIR R4 Standard
            </span>
          </h2>
          <p className="text-muted text-lg max-w-2xl mx-auto">
            Upload messy patient records in any format. Our AI maps fields to
            official LOINC codes and builds compliant FHIR R4 bundles — instantly.
          </p>
        </div>

        {/* Upload Section */}
        <div className="glass-card p-8 mb-8">
          <div
            className={`upload-zone p-12 text-center ${isDragging ? "dragging" : ""}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv,.json,.sql,.txt,.xml"
              onChange={handleFileChange}
              className="hidden"
              id="file-upload"
            />
            {file ? (
              <div className="fade-up">
                <div className="text-4xl mb-3">{getFileIcon()}</div>
                <p className="text-lg font-medium">{file.name}</p>
                <p className="text-sm text-muted mt-1">
                  {(file.size / 1024).toFixed(1)} KB •{" "}
                  {file.name.split(".").pop()?.toUpperCase()}
                </p>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setFile(null);
                    setResult(null);
                  }}
                  className="mt-3 text-sm text-muted hover:text-danger transition-colors"
                >
                  Remove & choose another
                </button>
              </div>
            ) : (
              <div>
                <div className="text-5xl mb-4 opacity-40">⬆️</div>
                <p className="text-lg text-foreground/80 font-medium">
                  Drop your file here or click to browse
                </p>
                <p className="text-sm text-muted mt-2">
                  Supports CSV, JSON, and SQL INSERT dumps
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Patient Metadata */}
        <div className="glass-card p-6 mb-8">
          <h3 className="text-sm font-medium text-muted mb-4 uppercase tracking-wider">
            Patient Information{" "}
            <span className="text-muted/50 normal-case tracking-normal font-normal">
              (optional — enriches the FHIR Bundle)
            </span>
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div>
              <label
                htmlFor="patient-id"
                className="text-xs text-muted mb-1 block"
              >
                Patient ID
              </label>
              <input
                id="patient-id"
                type="text"
                placeholder="P001"
                value={patientId}
                onChange={(e) => setPatientId(e.target.value)}
                className="w-full bg-background border border-card-border rounded-lg px-3 py-2.5 text-sm text-foreground placeholder:text-muted/40 focus:outline-none focus:border-accent/50 transition-colors"
              />
            </div>
            <div>
              <label
                htmlFor="first-name"
                className="text-xs text-muted mb-1 block"
              >
                First Name
              </label>
              <input
                id="first-name"
                type="text"
                placeholder="Raj"
                value={firstName}
                onChange={(e) => setFirstName(e.target.value)}
                className="w-full bg-background border border-card-border rounded-lg px-3 py-2.5 text-sm text-foreground placeholder:text-muted/40 focus:outline-none focus:border-accent/50 transition-colors"
              />
            </div>
            <div>
              <label
                htmlFor="last-name"
                className="text-xs text-muted mb-1 block"
              >
                Last Name
              </label>
              <input
                id="last-name"
                type="text"
                placeholder="Sharma"
                value={lastName}
                onChange={(e) => setLastName(e.target.value)}
                className="w-full bg-background border border-card-border rounded-lg px-3 py-2.5 text-sm text-foreground placeholder:text-muted/40 focus:outline-none focus:border-accent/50 transition-colors"
              />
            </div>
            <div>
              <label htmlFor="dob" className="text-xs text-muted mb-1 block">
                Date of Birth
              </label>
              <input
                id="dob"
                type="date"
                value={dob}
                onChange={(e) => setDob(e.target.value)}
                className="w-full bg-background border border-card-border rounded-lg px-3 py-2.5 text-sm text-foreground placeholder:text-muted/40 focus:outline-none focus:border-accent/50 transition-colors"
              />
            </div>
            <div>
              <label
                htmlFor="gender"
                className="text-xs text-muted mb-1 block"
              >
                Gender
              </label>
              <select
                id="gender"
                value={gender}
                onChange={(e) => setGender(e.target.value)}
                className="w-full bg-background border border-card-border rounded-lg px-3 py-2.5 text-sm text-foreground focus:outline-none focus:border-accent/50 transition-colors"
              >
                <option value="">Select...</option>
                <option value="male">Male</option>
                <option value="female">Female</option>
                <option value="other">Other</option>
              </select>
            </div>
          </div>
        </div>

        {/* Convert Button */}
        <div className="flex justify-center mb-12">
          <button
            id="convert-button"
            onClick={handleConvert}
            disabled={!file || isLoading}
            className={`
              px-8 py-4 rounded-2xl font-semibold text-base transition-all duration-300
              ${
                file && !isLoading
                  ? "bg-gradient-to-r from-accent to-cyan-500 text-black hover:shadow-lg hover:shadow-accent/20 hover:scale-[1.02] active:scale-[0.98] cursor-pointer"
                  : "bg-card border border-card-border text-muted cursor-not-allowed"
              }
            `}
          >
            {isLoading ? (
              <span className="flex items-center gap-3">
                <span className="spinner" />
                Converting...
              </span>
            ) : (
              "🔄 Convert to FHIR R4"
            )}
          </button>
        </div>

        {/* Error */}
        {error && (
          <div className="glass-card p-6 mb-8 border-danger/30 bg-danger/5 fade-up">
            <p className="text-danger font-medium">❌ {error}</p>
          </div>
        )}

        {/* Results */}
        {result && (
          <div ref={resultsRef} className="space-y-6">
            {/* Summary Bar */}
            <div className="glass-card p-6 fade-up">
              <div className="flex flex-wrap gap-6 items-center justify-between">
                <div className="flex items-center gap-6">
                  <div>
                    <p className="text-xs text-muted uppercase tracking-wider">
                      Format
                    </p>
                    <p className="text-lg font-semibold font-mono">
                      {result.format_detected.toUpperCase()}
                    </p>
                  </div>
                  <div className="w-px h-10 bg-card-border" />
                  <div>
                    <p className="text-xs text-muted uppercase tracking-wider">
                      Records
                    </p>
                    <p className="text-lg font-semibold">
                      {result.records_parsed}
                    </p>
                  </div>
                  <div className="w-px h-10 bg-card-border" />
                  <div>
                    <p className="text-xs text-muted uppercase tracking-wider">
                      Fields Mapped
                    </p>
                    <p className="text-lg font-semibold">
                      {
                        result.mapped_fields.filter(
                          (f) => f.status === "MAPPED"
                        ).length
                      }
                      <span className="text-muted font-normal">
                        /{result.mapped_fields.length}
                      </span>
                    </p>
                  </div>
                  <div className="w-px h-10 bg-card-border" />
                  <div>
                    <p className="text-xs text-muted uppercase tracking-wider">
                      FHIR Resources
                    </p>
                    <p className="text-lg font-semibold">
                      {
                        (
                          result.fhir_bundle.entry as Array<unknown>
                        )?.length || 0
                      }
                    </p>
                  </div>
                </div>
                <button
                  onClick={handleDownload}
                  className="px-5 py-2.5 rounded-xl bg-accent/10 border border-accent/20 text-accent text-sm font-medium hover:bg-accent/20 transition-colors cursor-pointer"
                >
                  📥 Download FHIR Bundle
                </button>
              </div>
            </div>

            {/* Field Mapping Table */}
            <div className="glass-card p-6 fade-up fade-up-delay-1">
              <h3 className="text-sm font-medium text-muted uppercase tracking-wider mb-4">
                AI Field Mapping
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-muted text-xs uppercase tracking-wider">
                      <th className="pb-3 pr-4">Raw Field</th>
                      <th className="pb-3 pr-4">FHIR Mapping</th>
                      <th className="pb-3 pr-4">LOINC</th>
                      <th className="pb-3 pr-4">Value</th>
                      <th className="pb-3 pr-4">Confidence</th>
                      <th className="pb-3">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.mapped_fields.map((field, idx) => (
                      <tr
                        key={idx}
                        className="border-t border-card-border/50"
                      >
                        <td className="py-3 pr-4 font-mono text-accent/80">
                          {field.raw_field}
                        </td>
                        <td className="py-3 pr-4">
                          {field.display || (
                            <span className="text-muted italic">
                              Unknown
                            </span>
                          )}
                        </td>
                        <td className="py-3 pr-4 font-mono text-muted">
                          {field.loinc_code || "—"}
                        </td>
                        <td className="py-3 pr-4 text-foreground/70 max-w-[150px] truncate" title={typeof field.raw_value === 'object' ? JSON.stringify(field.raw_value) : String(field.raw_value)}>
                          {typeof field.raw_value === 'object' && field.raw_value !== null 
                            ? JSON.stringify(field.raw_value) 
                            : String(field.raw_value || "—")}
                        </td>
                        <td className="py-3 pr-4 w-36">
                          <div className="flex items-center gap-2">
                            <div className="confidence-bar flex-1">
                              <div
                                className={`confidence-fill ${getConfidenceColor(field.confidence)}`}
                                style={{
                                  width: `${Math.round(field.confidence * 100)}%`,
                                }}
                              />
                            </div>
                            <span className="text-xs text-muted w-10 text-right">
                              {Math.round(field.confidence * 100)}%
                            </span>
                          </div>
                        </td>
                        <td className="py-3">
                          {field.status === "MAPPED" ? (
                            <span className="text-xs px-2 py-1 rounded-full bg-success/10 text-success border border-success/20">
                              ✓ Mapped
                            </span>
                          ) : (
                            <span className="text-xs px-2 py-1 rounded-full bg-warning/10 text-warning border border-warning/20">
                              ? Review
                            </span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Risk Assessment */}
            <div
              className={`glass-card p-6 fade-up fade-up-delay-2 border ${getRiskBg(result.dri_result.risk_level)}`}
            >
              <h3 className="text-sm font-medium text-muted uppercase tracking-wider mb-5">
                Decision Risk Index (DRI)
              </h3>
              <div className="flex flex-col md:flex-row gap-8 items-start">
                {/* Score */}
                <div className="text-center md:text-left flex-shrink-0">
                  <div
                    className={`text-6xl font-bold ${getRiskColor(result.dri_result.risk_level)}`}
                  >
                    {result.dri_result.dri_score}
                    <span className="text-2xl text-muted font-normal">
                      /100
                    </span>
                  </div>
                  <p
                    className={`text-sm font-semibold mt-1 ${getRiskColor(result.dri_result.risk_level)}`}
                  >
                    {result.dri_result.risk_level}
                  </p>
                </div>

                {/* Details */}
                <div className="flex-1 w-full">
                  {/* Gauge */}
                  <div className="mb-5">
                    <div className="risk-gauge-track relative">
                      <div
                        className="risk-gauge-fill absolute top-0 left-0"
                        style={{
                          width: `${result.dri_result.dri_score}%`,
                          opacity: 1,
                        }}
                      />
                    </div>
                    <div className="flex justify-between text-[10px] text-muted mt-1">
                      <span>LOW</span>
                      <span>MEDIUM</span>
                      <span>HIGH</span>
                    </div>
                  </div>

                  {/* Signals */}
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-2 mb-4">
                    {Object.entries(result.dri_result.signals).map(
                      ([key, val]) => (
                        <div
                          key={key}
                          className={`flex items-center gap-2 text-xs px-3 py-2 rounded-lg ${
                            val
                              ? "bg-success/5 text-success/80"
                              : "bg-danger/5 text-danger/80"
                          }`}
                        >
                          <span>{val ? "✓" : "✗"}</span>
                          <span>
                            {key
                              .replace(/_/g, " ")
                              .replace(/\b\w/g, (c) => c.toUpperCase())}
                          </span>
                        </div>
                      )
                    )}
                  </div>

                  {/* Recommendation */}
                  <div className="bg-card/50 rounded-lg p-4 border border-card-border">
                    <p className="text-sm font-medium mb-1">
                      Recommendation
                    </p>
                    <p className="text-sm text-muted">
                      {result.alert.recommendation}
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* FHIR Bundle Viewer */}
            <div className="glass-card p-6 fade-up fade-up-delay-3">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-medium text-muted uppercase tracking-wider">
                  FHIR R4 Bundle
                </h3>
                <button
                  onClick={() => setShowBundle(!showBundle)}
                  className="text-sm text-accent hover:text-accent/80 transition-colors cursor-pointer"
                >
                  {showBundle ? "▼ Hide JSON" : "▶ Show JSON"}
                </button>
              </div>
              {showBundle && (
                <div className="json-viewer p-4 font-mono">
                  <pre className="text-foreground/70 whitespace-pre-wrap break-words">
                    {JSON.stringify(result.fhir_bundle, null, 2)}
                  </pre>
                </div>
              )}
              {!showBundle && (
                <p className="text-sm text-muted">
                  Bundle contains{" "}
                  <span className="text-foreground font-medium">
                    {
                      (result.fhir_bundle.entry as Array<unknown>)
                        ?.length || 0
                    }
                  </span>{" "}
                  FHIR R4 resources. Click &quot;Show JSON&quot; to view the full
                  bundle.
                </p>
              )}
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-card-border py-6 mt-auto">
        <div className="max-w-6xl mx-auto px-6 flex items-center justify-between text-xs text-muted">
          <p>Samayik AI — Built for FHIR R4 compliance</p>
          <p>LOINC codes from HL7 FHIR R4 Vital Signs Profile</p>
        </div>
      </footer>
    </div>
  );
}
