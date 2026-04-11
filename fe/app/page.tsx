"use client";

import { useState, useRef, useCallback, useEffect } from "react";

// ─── Types ──────────────────────────────────
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
}

interface DRIResult {
  dri_score: number;
  risk_level: string;
  signals: Record<string, boolean>;
  missing: string[];
  max_possible: number;
}

interface AlertResult {
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

// ─── Empty State Flow Diagram ────────────────
const PipelineDiagram = () => (
  <div className="h-full flex flex-col items-center justify-center p-6 text-center select-none overflow-y-auto">
    <div className="flex items-center gap-2 mb-2 bg-slate-100 border border-slate-200 px-3 py-1 rounded-full text-[10px] uppercase font-bold text-slate-500 tracking-wider">
      <div className="status-dot"></div> System Idle
    </div>
    
    <h3 className="text-lg font-bold text-slate-800 mb-8 mt-2">Samayik Architecture</h3>
    
    <div className="w-full max-w-[280px] flex flex-col items-center">
      <div className="w-full bg-white border border-slate-200 py-3 rounded-lg text-xs font-semibold text-slate-600 shadow-sm relative z-10">
        Raw Clinical Data
        <div className="text-[10px] text-slate-400 font-normal mt-0.5">CSV, JSON, SQL, XML</div>
      </div>
      
      <div className="flow-line -my-1 relative z-0"></div>
      
      <div className="w-full bg-emerald-50 border border-emerald-200 p-4 rounded-xl shadow-sm relative z-10">
        <div className="absolute -top-2.5 left-1/2 -translate-x-1/2 bg-white px-2 text-[10px] text-emerald-600 font-bold border border-emerald-200 rounded-full uppercase">Engine</div>
        <p className="text-emerald-800 text-sm font-bold">Semantic Mapper</p>
        <p className="text-emerald-600/80 text-[10px] font-mono mt-1">all-MiniLM-L6-v2</p>
      </div>

      <div className="flow-line -my-1 relative z-0"></div>
      
      <div className="w-full flex gap-3 relative z-10">
        <div className="flex-1 bg-blue-50 border border-blue-200 py-3 rounded-lg text-[10px] font-semibold text-blue-800 shadow-sm">
          Schema Guard<br/><span className="font-normal text-blue-600/80">Deterministic</span>
        </div>
        <div className="flex-1 bg-purple-50 border border-purple-200 py-3 rounded-lg text-[10px] font-semibold text-purple-800 shadow-sm">
          Knowledge Graph<br/><span className="font-normal text-purple-600/80">LOINC Embeddings</span>
        </div>
      </div>

      <div className="flow-line -my-1 relative z-0"></div>

      <div className="w-full bg-slate-800 border border-slate-700 py-3 rounded-lg text-xs font-semibold text-slate-200 shadow-sm relative z-10">
        FHIR R4 Generation
      </div>
    </div>
  </div>
);

// ─── Loading State Animation ─────────────────
const ProcessingAnimation = () => {
  const [step, setStep] = useState(0);

  useEffect(() => {
    // Artificial 3.5s sequence
    const t1 = setTimeout(() => setStep(1), 800);
    const t2 = setTimeout(() => setStep(2), 2000);
    const t3 = setTimeout(() => setStep(3), 2900);
    return () => { clearTimeout(t1); clearTimeout(t2); clearTimeout(t3); };
  }, []);

  return (
  <div className="h-full flex flex-col items-center justify-center p-6 text-center select-none overflow-y-auto">
    <div className="flex items-center gap-2 mb-2 bg-emerald-100 border border-emerald-200 px-3 py-1 rounded-full text-[10px] uppercase font-bold text-emerald-700 tracking-wider">
      <div className="status-dot"></div> Processing
    </div>
    
    <h3 className="text-lg font-bold text-slate-800 mb-8 mt-2">Running AI Pipeline</h3>
    
    <div className="w-full max-w-[280px] flex flex-col items-center">
      
      {/* Step 0: Parsing */}
      <div className={`w-full py-3.5 rounded-lg shadow-sm relative z-10 overflow-hidden transition-all duration-300 border-2 ${step === 0 ? 'bg-white border-emerald-400' : step > 0 ? 'bg-emerald-50 border-emerald-200' : 'bg-slate-50 border-slate-200 border'}`}>
        {step === 0 && <div className="absolute inset-0 bg-gradient-to-r from-transparent via-emerald-100/30 to-transparent w-full skeleton-shimmer"></div>}
        <div className={`relative text-xs font-semibold flex items-center justify-center gap-2 ${step >= 0 ? 'text-emerald-700' : 'text-slate-400'}`}>
          {step === 0 ? <svg className="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg> : (step > 0 ? '✓' : '')}
          Parsing Raw Data
        </div>
      </div>
      
      <div className={`flow-line -my-1 relative z-0 transition-opacity duration-300 ${step >= 1 ? 'opacity-100' : 'opacity-30'}`} style={step >= 1 ? { background: 'linear-gradient(to bottom, #10b981 50%, transparent 50%)', backgroundSize: '100% 8px' } : {}}></div>
      
      {/* Step 1: Semantic Mapping */}
      <div className={`w-full py-5 rounded-xl shadow-sm relative z-10 overflow-hidden transition-all duration-300 border-2 ${step === 1 ? 'bg-white border-emerald-400' : step > 1 ? 'bg-emerald-50 border-emerald-200' : 'bg-slate-50 border-slate-200 border'}`}>
        {step === 1 && <div className="absolute inset-0 bg-gradient-to-r from-transparent via-emerald-100/30 to-transparent w-full skeleton-shimmer"></div>}
        <div className={`relative text-sm font-bold flex flex-col items-center justify-center gap-1.5 ${step >= 1 ? 'text-emerald-700' : 'text-slate-400'}`}>
          <div className="flex items-center gap-2">
            {step === 1 ? <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg> : (step > 1 ? '✓' : '')}
            Semantic Mapping
          </div>
          {step >= 1 && <span className="text-[10px] font-mono text-emerald-600/70 font-normal">all-MiniLM-L6-v2</span>}
        </div>
      </div>

      <div className={`flow-line -my-1 relative z-0 transition-opacity duration-300 ${step >= 2 ? 'opacity-100' : 'opacity-30'}`} style={step >= 2 ? { background: 'linear-gradient(to bottom, #10b981 50%, transparent 50%)', backgroundSize: '100% 8px' } : {}}></div>
      
      {/* Step 2: Knowledge Graph + Schema */}
      <div className="w-full flex gap-3 relative z-10">
        <div className={`flex-1 py-4 rounded-lg shadow-sm overflow-hidden relative transition-all duration-300 border-2 ${step === 2 ? 'bg-white border-blue-400' : step > 2 ? 'bg-blue-50 border-blue-200' : 'bg-slate-50 border-slate-200 border'}`}>
          {step === 2 && <div className="absolute inset-0 bg-gradient-to-r from-transparent via-blue-100/30 to-transparent w-full skeleton-shimmer"></div>}
          <div className={`relative text-[10px] font-semibold flex flex-col items-center justify-center gap-1 ${step >= 2 ? 'text-blue-800' : 'text-slate-400'}`}>
            {step === 2 ? <div className="w-2.5 h-2.5 rounded-full border-2 border-blue-500 border-t-transparent animate-spin"></div> : (step > 2 ? '✓' : '')}
            Schema Guard
          </div>
        </div>
        <div className={`flex-1 py-4 rounded-lg shadow-sm overflow-hidden relative transition-all duration-300 border-2 ${step === 2 ? 'bg-white border-purple-400' : step > 2 ? 'bg-purple-50 border-purple-200' : 'bg-slate-50 border-slate-200 border'}`}>
          {step === 2 && <div className="absolute inset-0 bg-gradient-to-r from-transparent via-purple-100/30 to-transparent w-full skeleton-shimmer" style={{animationDelay: '0.2s'}}></div>}
          <div className={`relative text-[10px] font-semibold flex flex-col items-center justify-center gap-1 ${step >= 2 ? 'text-purple-800' : 'text-slate-400'}`}>
            {step === 2 ? <div className="w-2.5 h-2.5 rounded-full border-2 border-purple-500 border-t-transparent animate-spin"></div> : (step > 2 ? '✓' : '')}
            Knowledge Graph
          </div>
        </div>
      </div>

      <div className={`flow-line -my-1 relative z-0 transition-opacity duration-300 ${step >= 3 ? 'opacity-100' : 'opacity-30'}`} style={step >= 3 ? { background: 'linear-gradient(to bottom, #10b981 50%, transparent 50%)', backgroundSize: '100% 8px' } : {}}></div>

      {/* Step 3: FHIR Assembly */}
      <div className={`w-full py-3.5 rounded-lg shadow-sm relative z-10 overflow-hidden transition-all duration-300 border-2 ${step >= 3 ? 'bg-slate-800 border-slate-600' : 'bg-slate-50 border-slate-200 border'}`}>
        {step >= 3 && <div className="absolute inset-0 bg-gradient-to-r from-transparent via-slate-600/50 to-transparent w-full skeleton-shimmer"></div>}
        <div className={`relative text-xs font-semibold flex items-center justify-center gap-2 ${step >= 3 ? 'text-slate-200' : 'text-slate-400'}`}>
          {step >= 3 ? <svg className="w-3 h-3 animate-spin text-slate-300" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg> : ''}
          FHIR R4 Generation
        </div>
      </div>
    </div>
  </div>
  );
};

// ─── Main Application ───────────────────────
export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<ConvertResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  // Form State
  const [patientId, setPatientId] = useState("");
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");

  const fileInputRef = useRef<HTMLInputElement>(null);

  // ─── Handlers ──────────────────────────────
  const handleDragOver = useCallback((e: React.DragEvent) => { e.preventDefault(); if (!isDragging) setIsDragging(true); }, [isDragging]);
  const handleDragLeave = useCallback((e: React.DragEvent) => { e.preventDefault(); setIsDragging(false); }, []);
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); setIsDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) { setFile(f); setResult(null); setError(null); }
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) { setFile(f); setResult(null); setError(null); }
  };

  const handleConvert = async () => {
    if (!file) return;
    setIsLoading(true); setError(null); setResult(null);

    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("patient_id", patientId || "UNKNOWN");
      fd.append("first_name", firstName);
      fd.append("last_name", lastName);

      const base = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const res = await fetch(`${base}/convert`, { method: "POST", body: fd });
      const data = await res.json();
      
      // Artificial delay for the pitch demo to show off the pipeline animation
      await new Promise((resolve) => setTimeout(resolve, 3500));
      
      if (!data.success) setError(data.error || "Failed");
      else setResult(data);
    } catch {
      setError("Failed to connect to Samayik API.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownload = () => {
    if (!result?.fhir_bundle) return;
    const blob = new Blob([JSON.stringify(result.fhir_bundle, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = `bundle.json`; a.click();
    URL.revokeObjectURL(url);
  };

  const handleCopy = () => {
    if (!result?.fhir_bundle) return;
    navigator.clipboard.writeText(JSON.stringify(result.fhir_bundle, null, 2));
    setCopied(true); setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="h-screen w-full flex flex-col bg-slate-50 overflow-hidden text-slate-800">
      
      {/* ─── Header ─── */}
      <header className="h-12 bg-white border-b border-slate-200 flex items-center justify-between px-4 shrink-0">
        <div className="flex items-center gap-2.5">
          <div className="w-6 h-6 rounded bg-emerald-600 text-white flex items-center justify-center font-bold text-xs shadow-sm">S</div>
          <span className="font-bold text-[15px] tracking-tight text-slate-800">Samayik</span>
        </div>
        <div className="flex items-center">
          <a href="https://hl7.org/fhir/R4/" target="_blank" rel="noopener noreferrer" className="text-[11px] font-semibold text-slate-400 hover:text-emerald-600 transition-colors uppercase tracking-wider">
            FHIR R4 Reference ↗
          </a>
        </div>
      </header>

      {/* ─── 3 Column Dashboard ─── */}
      <main className="flex-1 flex gap-4 p-4 overflow-hidden min-h-0">
        
        {/* Col 1: Input */}
        <section className="w-80 flex flex-col shrink-0 panel">
          <div className="panel-header">1. Data Ingestion</div>
          <div className="panel-body flex flex-col gap-5">
            
            <div
              className={`upload-area p-8 flex flex-col items-center justify-center ${isDragging ? 'active' : ''}`}
              onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop} onClick={() => fileInputRef.current?.click()}
            >
              <input ref={fileInputRef} type="file" onChange={handleFileChange} className="hidden" />
              {file ? (
                <>
                  <svg className="w-8 h-8 text-emerald-500 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>
                  <p className="font-semibold text-sm truncate max-w-full px-2 text-slate-700">{file.name}</p>
                  <p className="text-[10px] text-slate-400 mt-0.5">{(file.size/1024).toFixed(1)} KB</p>
                  <button onClick={(e) => { e.stopPropagation(); setFile(null); setResult(null); }} className="mt-2 text-[10px] text-red-500 hover:underline">Remove</button>
                </>
              ) : (
                <>
                  <svg className="w-6 h-6 text-slate-400 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/></svg>
                  <p className="text-xs font-semibold text-slate-600">Drop file to process</p>
                  <p className="text-[10px] text-slate-400 mt-1">Supports CSV, JSON, SQL</p>
                </>
              )}
            </div>

            <div>
              <h4 className="text-[10px] font-bold uppercase text-slate-400 mb-3 tracking-wider">Context Envelope</h4>
              <div className="space-y-3">
                <div><label className="form-label">Patient ID</label><input className="form-input" placeholder="e.g. PX-489" value={patientId} onChange={e=>setPatientId(e.target.value)} /></div>
                <div className="flex gap-2">
                  <div className="flex-1"><label className="form-label">First Name</label><input className="form-input" value={firstName} onChange={e=>setFirstName(e.target.value)} /></div>
                  <div className="flex-1"><label className="form-label">Last Name</label><input className="form-input" value={lastName} onChange={e=>setLastName(e.target.value)} /></div>
                </div>
              </div>
            </div>

            <div className="mt-auto pt-4 border-t border-slate-100 flex flex-col gap-2">
              {error && <div className="text-[10px] p-2 bg-red-50 text-red-600 rounded border border-red-100">{error}</div>}
              <button
                onClick={handleConvert} disabled={!file || isLoading}
                className={`w-full py-2.5 rounded-lg text-sm font-semibold transition-all ${file && !isLoading ? 'bg-emerald-600 text-white hover:bg-emerald-700 shadow-sm' : 'bg-slate-100 text-slate-400'}`}
              >
                {isLoading ? "Running Pipeline..." : "Execute"}
              </button>
            </div>
          </div>
        </section>

        {/* Col 2: Engine Status */}
        <section className="flex-1 flex flex-col min-w-[320px] panel relative">
          <div className="panel-header">
            <span>2. Processing Engine</span>
            {result && <span className="bg-emerald-100 text-emerald-700 px-2 py-0.5 rounded text-[10px] flex items-center gap-1"><div className="w-1.5 h-1.5 bg-emerald-500 rounded-full"></div> Complete</span>}
          </div>
          
          <div className="panel-body p-0 flex flex-col overflow-hidden">
            {isLoading ? (
              <ProcessingAnimation />
            ) : !result ? (
              <PipelineDiagram />
            ) : (
              <div className="flex flex-col h-full"> 
                {/* Score Header */}
                <div className="p-4 border-b border-slate-200 bg-slate-50/50 flex justify-between items-center shrink-0">
                  <div>
                    <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Quality Index</p>
                    <div className="flex items-baseline gap-2 mt-1">
                      <span className={`text-3xl font-extrabold tracking-tight ${result.dri_result.dri_score >= 70 ? 'text-emerald-600' : 'text-amber-500'}`}>{result.dri_result.dri_score}</span>
                      <span className="text-[11px] font-bold uppercase rounded px-1.5 py-0.5 bg-white border border-slate-200 text-slate-600">{result.dri_result.risk_level}</span>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-xs text-slate-500">Processed: <span className="font-bold text-slate-700">{result.records_parsed}</span> record(s)</p>
                    <p className="text-[10px] text-slate-400 font-mono mt-0.5 capitalize">{result.format_detected} format detected</p>
                  </div>
                </div>

                {/* Signals row */}
                <div className="px-4 py-3 border-b border-slate-200 shrink-0 flex flex-wrap gap-2 bg-white">
                  {Object.entries(result.dri_result.signals)
                    .sort(([, valA], [, valB]) => Number(valA) - Number(valB))
                    .map(([key, val]) => (
                    <div key={key} className={`border flex items-center gap-1.5 px-2 py-1 rounded text-[10px] font-medium whitespace-nowrap ${val ? 'bg-emerald-50 text-emerald-700 border-emerald-100' : 'bg-amber-50 text-amber-600 border-amber-200 shadow-sm'}`}>
                      {val ? '✓' : '✕'} {key.replace(/_/g, ' ')}
                    </div>
                  ))}
                </div>
                
                {/* Table */}
                <div className="flex-1 overflow-y-auto">
                  <table className="mapping-table">
                    <thead><tr><th>Raw Field</th><th>Extracted Value</th><th>Mapped Node</th><th className="text-right">Conf.</th></tr></thead>
                    <tbody>
                      {result.mapped_fields.map((f, i) => (
                        <tr key={i}>
                          <td className="font-mono text-[10px] text-emerald-700 bg-emerald-50/30 font-semibold">{f.raw_field}</td>
                          <td className="block max-w-[120px] truncate text-slate-500" title={typeof f.raw_value === 'object' ? JSON.stringify(f.raw_value) : String(f.raw_value)}>{typeof f.raw_value === 'object' && f.raw_value !== null ? JSON.stringify(f.raw_value) : String(f.raw_value || "—")}</td>
                          <td>
                            <div className="text-[11px] font-semibold text-slate-700">{f.display || "Unmapped"}</div>
                            <div className="text-[9px] font-mono text-slate-400 mt-0.5">{f.loinc_code}</div>
                          </td>
                          <td className="text-right">
                            {f.status === 'MAPPED' ? <span className="text-[10px] font-bold text-emerald-600 bg-emerald-50 border border-emerald-100 px-1.5 py-0.5 rounded">{(f.confidence*100).toFixed(0)}%</span> : <span className="text-[10px] font-bold text-amber-600 bg-amber-50 border border-amber-100 px-1.5 py-0.5 rounded">REV</span>}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

              </div>
            )}
          </div>
        </section>

        {/* Col 3: Output */}
        <section className="w-[400px] flex flex-col shrink-0 panel bg-white border-slate-200 shadow-sm">
          <div className="panel-header bg-slate-50 border-b border-slate-200 text-slate-700">
            <span>3. Output Bundle</span>
            <div className="flex gap-2">
              <div className="w-2 h-2 rounded-full bg-slate-200"></div>
              <div className="w-2 h-2 rounded-full bg-slate-200"></div>
              <div className="w-2 h-2 rounded-full bg-slate-300"></div>
            </div>
          </div>
          
          <div className="panel-body p-0 flex flex-col overflow-hidden bg-white relative">
            {!result ? (
              <div className="h-full flex flex-col items-center justify-center text-slate-400 p-6 text-center select-none font-mono text-[10px]">
                 <svg className="w-8 h-8 text-slate-300 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"/></svg>
                Waiting for pipeline execution...<br/><br/>
                Output will be formatted as a valid FHIR R4 JSON Array.
              </div>
            ) : (
              <>
                <div className="absolute top-3 right-3 flex gap-2 z-10">
                  <button onClick={handleCopy} className="bg-white hover:bg-slate-50 text-slate-600 font-medium tracking-wide text-[10px] px-2.5 py-1.5 rounded-md border border-slate-200 shadow-sm transition-colors">
                    {copied ? 'Copied' : 'Copy JSON'}
                  </button>
                  <button onClick={handleDownload} className="bg-emerald-600 hover:bg-emerald-700 text-white font-medium tracking-wide text-[10px] px-2.5 py-1.5 rounded-md border border-emerald-600 shadow-sm transition-colors">
                    Download
                  </button>
                </div>
                <div className="p-4 pt-12 overflow-auto h-full">
                  <pre className="code-viewer border border-slate-100 shadow-inner">
                    {JSON.stringify(result.fhir_bundle, null, 2)}
                  </pre>
                </div>
              </>
            )}
          </div>
        </section>

      </main>
    </div>
  );
}
