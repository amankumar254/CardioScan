import { useState, useCallback } from "react";
import UploadView from "./components/UploadView";
import ResultsView from "./components/ResultsView";

export default function App() {
  const [view, setView] = useState("upload");
  const [results, setResults] = useState(null);
  const [fileName, setFileName] = useState("");
  const [error, setError] = useState("");

  const handleUpload = useCallback(async ({ dat, hea }) => {
    setFileName(dat.name);
    setError("");
    setView("analyzing");
    try {
      const form = new FormData();
      form.append("dat", dat);
      if (hea) form.append("hea", hea);
      const response = await fetch("/api/analyze", { method: "POST", body: form });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Analysis failed");
      setResults(data);
      setView("results");
    } catch (err) {
      setError(err.message || "Analysis failed");
      setView("upload");
    }
  }, []);

  const reset = () => {
    setView("upload");
    setResults(null);
    setFileName("");
    setError("");
  };

  return (
    <div className="app">
      <header className="header">
        <div className="brand"><span className="brand-mark">ECG</span><span>CardioScan AI</span></div>
        <span className="header-badge">ECG Arrhythmia Detection</span>
      </header>
      <main className="main">
        {view === "upload" && <UploadView onUpload={handleUpload} error={error} />}
        {view === "analyzing" && <div className="card analyzing"><div className="pulse">+</div><h2>Analyzing ECG</h2><p>Processing {fileName}</p><div className="progress"><span /></div></div>}
        {view === "results" && <ResultsView results={results} fileName={fileName} onReset={reset} />}
      </main>
      <footer className="footer">CardioScan AI is an academic research prototype and not a medical device.</footer>
    </div>
  );
}
