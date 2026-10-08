import { useRef } from "react";

export default function UploadView({ onUpload, error }) {
  const ref = useRef(null);

  const selectFiles = (files) => {
    const list = Array.from(files || []);
    const dat = list.find(f => f.name.toLowerCase().endsWith(".dat"));
    const hea = list.find(f => f.name.toLowerCase().endsWith(".hea"));
    if (dat) onUpload({ dat, hea });
  };

  return (
    <section className="upload">
      <p className="eyebrow">ECG ARRHYTHMIA DETECTION</p>
      <h1>Analyze an ECG recording</h1>
      <p className="lead">Upload a matching WFDB <strong>.dat</strong> signal file and <strong>.hea</strong> header to run the CardioScan pipeline.</p>
      <div className="dropzone" onClick={() => ref.current?.click()} onDragOver={e => e.preventDefault()} onDrop={e => { e.preventDefault(); selectFiles(e.dataTransfer.files); }}>
        <input ref={ref} type="file" accept=".dat,.hea" multiple hidden onChange={e => selectFiles(e.target.files)} />
        <div className="upload-icon">ECG</div>
        <h3>Drag and drop your files here</h3>
        <p>or click to choose .dat and .hea files</p>
        <button type="button">Select files</button>
      </div>
      {error && <div className="error">{error}</div>}
      <div className="info-grid">
        <div><b>5</b><span>AAMI-oriented classes</span></div>
        <div><b>280</b><span>Samples per beat</span></div>
        <div><b>0.5–45 Hz</b><span>Bandpass filter</span></div>
      </div>
    </section>
  );
}
