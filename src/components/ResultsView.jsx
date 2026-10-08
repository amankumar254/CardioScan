import { useEffect, useRef } from "react";

function ECGChart({ signal, peaks, events }) {
  const ref = useRef(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !signal.length) return;
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth || 900;
    const height = 260;
    canvas.width = width * dpr; canvas.height = height * dpr;
    const ctx = canvas.getContext("2d"); ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);
    ctx.strokeStyle = "#dfe8e7"; ctx.lineWidth = 1;
    for (let x=0; x<width; x+=20) { ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,height); ctx.stroke(); }
    for (let y=0; y<height; y+=20) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(width,y); ctx.stroke(); }
    const min=Math.min(...signal), max=Math.max(...signal), range=max-min || 1;
    const y=v => height-20-((v-min)/range)*(height-40);
    const x=i => (i/(signal.length-1))*width;
    ctx.strokeStyle="#0d7377"; ctx.lineWidth=1.5; ctx.beginPath();
    signal.forEach((v,i)=>{ const px=x(i), py=y(v); i?ctx.lineTo(px,py):ctx.moveTo(px,py); }); ctx.stroke();
    ctx.fillStyle="#0d7377"; peaks.forEach(p=>{ const i=Math.min(p,signal.length-1); ctx.beginPath(); ctx.arc(x(i),y(signal[i]),3,0,Math.PI*2); ctx.fill(); });
    ctx.fillStyle="#c0394b"; events.forEach(e=>{ const i=Math.min(e.sample_index,signal.length-1); ctx.beginPath(); ctx.arc(x(i),y(signal[i]),5,0,Math.PI*2); ctx.fill(); });
  }, [signal, peaks, events]);
  return <canvas ref={ref} className="ecg-canvas" />;
}

export default function ResultsView({ results, fileName, onReset }) {
  const events = results.arrhythmia_events || [];
  const status = results.overall_status || "normal";
  return (
    <section className="results">
      <div className="results-top"><div><p className="eyebrow">ANALYSIS COMPLETE</p><h1>ECG Analysis Report</h1><p>{fileName}</p></div><button className="secondary" onClick={onReset}>Analyze another</button></div>
      {results.model_warning && <div className="warning">{results.model_warning}</div>}
      <div className="summary">
        <div className={"metric "+status}><span>Overall status</span><strong>{status}</strong></div>
        <div className="metric"><span>Heart rate</span><strong>{results.heart_rate || 0}<small> bpm</small></strong></div>
        <div className="metric"><span>Total beats</span><strong>{results.total_beats || 0}</strong></div>
        <div className="metric"><span>Mean confidence</span><strong>{((results.mean_confidence || 0)*100).toFixed(1)}<small>%</small></strong></div>
      </div>
      <div className="card"><div className="panel-title">ECG waveform</div><ECGChart signal={results.ecg_signal || []} peaks={results.r_peaks || []} events={events}/><div className="legend"><span>R-peaks</span><span>Arrhythmia events</span></div></div>
      <div className="card"><div className="panel-title">Detected events ({events.length})</div>
        {events.length===0 ? <p className="empty">No non-normal events were returned for this recording.</p> :
        <div className="table-wrap"><table><thead><tr><th>Time</th><th>Type</th><th>Severity</th><th>Confidence</th></tr></thead><tbody>
          {events.map((e,i)=><tr key={i}><td>{e.time_sec}s</td><td>{e.type}</td><td><span className={"tag "+e.severity}>{e.severity}</span></td><td>{(e.confidence*100).toFixed(1)}%{e.low_confidence ? " (low)" : ""}</td></tr>)}
        </tbody></table></div>}
      </div>
      <p className="disclaimer">Research prototype only. Results are not clinically validated and must not be used as a diagnosis or treatment decision.</p>
    </section>
  );
}
