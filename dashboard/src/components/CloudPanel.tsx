import React, { useMemo } from 'react';
import { AssistantState, LatencyRecord } from '../types';

interface CloudPanelProps {
  state: AssistantState;
  vadScore: number;
  silenceSeconds: number;
  latestLatency: LatencyRecord | null;
  latencyHistory: LatencyRecord[];
}

export const CloudPanel: React.FC<CloudPanelProps> = ({
  state,
  vadScore,
  silenceSeconds,
  latestLatency,
  latencyHistory,
}) => {
  const isStreaming = state === 'STREAMING';
  const safeVad = Math.min(1, Math.max(0, vadScore));
  const isSpeechActive = isStreaming && safeVad >= 0.5;

  const latencyStats = useMemo(() => {
    if (!latencyHistory.length) return null;
    const values = latencyHistory.map((h) => h.ms);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const avg = values.reduce((a, v) => a + v, 0) / values.length;
    return { min, max, avg };
  }, [latencyHistory]);

  const sparklinePoints = useMemo(() => {
    if (latencyHistory.length < 2) return '';
    const W = 300; const H = 44;
    const values = latencyHistory.map((h) => h.ms);
    const lo = Math.min(...values) * 0.9;
    const hi = Math.max(...values) * 1.1 || 500;
    const range = hi - lo || 1;
    const step = W / (latencyHistory.length - 1);
    return latencyHistory.map((rec, i) => {
      const x = (i * step).toFixed(1);
      const norm = (rec.ms - lo) / range;
      const y = (H - norm * (H - 6) - 3).toFixed(1);
      return `${x},${y}`;
    }).join(' ');
  }, [latencyHistory]);

  const sparklineFill = useMemo(() => {
    if (latencyHistory.length < 2) return '';
    const W = 300; const H = 44;
    const values = latencyHistory.map((h) => h.ms);
    const lo = Math.min(...values) * 0.9;
    const hi = Math.max(...values) * 1.1 || 500;
    const range = hi - lo || 1;
    const step = W / (latencyHistory.length - 1);
    const pts = latencyHistory.map((rec, i) => {
      const x = (i * step).toFixed(1);
      const norm = (rec.ms - lo) / range;
      const y = (H - norm * (H - 6) - 3).toFixed(1);
      return `${x},${y}`;
    });
    return `M0,${H} ${pts.map(p => `L${p}`).join(' ')} L${W},${H} Z`;
  }, [latencyHistory]);

  const latencyRating = latestLatency
    ? latestLatency.ms < 250 ? { label: 'Optimal', color: '#34c759' }
    : latestLatency.ms < 400 ? { label: 'Normal', color: '#0071e3' }
    : { label: 'High', color: '#ff9f0a' }
    : null;

  return (
    <div
      id="cloud-telemetry-panel"
      className="rounded-2xl flex flex-col h-full animate-fade-up"
      style={{
        background: '#ffffff',
        border: '1px solid rgba(0,0,0,0.07)',
        padding: '20px',
        animationDelay: '0.05s',
      }}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-2.5">
          <div className="w-6 h-6 rounded-md flex items-center justify-center shrink-0" style={{ background: 'rgba(0,113,227,0.15)' }}>
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
              <path d="M2 8.5C1.2 8.5 0.5 7.8 0.5 7C0.5 6.3 1 5.7 1.7 5.5C1.6 5.3 1.5 5.1 1.5 4.8C1.5 3.8 2.3 3 3.3 3C3.5 3 3.7 3 3.9 3.1C4.3 2.4 5 2 5.8 2C7.1 2 8 3 8 4.3C8 4.4 8 4.5 8 4.5C8.2 4.5 8.4 4.5 8.5 4.5C9.3 4.5 10 5.2 10 6C10 6.8 9.3 7.5 8.5 7.5" stroke="#0071e3" strokeWidth="1" strokeLinecap="round"/>
              <path d="M4 9L6 11L8 9" stroke="#0071e3" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M6 6V11" stroke="#0071e3" strokeWidth="1" strokeLinecap="round"/>
            </svg>
          </div>
          <span className="text-sm font-semibold text-[#1d1d1f]">Cloud ASR</span>
          <span className="text-sm" style={{ color: 'rgba(0,0,0,0.67)' }}>·</span>
          <span className="text-sm" style={{ color: 'rgba(0,0,0,0.72)' }}>Moonshine</span>
        </div>
        <div
          className="flex items-center gap-1.5 text-[13px] font-semibold uppercase tracking-widest px-2 py-0.5 rounded-full"
          style={{
            background: isStreaming ? 'rgba(0,113,227,0.15)' : 'rgba(0,0,0,0.05)',
            color: isStreaming ? '#0071e3' : 'rgba(0,0,0,0.3)',
          }}
        >
          {isStreaming && (
            <div className="w-1.5 h-1.5 rounded-full" style={{ background: '#0071e3', boxShadow: '0 0 5px #0071e3' }} />
          )}
          {isStreaming ? 'Streaming' : 'Standby'}
        </div>
      </div>

      {/* Latency readout */}
      <div className="mb-4">
        <div className="flex items-baseline justify-between mb-3">
          <span className="text-[13px] uppercase tracking-widest font-semibold" style={{ color: 'rgba(0,0,0,0.72)' }}>
            Ingest latency
          </span>
          {latencyRating && (
            <span className="text-[13px] font-semibold" style={{ color: latencyRating.color }}>
              {latencyRating.label}
            </span>
          )}
        </div>

        <div className="flex items-baseline gap-1.5 mb-1">
          {latestLatency ? (
            <>
              <span className="text-5xl font-bold text-[#1d1d1f] num" style={{ letterSpacing: '-0.05em' }}>
                {latestLatency.ms.toFixed(0)}
              </span>
              <span className="text-xl font-medium" style={{ color: 'rgba(0,0,0,0.67)' }}>ms</span>
            </>
          ) : (
            <span className="text-3xl font-medium" style={{ color: 'rgba(0,0,0,0.57)' }}>—</span>
          )}
        </div>
        <p className="text-[14px]" style={{ color: 'rgba(0,0,0,0.62)' }}>
          Keyword end → cloud audio chunk
        </p>
      </div>

      {/* Latency sparkline */}
      <div className="mb-4 rounded-xl overflow-hidden" style={{ background: 'rgba(0,0,0,0.03)', border: '1px solid rgba(0,0,0,0.05)', padding: '12px 12px 8px' }}>
        <div className="flex items-center justify-between mb-2">
          <span className="text-[13px] uppercase tracking-widest font-semibold" style={{ color: 'rgba(0,0,0,0.62)' }}>
            Latency timeline
          </span>
          {latencyStats && (
            <span className="text-[13px] font-mono" style={{ color: 'rgba(0,0,0,0.67)' }}>
              avg {latencyStats.avg.toFixed(0)}ms · min {latencyStats.min.toFixed(0)}ms
            </span>
          )}
        </div>
        {sparklinePoints ? (
          <svg width="100%" height="44" viewBox="0 0 300 44" preserveAspectRatio="none">
            <defs>
              <linearGradient id="lat-grad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#0071e3" stopOpacity="0.3"/>
                <stop offset="100%" stopColor="#0071e3" stopOpacity="0"/>
              </linearGradient>
            </defs>
            <path d={sparklineFill} fill="url(#lat-grad)"/>
            <polyline fill="none" stroke="#0071e3" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" points={sparklinePoints}/>
            {/* Latest point dot */}
          </svg>
        ) : (
          <div className="h-11 flex items-center justify-center text-sm" style={{ color: 'rgba(0,0,0,0.57)' }}>
            {latestLatency ? '1 event captured' : 'Awaiting latency data...'}
          </div>
        )}
      </div>

      {/* VAD */}
      <div className="mt-auto">
        <div className="flex items-center justify-between mb-2">
          <span className="text-[13px] uppercase tracking-widest font-semibold" style={{ color: 'rgba(0,0,0,0.72)' }}>
            Voice activity
          </span>
          {isStreaming && (
            <span className="text-sm font-mono" style={{ color: 'rgba(0,0,0,0.77)' }}>
              {(safeVad * 100).toFixed(0)}%
            </span>
          )}
        </div>

        {isStreaming ? (
          <div className="rounded-xl p-3" style={{ background: 'rgba(0,0,0,0.03)', border: '1px solid rgba(0,0,0,0.06)' }}>
            <div className="flex items-center justify-between mb-2.5">
              <div className="flex items-center gap-2">
                <div
                  className="relative w-2 h-2"
                >
                  <div
                    className="absolute inset-0 rounded-full"
                    style={{ background: isSpeechActive ? '#34c759' : 'rgba(0,0,0,0.2)' }}
                  />
                  {isSpeechActive && (
                    <div
                      className="absolute inset-0 rounded-full animate-ping"
                      style={{ background: '#34c759', opacity: 0.4 }}
                    />
                  )}
                </div>
                <span className="text-sm font-medium" style={{ color: isSpeechActive ? '#248a3d' : 'rgba(0,0,0,0.4)' }}>
                  {isSpeechActive ? 'Speech detected' : 'Silence / filtering'}
                </span>
              </div>
              {silenceSeconds > 0 && (
                <span className="text-sm font-mono" style={{ color: silenceSeconds > 0.8 ? '#ff9f0a' : 'rgba(0,0,0,0.35)' }}>
                  {silenceSeconds.toFixed(2)}s silence
                </span>
              )}
            </div>
            <div className="relative h-1 rounded-full overflow-hidden" style={{ background: 'rgba(0,0,0,0.06)' }}>
              <div
                style={{
                  height: '100%', borderRadius: '9999px',
                  width: `${Math.max(2, safeVad * 100)}%`,
                  background: safeVad >= 0.5 ? '#34c759' : '#0071e3',
                  transition: 'width 0.08s ease, background 0.3s ease',
                }}
              />
            </div>
          </div>
        ) : (
          <div className="rounded-xl p-3 flex items-center justify-center" style={{ background: 'rgba(0,0,0,0.02)', border: '1px solid rgba(0,0,0,0.05)' }}>
            <span className="text-sm" style={{ color: 'rgba(0,0,0,0.57)' }}>
              Active during streaming
            </span>
          </div>
        )}
      </div>
    </div>
  );
};
