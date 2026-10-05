import React, { useMemo } from 'react';
import { AssistantState } from '../types';

interface EdgePanelProps {
  confidence: number;
  peakMicLevel: number;
  confidenceHistory: number[];
  state: AssistantState;
}

export const EdgePanel: React.FC<EdgePanelProps> = ({
  confidence,
  peakMicLevel,
  confidenceHistory,
  state,
}) => {
  const safeConf = Math.min(1, Math.max(0, confidence));
  const safePeak = Math.min(1, Math.max(0, peakMicLevel));

  const peakHold = useMemo(() => {
    if (!confidenceHistory.length) return safeConf;
    return Math.max(...confidenceHistory, safeConf);
  }, [confidenceHistory, safeConf]);

  const sparklinePoints = useMemo(() => {
    if (!confidenceHistory.length) return '';
    const W = 300; const H = 44;
    const step = W / Math.max(1, confidenceHistory.length - 1);
    return confidenceHistory.map((val, i) => {
      const x = (i * step).toFixed(1);
      const y = (H - Math.min(1, Math.max(0, val)) * (H - 6) - 3).toFixed(1);
      return `${x},${y}`;
    }).join(' ');
  }, [confidenceHistory]);

  const sparklineFill = useMemo(() => {
    if (!confidenceHistory.length) return '';
    const W = 300; const H = 44;
    const step = W / Math.max(1, confidenceHistory.length - 1);
    const pts = confidenceHistory.map((val, i) => {
      const x = (i * step).toFixed(1);
      const y = (H - Math.min(1, Math.max(0, val)) * (H - 6) - 3).toFixed(1);
      return `${x},${y}`;
    });
    return `M0,${H} ${pts.map(p => `L${p}`).join(' ')} L${W},${H} Z`;
  }, [confidenceHistory]);

  const confPct = safeConf * 100;
  const isHot = safeConf >= 0.8;

  return (
    <div
      id="edge-telemetry-panel"
      className="rounded-2xl flex flex-col h-full animate-fade-up"
      style={{
        background: '#ffffff',
        border: `1px solid ${isHot ? 'rgba(0,113,227,0.35)' : 'rgba(0,0,0,0.07)'}`,
        padding: '20px',
        transition: 'border-color 0.4s ease',
      }}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-2.5">
          <div className="w-6 h-6 rounded-md flex items-center justify-center shrink-0" style={{ background: 'rgba(0,113,227,0.15)' }}>
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
              <rect x="1" y="1" width="4" height="4" rx="1" fill="#0071e3"/>
              <rect x="7" y="1" width="4" height="4" rx="1" fill="#0071e3" opacity="0.5"/>
              <rect x="1" y="7" width="4" height="4" rx="1" fill="#0071e3" opacity="0.5"/>
              <rect x="7" y="7" width="4" height="4" rx="1" fill="#0071e3" opacity="0.3"/>
            </svg>
          </div>
          <span className="text-sm font-semibold text-[#1d1d1f]">Edge</span>
          <span className="text-sm" style={{ color: 'rgba(0,0,0,0.67)' }}>·</span>
          <span className="text-sm" style={{ color: 'rgba(0,0,0,0.72)' }}>On-device INT8</span>
        </div>
        <div
          className="text-[13px] font-semibold uppercase tracking-widest px-2 py-0.5 rounded-full"
          style={{ background: 'rgba(0,113,227,0.1)', color: 'rgba(0,113,227,0.7)' }}
        >
          Laptop mic
        </div>
      </div>

      {/* Big confidence number */}
      <div className="mb-4">
        <div className="flex items-baseline justify-between mb-3">
          <span className="text-[13px] uppercase tracking-widest font-semibold" style={{ color: 'rgba(0,0,0,0.72)' }}>
            Wake confidence
          </span>
          <span className="text-[13px] font-medium" style={{ color: 'rgba(0,0,0,0.67)' }}>
            Peak {(peakHold * 100).toFixed(0)}%
          </span>
        </div>
        <div className="flex items-baseline gap-1 mb-3">
          <span className="text-5xl font-bold text-[#1d1d1f] num" style={{ letterSpacing: '-0.05em' }}>
            {confPct.toFixed(1)}
          </span>
          <span className="text-xl font-medium" style={{ color: 'rgba(0,0,0,0.67)' }}>%</span>
          {isHot && (
            <span className="ml-2 text-[13px] font-semibold uppercase tracking-widest px-2 py-0.5 rounded-full" style={{ background: 'rgba(0,113,227,0.2)', color: '#0071e3' }}>
              Triggered
            </span>
          )}
        </div>

        {/* Confidence bar */}
        <div className="relative h-1.5 rounded-full overflow-hidden" style={{ background: 'rgba(0,0,0,0.06)' }}>
          <div
            className="bar-fill"
            style={{ width: `${Math.max(1, confPct)}%` }}
          />
          {/* Threshold marker */}
          <div className="absolute top-0 bottom-0 w-px" style={{ left: '80%', background: 'rgba(0,0,0,0.25)' }} />
        </div>
        <div className="flex justify-between mt-1.5 text-[13px]" style={{ color: 'rgba(0,0,0,0.57)' }}>
          <span>0%</span>
          <span style={{ color: 'rgba(0,0,0,0.72)' }}>80% threshold</span>
          <span>100%</span>
        </div>
      </div>

      {/* Sparkline */}
      <div className="mb-4 rounded-xl overflow-hidden" style={{ background: 'rgba(0,0,0,0.03)', border: '1px solid rgba(0,0,0,0.05)', padding: '12px 12px 8px' }}>
        <div className="text-[13px] uppercase tracking-widest font-semibold mb-2" style={{ color: 'rgba(0,0,0,0.62)' }}>
          Confidence timeline
        </div>
        {sparklinePoints ? (
          <svg width="100%" height="44" viewBox="0 0 300 44" preserveAspectRatio="none">
            <defs>
              <linearGradient id="spark-grad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#0071e3" stopOpacity="0.35"/>
                <stop offset="100%" stopColor="#0071e3" stopOpacity="0"/>
              </linearGradient>
            </defs>
            {/* Reference line at 80% */}
            <line x1="0" y1={44 * 0.2} x2="300" y2={44 * 0.2} stroke="rgba(0,0,0,0.1)" strokeDasharray="4 3" strokeWidth="1"/>
            <path d={sparklineFill} fill="url(#spark-grad)"/>
            <polyline fill="none" stroke="#0071e3" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" points={sparklinePoints}/>
          </svg>
        ) : (
          <div className="h-11 flex items-center justify-center text-sm" style={{ color: 'rgba(0,0,0,0.57)' }}>
            Awaiting frames...
          </div>
        )}
      </div>

      {/* Mic level */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <span className="text-[13px] uppercase tracking-widest font-semibold" style={{ color: 'rgba(0,0,0,0.72)' }}>
            Mic peak
          </span>
          <span className="text-sm font-mono" style={{ color: 'rgba(0,0,0,0.77)' }}>
            {(safePeak * 100).toFixed(0)}% &nbsp;
            <span style={{ color: 'rgba(0,0,0,0.57)' }}>
              ({safePeak > 0 ? (20 * Math.log10(Math.max(0.001, safePeak))).toFixed(1) : '−∞'} dB)
            </span>
          </span>
        </div>
        <div className="relative h-1 rounded-full overflow-hidden" style={{ background: 'rgba(0,0,0,0.06)' }}>
          <div
            style={{
              height: '100%',
              width: `${Math.max(1, safePeak * 100)}%`,
              borderRadius: '9999px',
              background: safePeak > 0.8 ? '#ff3b30' : safePeak > 0.5 ? '#ff9f0a' : '#0071e3',
              transition: 'width 0.08s ease, background 0.3s ease',
            }}
          />
        </div>
      </div>
    </div>
  );
};
