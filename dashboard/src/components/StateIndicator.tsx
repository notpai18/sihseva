import React from 'react';
import { AssistantState } from '../types';

interface StateIndicatorProps {
  state: AssistantState;
  lastConfidence: number;
  vadScore: number;
  silenceSeconds: number;
}

const StateIcon: React.FC<{ state: AssistantState }> = ({ state }) => {
  if (state === 'LISTENING') {
    return (
      <div className="relative w-12 h-12 flex items-center justify-center">
        {/* Outer ring */}
        <div
          className="absolute inset-0 rounded-full"
          style={{ border: '1px solid rgba(0,113,227,0.25)' }}
        />
        {/* Pulsing ring */}
        <div
          className="absolute inset-0 rounded-full animate-breathe"
          style={{ border: '1px solid rgba(0,113,227,0.15)' }}
        />
        {/* Inner dot */}
        <div className="w-2.5 h-2.5 rounded-full" style={{ background: '#0071e3', boxShadow: '0 0 10px rgba(0,113,227,0.6)' }} />
      </div>
    );
  }
  if (state === 'DETECTED') {
    return (
      <div className="relative w-12 h-12 flex items-center justify-center">
        <div className="absolute inset-0 rounded-full animate-ping" style={{ background: 'rgba(0,113,227,0.15)' }} />
        <div className="w-10 h-10 rounded-full flex items-center justify-center" style={{ background: 'rgba(0,113,227,0.2)', border: '1px solid rgba(0,113,227,0.4)' }}>
          <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
            <path d="M4 9l3.5 3.5L14 6" stroke="#0071e3" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </div>
      </div>
    );
  }
  // STREAMING
  return (
    <div className="relative w-12 h-12 flex items-center justify-center">
      <div className="flex items-end gap-0.5 h-6">
        {[1,2,3,4,5].map(i => (
          <div key={i} className={`wave-bar h-6 animate-wave-${i}`} />
        ))}
      </div>
    </div>
  );
};

export const StateIndicator: React.FC<StateIndicatorProps> = ({
  state,
  lastConfidence,
  vadScore,
  silenceSeconds,
}) => {
  const stateLabel =
    state === 'LISTENING' ? 'Listening'
    : state === 'DETECTED' ? 'Detected'
    : 'Streaming';

  const stateSubtext =
    state === 'LISTENING' ? 'Waiting for wake word'
    : state === 'DETECTED' ? `${(lastConfidence * 100).toFixed(1)}% confidence — pipeline opening`
    : 'Transmitting to Moonshine ASR';

  const accentColor = state === 'STREAMING' ? '#0071e3' : state === 'DETECTED' ? '#0071e3' : 'rgba(0,113,227,0.7)';

  return (
    <section
      id="main-state-indicator"
      className="relative overflow-hidden rounded-2xl animate-fade-up"
      style={{
        background: 'linear-gradient(135deg, #ffffff 0%, #fbfbfd 100%)',
        border: '1px solid rgba(0,0,0,0.07)',
        padding: '20px 24px',
      }}
    >
      {/* Ambient glow */}
      <div
        className="absolute top-0 right-0 w-64 h-64 pointer-events-none"
        style={{
          background: `radial-gradient(circle at 80% 0%, ${accentColor}20 0%, transparent 60%)`,
          transition: 'background 0.6s ease',
        }}
      />

      <div className="relative z-10 flex items-center justify-between gap-6">
        {/* Left: State display */}
        <div className="flex items-center gap-5">
          <StateIcon state={state} />
          <div>
            <div className="flex items-center gap-2.5 mb-0.5">
              <h2 className="text-2xl font-semibold text-[#1d1d1f] tracking-tight leading-none">
                {stateLabel}
              </h2>
              <span
                className="text-[13px] font-semibold uppercase tracking-widest px-2 py-0.5 rounded-full"
                style={{ background: 'rgba(0,113,227,0.15)', color: '#0071e3' }}
              >
                {state === 'LISTENING' ? 'Edge' : state === 'DETECTED' ? 'Wake' : 'Cloud'}
              </span>
            </div>
            <p className="text-base" style={{ color: 'rgba(0,0,0,0.77)' }}>
              {stateSubtext}
            </p>
          </div>
        </div>

        {/* Right: Live metrics */}
        <div className="hidden sm:flex items-center gap-5">
          <div className="text-right">
            <div className="text-[13px] uppercase tracking-widest font-semibold mb-1" style={{ color: 'rgba(0,0,0,0.67)' }}>
              Confidence
            </div>
            <div className="text-3xl font-bold text-[#1d1d1f] num" style={{ letterSpacing: '-0.04em' }}>
              {(lastConfidence * 100).toFixed(0)}
              <span className="text-base font-medium ml-0.5" style={{ color: 'rgba(0,0,0,0.72)' }}>%</span>
            </div>
          </div>

          <div className="w-px h-10" style={{ background: 'rgba(0,0,0,0.08)' }} />

          <div className="text-right">
            <div className="text-[13px] uppercase tracking-widest font-semibold mb-1" style={{ color: 'rgba(0,0,0,0.67)' }}>
              {state === 'STREAMING' ? 'VAD' : 'Status'}
            </div>
            {state === 'STREAMING' ? (
              <div className="text-3xl font-bold text-[#1d1d1f] num" style={{ letterSpacing: '-0.04em' }}>
                {(vadScore * 100).toFixed(0)}
                <span className="text-base font-medium ml-0.5" style={{ color: 'rgba(0,0,0,0.72)' }}>%</span>
              </div>
            ) : (
              <div className="flex items-center gap-1.5 justify-end">
                <div className="w-1.5 h-1.5 rounded-full" style={{ background: '#34c759', boxShadow: '0 0 6px #34c759' }} />
                <span className="text-base font-semibold" style={{ color: '#34c759' }}>Active</span>
              </div>
            )}
          </div>

          {state === 'STREAMING' && silenceSeconds > 0 && (
            <>
              <div className="w-px h-10" style={{ background: 'rgba(0,0,0,0.08)' }} />
              <div className="text-right">
                <div className="text-[13px] uppercase tracking-widest font-semibold mb-1" style={{ color: 'rgba(0,0,0,0.67)' }}>
                  Silence
                </div>
                <div className="text-3xl font-bold num" style={{ color: silenceSeconds > 0.8 ? '#ff9f0a' : 'white', letterSpacing: '-0.04em' }}>
                  {silenceSeconds.toFixed(1)}
                  <span className="text-base font-medium ml-0.5" style={{ color: 'rgba(0,0,0,0.72)' }}>s</span>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </section>
  );
};
