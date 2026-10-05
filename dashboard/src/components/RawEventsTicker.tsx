import React, { useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { RawEventItem } from '../types';

interface RawEventsTickerProps {
  events: RawEventItem[];
  onClear: () => void;
}

const EVENT_COLORS: Record<string, { bg: string; text: string }> = {
  detected:     { bg: 'rgba(0,113,227,0.2)',  text: '#0071e3' },
  streaming:    { bg: 'rgba(0,113,227,0.1)',  text: 'rgba(0,113,227,0.7)' },
  asr_final:    { bg: 'rgba(52,199,89,0.1)',   text: '#248a3d' },
  asr_partial:  { bg: 'rgba(0,113,227,0.08)', text: 'rgba(0,113,227,0.6)' },
  latency:      { bg: 'rgba(175,82,222,0.1)',  text: '#8944ab' },
  listening:    { bg: 'rgba(0,0,0,0.05)' , text: 'rgba(0,0,0,0.3)' },
  utterance_end:{ bg: 'rgba(255,159,10,0.1)',  text: '#b25000' },
};

export const RawEventsTicker: React.FC<RawEventsTickerProps> = ({ events, onClear }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const recent = events.slice(0, 20);

  return (
    <div
      id="raw-events-drawer"
      className="rounded-xl overflow-hidden text-sm"
      style={{
        background: '#ffffff',
        border: '1px solid rgba(0,0,0,0.07)',
      }}
    >
      <button
        type="button"
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-3.5 py-2 flex items-center justify-between transition-all group"
        style={{ color: 'rgba(0,0,0,0.77)' }}
      >
        <div className="flex items-center gap-2.5">
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
            <rect x="1" y="3" width="6" height="1" rx="0.5" fill="currentColor"/>
            <rect x="1" y="5.5" width="10" height="1" rx="0.5" fill="currentColor" opacity="0.6"/>
            <rect x="1" y="8" width="8" height="1" rx="0.5" fill="currentColor" opacity="0.4"/>
          </svg>
          <span className="font-medium text-[14px]" style={{ color: 'rgba(0,0,0,0.62)' }}>
            Wire frames
          </span>
          <span
            className="text-[13px] px-1.5 py-0.5 rounded-full font-semibold"
            style={{ background: 'rgba(0,0,0,0.06)', color: 'rgba(0,0,0,0.72)' }}
          >
            {events.length}
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-[13px]">{isExpanded ? 'Hide' : 'Inspect'}</span>
          {isExpanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
        </div>
      </button>

      {isExpanded && (
        <div
          className="animate-fade-up"
          style={{ borderTop: '1px solid rgba(0,0,0,0.06)', padding: '10px 12px' }}
        >
          <div className="flex justify-between items-center mb-2">
            <span className="text-[13px] uppercase tracking-widest font-semibold" style={{ color: 'rgba(0,0,0,0.62)' }}>
              Last 20 messages
            </span>
            <button
              type="button"
              onClick={onClear}
              className="text-[13px] transition-opacity hover:opacity-100 opacity-50"
              style={{ color: 'rgba(0,0,0,0.62)' }}
            >
              Clear
            </button>
          </div>

          <div className="space-y-1 max-h-44 overflow-y-auto">
            {recent.length === 0 ? (
              <div className="py-3 text-center text-[14px]" style={{ color: 'rgba(0,0,0,0.57)' }}>
                No frames yet. Connect or start demo.
              </div>
            ) : (
              recent.map((evt) => {
                const c = EVENT_COLORS[evt.type] ?? { bg: 'rgba(0,0,0,0.05)', text: 'rgba(0,0,0,0.3)' };
                return (
                  <div
                    key={evt.id}
                    className="flex items-start gap-2 px-2 py-1.5 rounded-lg"
                    style={{ background: 'rgba(0,0,0,0.02)' }}
                  >
                    <span className="text-[12px] font-mono shrink-0 pt-0.5" style={{ color: 'rgba(0,0,0,0.57)' }}>
                      {evt.timestamp}
                    </span>
                    <span
                      className="text-[12px] font-semibold px-1.5 py-0.5 rounded shrink-0"
                      style={{ background: c.bg, color: c.text }}
                    >
                      {evt.type}
                    </span>
                    <span className="text-[13px] font-mono truncate" style={{ color: 'rgba(0,0,0,0.67)' }}>
                      {JSON.stringify(evt.data)}
                    </span>
                  </div>
                );
              })
            )}
          </div>
        </div>
      )}
    </div>
  );
};
