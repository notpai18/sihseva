import React, { useEffect, useRef } from 'react';
import { Copy, Check } from 'lucide-react';
import { FinalTranscriptItem, AssistantState } from '../types';

interface TranscriptPanelProps {
  finals: FinalTranscriptItem[];
  partialText: string;
  state: AssistantState;
  onClear: () => void;
}

export const TranscriptPanel: React.FC<TranscriptPanelProps> = ({
  finals,
  partialText,
  state,
}) => {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [copied, setCopied] = React.useState(false);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [finals, partialText]);

  const handleCopy = () => {
    const text = finals.map((f) => `[${f.timestamp}] ${f.text}`).join('\n');
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const hasContent = finals.length > 0 || !!partialText;

  return (
    <section
      id="live-transcript-panel"
      className="rounded-2xl flex flex-col flex-1 min-h-0 overflow-hidden"
      style={{ background: '#ffffff', border: '1px solid rgba(0,0,0,0.07)' }}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between px-5 py-3.5 shrink-0"
        style={{ borderBottom: '1px solid rgba(0,0,0,0.06)' }}
      >
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded-md flex items-center justify-center shrink-0" style={{ background: 'rgba(0,113,227,0.15)' }}>
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                <path d="M1 2.5C1 1.7 1.7 1 2.5 1h7C10.3 1 11 1.7 11 2.5v5C11 8.3 10.3 9 9.5 9H7l-2 2V9H2.5C1.7 9 1 8.3 1 7.5v-5z" stroke="#0071e3" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </div>
            <span className="text-sm font-semibold text-[#1d1d1f]">Transcript</span>
          </div>

          {state === 'STREAMING' && (
            <div className="flex items-center gap-1.5 animate-fade-in">
              <div className="flex items-end gap-0.5 h-3">
                {[1,2,3].map(i => (
                  <div key={i} className={`wave-bar h-3 animate-wave-${i}`} style={{ width: '2px' }} />
                ))}
              </div>
              <span className="text-[13px] font-medium" style={{ color: '#0071e3' }}>
                Live
              </span>
            </div>
          )}
        </div>

        <div className="flex items-center gap-3">
          <span className="text-[14px]" style={{ color: 'rgba(0,0,0,0.62)' }}>
            {finals.length} {finals.length === 1 ? 'utterance' : 'utterances'}
          </span>
          {finals.length > 0 && (
            <button
              type="button"
              onClick={handleCopy}
              className="flex items-center gap-1.5 text-sm font-medium px-2.5 py-1 rounded-lg transition-all"
              style={{
                background: 'rgba(0,0,0,0.05)',
                color: copied ? '#248a3d' : 'rgba(0,0,0,0.4)',
                border: '1px solid rgba(0,0,0,0.07)',
              }}
            >
              {copied ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
              <span>{copied ? 'Copied' : 'Copy'}</span>
            </button>
          )}
        </div>
      </div>

      {/* Content */}
      <div
        ref={scrollRef}
        className="flex-1 min-h-0 overflow-y-auto p-5"
        style={{ scrollBehavior: 'smooth' }}
      >
        {!hasContent ? (
          <div className="h-full flex flex-col items-center justify-center text-center gap-3 min-h-[80px]">
            <div
              className="w-10 h-10 rounded-2xl flex items-center justify-center"
              style={{ background: 'rgba(0,0,0,0.04)', border: '1px solid rgba(0,0,0,0.06)' }}
            >
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M8 2C6 2 4.3 3.5 4 5.4C3.3 5.6 3 6.2 3 7C3 7.8 3.7 8.5 4.5 8.5H11.5C12.3 8.5 13 7.8 13 7C13 6.2 12.7 5.6 12 5.4C11.7 3.5 10 2 8 2Z" stroke="rgba(0,0,0,0.2)" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M8 8.5V13" stroke="rgba(0,0,0,0.15)" strokeWidth="1" strokeLinecap="round"/>
              </svg>
            </div>
            <div>
              <p className="text-base font-medium text-[#1d1d1f] mb-0.5 opacity-60">Waiting for "Hey Tara"</p>
              <p className="text-sm" style={{ color: 'rgba(0,0,0,0.57)' }}>
                Committed utterances will appear here in real time
              </p>
            </div>
          </div>
        ) : (
          <div className="space-y-2.5">
            {finals.map((item, idx) => (
              <div
                key={item.id}
                className="group rounded-xl p-4 animate-fade-up"
                style={{
                  background: 'rgba(0,0,0,0.03)',
                  border: '1px solid rgba(0,0,0,0.06)',
                  animationDelay: `${idx * 0.03}s`,
                  transition: 'border-color 0.2s ease',
                }}
                onMouseEnter={e => (e.currentTarget.style.borderColor = 'rgba(0,113,227,0.2)')}
                onMouseLeave={e => (e.currentTarget.style.borderColor = 'rgba(0,0,0,0.06)')}
              >
                <div className="flex items-center justify-between gap-2 mb-2">
                  <span className="text-[13px] font-mono" style={{ color: 'rgba(0,0,0,0.62)' }}>
                    {item.timestamp}
                  </span>
                  {item.moonshineLatencyMs !== undefined && (
                    <span
                      className="text-[13px] font-semibold px-2 py-0.5 rounded-full"
                      style={{ background: 'rgba(0,113,227,0.12)', color: 'rgba(0,113,227,0.8)' }}
                    >
                      {item.moonshineLatencyMs}ms
                    </span>
                  )}
                </div>
                <p className="text-base font-medium leading-relaxed" style={{ color: 'rgba(0,0,0,0.88)' }}>
                  {item.text}
                </p>
              </div>
            ))}

            {partialText && (
              <div
                className="rounded-xl p-4 animate-fade-in"
                style={{
                  background: 'rgba(0,113,227,0.06)',
                  border: '1px solid rgba(0,113,227,0.2)',
                }}
              >
                <div className="flex items-center gap-2 mb-2">
                  <div className="flex items-end gap-0.5 h-3">
                    {[1,2,3,4].map(i => (
                      <div key={i} className={`wave-bar h-3 animate-wave-${i}`} style={{ width: '2px' }} />
                    ))}
                  </div>
                  <span className="text-[13px] font-medium" style={{ color: 'rgba(0,113,227,0.7)' }}>
                    Transcribing...
                  </span>
                </div>
                <p className="text-base italic leading-relaxed" style={{ color: 'rgba(0,0,0,0.62)' }}>
                  "{partialText}
                  <span
                    className="inline-block w-0.5 h-3.5 ml-0.5 align-middle rounded-full animate-pulse"
                    style={{ background: '#0071e3' }}
                  />
                  "
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </section>
  );
};
