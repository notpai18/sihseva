import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  AssistantState,
  ConnectionStatus,
  FinalTranscriptItem,
  LatencyRecord,
  RawEventItem,
  WebSocketPayload,
} from './types';
import { HeaderSettings } from './components/HeaderSettings';
import { StateIndicator } from './components/StateIndicator';
import { EdgePanel } from './components/EdgePanel';
import { CloudPanel } from './components/CloudPanel';
import { TranscriptPanel } from './components/TranscriptPanel';
import { RawEventsTicker } from './components/RawEventsTicker';
import { StreamSimulator } from './utils/simulator';

export default function App() {
  // Config & Connection State
  const [wsUrl, setWsUrl] = useState<string>('ws://127.0.0.1:8766');
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('connecting');
  const [isSimulating, setIsSimulating] = useState<boolean>(false);

  // Core System Telemetry States
  const [assistantState, setAssistantState] = useState<AssistantState>('LISTENING');
  const [confidence, setConfidence] = useState<number>(0.0);
  const [peakMicLevel, setPeakMicLevel] = useState<number>(0.0);
  const [vadScore, setVadScore] = useState<number>(0.0);
  const [silenceSeconds, setSilenceSeconds] = useState<number>(0.0);

  // Sparklines & Histories
  const [confidenceHistory, setConfidenceHistory] = useState<number[]>([]);
  const [latestLatency, setLatestLatency] = useState<LatencyRecord | null>(null);
  const [latencyHistory, setLatencyHistory] = useState<LatencyRecord[]>([]);

  // Transcripts
  const [finals, setFinals] = useState<FinalTranscriptItem[]>([]);
  const [partialText, setPartialText] = useState<string>('');

  // Diagnostic Logs & Counts
  const [rawEvents, setRawEvents] = useState<RawEventItem[]>([]);
  const [utteranceCount, setUtteranceCount] = useState<number>(0);
  const [messageCount, setMessageCount] = useState<number>(0);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const detectedTimeoutRef = useRef<number | null>(null);
  const simulatorRef = useRef<StreamSimulator | null>(null);

  const getTimestamp = () => {
    const d = new Date();
    return d.toTimeString().split(' ')[0] + '.' + String(d.getMilliseconds()).padStart(3, '0').slice(0, 2);
  };

  const processIncomingMessage = useCallback((payload: WebSocketPayload) => {
    if (!payload || typeof payload !== 'object' || !('type' in payload)) return;

    setMessageCount((prev) => prev + 1);

    const eventRecord: RawEventItem = {
      id: Math.random().toString(36).substring(2, 9),
      timestamp: getTimestamp(),
      type: payload.type,
      data: payload as unknown as Record<string, unknown>,
    };
    setRawEvents((prev) => [eventRecord, ...prev.slice(0, 49)]);

    switch (payload.type) {
      case 'listening': {
        setAssistantState('LISTENING');
        const confVal = typeof payload.conf === 'number' ? payload.conf : 0.0;
        const peakVal = typeof payload.peak === 'number' ? payload.peak : 0.0;
        setConfidence(confVal);
        setPeakMicLevel(peakVal);
        setConfidenceHistory((prev) => [...prev.slice(-30), confVal]);
        break;
      }
      case 'detected': {
        setAssistantState('DETECTED');
        setUtteranceCount((prev) => prev + 1);
        const detConf =
          typeof payload.confidence === 'number' ? payload.confidence
          : typeof payload.conf === 'number' ? payload.conf
          : 0.95;
        setConfidence(detConf);
        setConfidenceHistory((prev) => [...prev.slice(-30), detConf]);
        if (detectedTimeoutRef.current) clearTimeout(detectedTimeoutRef.current);
        detectedTimeoutRef.current = window.setTimeout(() => {
          setAssistantState((curr) => (curr === 'DETECTED' ? 'STREAMING' : curr));
        }, 850);
        break;
      }
      case 'streaming': {
        setAssistantState('STREAMING');
        if (typeof payload.conf === 'number') {
          setConfidence(payload.conf);
          setConfidenceHistory((prev) => [...prev.slice(-30), payload.conf]);
        }
        if (typeof payload.vad === 'number') setVadScore(payload.vad);
        if (typeof payload.silence_s === 'number') setSilenceSeconds(payload.silence_s);
        break;
      }
      case 'utterance_end': {
        setAssistantState('LISTENING');
        setPartialText('');
        setVadScore(0.0);
        setSilenceSeconds(0.0);
        break;
      }
      case 'asr_partial': {
        if (typeof payload.text === 'string') setPartialText(payload.text);
        break;
      }
      case 'asr_final': {
        if (typeof payload.text === 'string') {
          const newFinal: FinalTranscriptItem = {
            id: Math.random().toString(36).substring(2, 9),
            timestamp: getTimestamp(),
            text: payload.text,
            moonshineLatencyMs: payload.moonshine_latency_ms,
          };
          setFinals((prev) => [...prev, newFinal]);
          setPartialText('');
        }
        break;
      }
      case 'latency': {
        if (typeof payload.ms === 'number') {
          const newRecord: LatencyRecord = {
            id: Math.random().toString(36).substring(2, 9),
            timestamp: getTimestamp(),
            ms: payload.ms,
          };
          setLatestLatency(newRecord);
          setLatencyHistory((prev) => [...prev.slice(-25), newRecord]);
        }
        break;
      }
      default:
        break;
    }
  }, []);

  useEffect(() => {
    simulatorRef.current = new StreamSimulator(processIncomingMessage);
    return () => { simulatorRef.current?.stop(); };
  }, [processIncomingMessage]);

  const connectWebSocket = useCallback(() => {
    if (isSimulating) return;
    if (wsRef.current) {
      wsRef.current.onclose = null;
      wsRef.current.onerror = null;
      wsRef.current.onmessage = null;
      wsRef.current.close();
      wsRef.current = null;
    }
    setConnectionStatus('connecting');
    try {
      const socket = new WebSocket(wsUrl);
      socket.onopen = () => {
        setConnectionStatus('connected');
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
          reconnectTimeoutRef.current = null;
        }
      };
      socket.onmessage = (event) => {
        try { processIncomingMessage(JSON.parse(event.data)); } catch { /* ignore */ }
      };
      socket.onerror = () => { /* triggers onclose */ };
      socket.onclose = () => {
        setConnectionStatus('reconnecting');
        if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = window.setTimeout(() => { connectWebSocket(); }, 2000);
      };
      wsRef.current = socket;
    } catch {
      setConnectionStatus('reconnecting');
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = window.setTimeout(() => { connectWebSocket(); }, 2000);
    }
  }, [wsUrl, isSimulating, processIncomingMessage]);

  useEffect(() => {
    connectWebSocket();
    return () => {
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (detectedTimeoutRef.current) clearTimeout(detectedTimeoutRef.current);
      if (wsRef.current) { wsRef.current.onclose = null; wsRef.current.close(); wsRef.current = null; }
    };
  }, [connectWebSocket]);

  const handleToggleSimulation = () => {
    if (isSimulating) {
      simulatorRef.current?.stop();
      setIsSimulating(false);
      connectWebSocket();
    } else {
      if (wsRef.current) { wsRef.current.onclose = null; wsRef.current.close(); wsRef.current = null; }
      if (reconnectTimeoutRef.current) { clearTimeout(reconnectTimeoutRef.current); reconnectTimeoutRef.current = null; }
      setConnectionStatus('connected');
      setIsSimulating(true);
      simulatorRef.current?.start();
    }
  };

  const handleClearHistory = () => {
    setFinals([]);
    setPartialText('');
    setLatencyHistory([]);
    setLatestLatency(null);
    setRawEvents([]);
    setUtteranceCount(0);
    setMessageCount(0);
  };

  return (
    <div
      className="h-screen max-h-screen overflow-hidden flex flex-col"
      style={{ background: '#f5f5f7', color: '#1d1d1f' }}
    >
      <HeaderSettings
        wsUrl={wsUrl}
        onUpdateWsUrl={(newUrl) => {
          setWsUrl(newUrl);
          if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
        }}
        connectionStatus={connectionStatus}
        onManualReconnect={connectWebSocket}
        isSimulating={isSimulating}
        onToggleSimulation={handleToggleSimulation}
        utteranceCount={utteranceCount}
        messageCount={messageCount}
        onClearHistory={handleClearHistory}
      />

      {/* Main canvas */}
      <main className="flex-1 min-h-0 overflow-hidden">
        <div className="h-full max-w-[1440px] mx-auto px-5 py-4 flex flex-col gap-3">

          {/* State hero */}
          <div className="shrink-0">
            <StateIndicator
              state={assistantState}
              lastConfidence={confidence}
              vadScore={vadScore}
              silenceSeconds={silenceSeconds}
            />
          </div>

          {/* Telemetry row */}
          <div className="shrink-0 grid grid-cols-1 lg:grid-cols-2 gap-3" style={{ minHeight: '260px' }}>
            <EdgePanel
              confidence={confidence}
              peakMicLevel={peakMicLevel}
              confidenceHistory={confidenceHistory}
              state={assistantState}
            />
            <CloudPanel
              state={assistantState}
              vadScore={vadScore}
              silenceSeconds={silenceSeconds}
              latestLatency={latestLatency}
              latencyHistory={latencyHistory}
            />
          </div>

          {/* Transcript */}
          <div className="flex-1 min-h-0 flex flex-col">
            <TranscriptPanel
              finals={finals}
              partialText={partialText}
              state={assistantState}
              onClear={() => { setFinals([]); setPartialText(''); }}
            />
          </div>

          {/* Footer row */}
          <div className="shrink-0 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2">
            {/* System tag */}
            <div className="flex items-center gap-2.5" style={{ color: 'rgba(0,0,0,0.57)' }}>
              <div className="flex items-center gap-1.5">
                <div className="w-1 h-1 rounded-full" style={{ background: '#34c759' }} />
                <span className="text-[14px] font-medium">Edge wake-word</span>
              </div>
              <span className="text-[14px]">+</span>
              <div className="flex items-center gap-1.5">
                <div className="w-1 h-1 rounded-full" style={{ background: '#0071e3' }} />
                <span className="text-[14px] font-medium">Moonshine ASR</span>
              </div>
              <span className="text-[14px]" style={{ color: 'rgba(0,0,0,0.2)' }}>·</span>
              <span className="text-[14px]">Hey Tara</span>
            </div>

            {/* Wire inspector */}
            <div className="w-full sm:w-auto sm:min-w-[340px]">
              <RawEventsTicker events={rawEvents} onClear={() => setRawEvents([])} />
            </div>
          </div>

        </div>
      </main>
    </div>
  );
}
