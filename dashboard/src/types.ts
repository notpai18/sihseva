export type WebSocketPayload =
  | { type: 'listening'; conf?: number; peak?: number }
  | { type: 'detected'; confidence?: number; conf?: number }
  | { type: 'streaming'; conf?: number; vad?: number; silence_s?: number }
  | { type: 'utterance_end' }
  | { type: 'asr_partial'; text: string }
  | { type: 'asr_final'; text: string; moonshine_latency_ms?: number }
  | { type: 'latency'; ms: number };

export type AssistantState = 'LISTENING' | 'DETECTED' | 'STREAMING';

export type ConnectionStatus = 'connected' | 'reconnecting' | 'disconnected' | 'connecting';

export interface FinalTranscriptItem {
  id: string;
  timestamp: string;
  text: string;
  moonshineLatencyMs?: number;
}

export interface LatencyRecord {
  id: string;
  timestamp: string;
  ms: number;
}

export interface RawEventItem {
  id: string;
  timestamp: string;
  type: string;
  data: Record<string, unknown>;
}
