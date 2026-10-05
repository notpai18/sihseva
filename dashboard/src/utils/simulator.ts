import { WebSocketPayload } from '../types';

export class StreamSimulator {
  private timer: number | null = null;
  private onMessage: (msg: WebSocketPayload) => void;
  private isRunning = false;
  private step = 0;

  private sampleUtterances = [
    {
      partials: [
        "What",
        "What time",
        "What time is it in",
        "What time is it in Tokyo right now?"
      ],
      final: "What time is it in Tokyo right now?",
      moonshineLatency: 312,
      cloudLatency: 198.5
    },
    {
      partials: [
        "Set a",
        "Set a timer for",
        "Set a timer for five minutes",
        "Set a timer for five minutes please"
      ],
      final: "Set a timer for five minutes please.",
      moonshineLatency: 285,
      cloudLatency: 215.0
    },
    {
      partials: [
        "How is",
        "How is the traffic",
        "How is the traffic on the",
        "How is the traffic on the Bay Bridge?"
      ],
      final: "How is the traffic on the Bay Bridge?",
      moonshineLatency: 349,
      cloudLatency: 224.2
    }
  ];

  private currentUtteranceIndex = 0;

  constructor(onMessage: (msg: WebSocketPayload) => void) {
    this.onMessage = onMessage;
  }

  public start() {
    if (this.isRunning) return;
    this.isRunning = true;
    this.step = 0;
    this.scheduleNext();
  }

  public stop() {
    this.isRunning = false;
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }

  public get running(): boolean {
    return this.isRunning;
  }

  private scheduleNext() {
    if (!this.isRunning) return;

    // Simulation sequence:
    // 0..25: idle listening (approx 2.5s)
    // 26: detected wake word
    // 27..45: streaming with speech (VAD ~ 0.85, partials)
    // 46..52: silence trailing before cut (VAD ~ 0.05, silence_s up)
    // 53: asr_final + latency
    // 54: utterance_end
    // loop

    let delay = 100;

    if (this.step < 20) {
      // Idle listening state
      const peak = Math.random() * 0.12 + 0.02;
      const conf = Math.random() * 0.06 + 0.01;
      this.onMessage({ type: 'listening', conf, peak });
      this.step++;
      delay = 120;
    } else if (this.step === 20) {
      // Wake word detected!
      const confidence = 0.94 + Math.random() * 0.05;
      this.onMessage({ type: 'detected', confidence });
      this.step++;
      delay = 500; // Hold celebratory state briefly
    } else if (this.step < 36) {
      // Streaming speech
      const utt = this.sampleUtterances[this.currentUtteranceIndex];
      const vad = 0.72 + Math.random() * 0.25;
      const conf = 0.85 + Math.random() * 0.12;
      this.onMessage({ type: 'streaming', conf, vad, silence_s: 0.0 });

      // Emit partials at key steps
      const partialIndex = Math.floor(((this.step - 21) / 15) * utt.partials.length);
      if (utt.partials[partialIndex]) {
        this.onMessage({ type: 'asr_partial', text: utt.partials[partialIndex] });
      }

      this.step++;
      delay = 180;
    } else if (this.step < 42) {
      // Trailing silence
      const silence_s = Number(((this.step - 36) * 0.25 + 0.2).toFixed(2));
      const vad = Math.random() * 0.12;
      this.onMessage({ type: 'streaming', conf: 0.4, vad, silence_s });
      this.step++;
      delay = 200;
    } else if (this.step === 42) {
      // Final transcript & latency
      const utt = this.sampleUtterances[this.currentUtteranceIndex];
      this.onMessage({
        type: 'asr_final',
        text: utt.final,
        moonshine_latency_ms: utt.moonshineLatency + Math.floor(Math.random() * 30 - 15)
      });
      this.onMessage({
        type: 'latency',
        ms: Number((utt.cloudLatency + (Math.random() * 20 - 10)).toFixed(1))
      });
      this.step++;
      delay = 250;
    } else if (this.step === 43) {
      // Utterance end
      this.onMessage({ type: 'utterance_end' });
      this.currentUtteranceIndex = (this.currentUtteranceIndex + 1) % this.sampleUtterances.length;
      this.step = 0; // back to listening
      delay = 1500; // pause before next cycle
    }

    this.timer = window.setTimeout(() => this.scheduleNext(), delay);
  }
}
