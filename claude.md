# HEY TARA — Complete Project Documentation for PPT

> **Purpose of this file:** This is a complete, reference-grade description of every aspect of the "Hey Tara" project. Use this file as the sole input when generating a presentation with Claude. You do NOT need to open any other file in this repository.

---

## TABLE OF CONTENTS
1. Project Overview & Problem Statement
2. High-Level Architecture (End-to-End)
3. The AI Wake-Word Model
4. Dataset Pipeline — Synthetic Data Generation
5. Audio Augmentation System
6. Training Pipeline (train.py)
7. Feature Extraction (features.py)
8. Live Demo System (live_demo.py)
9. Cloud ASR Server (asr_server.py)
10. ASR Streaming Client (asr_client.py)
11. Voice Activity Detection (vad.py)
12. UI Dashboard Broadcaster (ui_broadcast.py)
13. Noise Generation (noise_gen.py)
14. Real Recordings & Evaluation
15. False-Accept Soak Test
16. Key Results & Performance Numbers
17. Tech Stack Decisions (Why Each Was Chosen)
18. Known Limitations (To State Proactively)
19. Roadmap & What's Next
20. Full File-by-File Reference
21. IAMH Website Project (Companion Web App)
22. FAQ — Likely Judge Questions & Answers

---

## 1. Project Overview & Problem Statement

### What is the Problem?
Voice-controlled IoT devices (smart speakers, home automation, assistants) that send **every second of microphone audio to the cloud** suffer from three serious problems:
- **Slow**: Every request must first establish a connection, upload audio, then wait for a response.
- **Expensive**: Continuously streaming audio consumes significant bandwidth and cloud compute cost.
- **Privacy risk**: The microphone is always streaming somewhere — even ambient conversations get uploaded.

### What is Our Solution?
A **Hybrid Edge-Cloud Voice Assistant** with a custom wake word: **"Hey Tara"**.

The core idea:
1. A **tiny, cheap AI model** runs entirely **on the local device (edge)** — using almost no power, no internet, no cloud.
2. It does nothing but listen for ONE specific phrase: "Hey Tara."
3. The moment it hears "Hey Tara," it wakes up the **cloud** (Moonshine ASR) and streams audio to it.
4. **The cloud never hears anything before the wake word** — complete privacy until explicitly triggered.

### Why "Hey Tara"?
- "Tara" is a name used across multiple cultures (Indian/Sanskrit: meaning "star"; Western English; Irish name).
- The name is phonetically rich and distinct from most common English words.
- The team chose it deliberately to reflect an India-relevant, multi-cultural identity for the assistant.

---

## 2. High-Level Architecture (End-to-End)

```
[Microphone]
     |
     v
[Edge: wake-word model, TinyWakeWordNet — always running, ~0.64ms inference]
     |
     | (hears "Hey Tara" with 2 consecutive confident detections)
     |
     v
[Pre-roll buffer: 1.5 seconds of audio BEFORE detection, already captured]
     |
     v
[Silero VAD: trims leading silence/room noise from pre-roll]
     |
     v
[Persistent WebSocket to ASR Server (already open — no handshake delay)]
     |
     v
[Cloud: Moonshine ASR transcribes the command that follows "Hey Tara"]
     |
     v
[Live Browser Dashboard: shows confidence, VAD state, transcript, latency in real time]
```

### Process Breakdown (all on ONE laptop for the demo):

| Process | Script File | Role |
|---------|-------------|------|
| Edge wake-word detector | scripts/live_demo.py | Mic capture, sliding window inference, wake-word trigger |
| Cloud ASR server | scripts/asr_server.py | Receives streamed PCM16 audio, transcribes with Moonshine |
| Dashboard broadcaster | scripts/ui_broadcast.py | Pushes live status/transcript to a browser tab in real time |

---

## 3. The AI Wake-Word Model

### Model Class: TinyWakeWordNet (in scripts/model.py)

**Architecture:** Depthwise-Separable CNN (DS-CNN) — same family as Google's "Hello Edge" reference models, designed specifically for on-device keyword spotting.

**Input:** (batch, 1, 40, n_frames) — a log-mel spectrogram with 40 mel bins.
**Output:** (batch,) — a single raw logit. Apply sigmoid to get P(wake word).

### Network Layers:
```
Stem:
  Conv2d(1, 24, kernel=(10,4), stride=(2,2)) → BatchNorm2d → ReLU

Block 1 (DepthwiseSeparableConv, stride=2):
  Depthwise Conv2d(24->24, 3x3, groups=24) → BN → ReLU
  Pointwise Conv2d(24->48, 1x1) → BN → ReLU

Block 2 (DepthwiseSeparableConv, stride=1):
  Depthwise Conv2d(48->48, 3x3, groups=48) → BN → ReLU
  Pointwise Conv2d(48->48, 1x1) → BN → ReLU

Block 3 (DepthwiseSeparableConv, stride=2):
  Depthwise Conv2d(48->48, 3x3, groups=48) → BN → ReLU
  Pointwise Conv2d(48->72, 1x1) → BN → ReLU

Global Average Pool → Flatten
Linear(72, 1) → scalar logit
```

### Why Depthwise-Separable?
A standard Conv2d (in_ch x out_ch x kH x kW) is replaced by:
- Depthwise conv: per-channel filtering (kH x kW per channel)
- Pointwise conv: 1x1 across channels

This reduces multiply-accumulate operations by roughly 8-9x compared to full convolution. Specifically designed for inference on microcontrollers and low-power edge hardware.

### Size Statistics:
| Metric | Value |
|--------|-------|
| Total parameters | **9,649** |
| Model file size | **37.7 KB** (float32 .pt) |
| Hardware budget | 256 KB (well under limit, even before int8 quantization) |
| Inference latency | **0.64 ms** per 2-second window (CPU only, no GPU) |

---

## 4. Dataset Pipeline — Synthetic Data Generation

### Script: scripts/generate_dataset.py
### Configuration: scripts/config.py

The model is trained on a **synthetic dataset** generated entirely from open-source Text-to-Speech voices — no API cost, no proprietary data.

### 4.1 What Gets Generated?

#### POSITIVES (Label = 1):
Only the full phrase "Hey Tara" in all pronunciation variants:
- "Hey Tara", "Hey, Tara", "Hey Tara.", "Hey Tara!", "Hey Tara?"
- "Hey Tarra", "Hey, Tarra", "Hey Tarra.", "Hey Tarra!", "Hey Tarra?"
- "Hey Terra", "Hey, Terra", "Hey Terra.", "Hey Terra!", "Hey Terra?"

**Why three spellings?** The name "Tara" has 3 common pronunciations:
- /ah/ — "TAH-ruh" (UK English + traditional Hindi/Sanskrit pronunciation of the name meaning "star"). This is espeak-ng's default.
- /æ/ — "TARR-uh" (US English). Triggered by spelling "Tarra" (double consonant forces short vowel).
- /e/ — "TAIR-uh". Triggered by spelling "Terra".

The team decided to **accept all three as positives** to be inclusive of Indian and international users.

#### NEGATIVES — 4 Categories:

**1. Confusable (Hard Negatives) — MOST IMPORTANT:**
```
"Hey Sara", "Hey Kara", "Hey Zara", "Hey Dara", "Hey Lara", "Hey Farah"
"Hey Tarot", "Hey Tiara", "Hey Tamra", "Hey Tanya", "Hey Tata"
"Hey Tyra", "Hey Tora", "Hey Tarun", "Hey Tarzan", "Hey Tarragon"
"Hey Tarp", "Hey Target", "Hey Tarmac", "Hey Tardy"
"Sahara", "Mascara", "Hey Barbara", "Hey guitar"
"The Sahara desert", "She wore mascara", "Play the guitar"
"Okay Tara", "Hi Tara", "A Tara", etc.
"Tara", "Tarra", "Terra", "Hey", "Hey there", "Hey Tar"
```

These teach the model a **tight decision boundary** — not just "speech vs. silence" but "THIS specific phrase vs. everything phonetically similar."

**2. Other Wake-words:**
"Hey Google", "OK Google", "Hey Siri", "Alexa", "Hey Cortana", "Computer", "Hey Bixby", "Jarvis", "Hey Assistant", "Wake up", "Hello Google", "Hey Alexa"

**3. General Speech Negatives:**
45 everyday sentences covering common command patterns, weather queries, numbers, days, months — broad phoneme and vocabulary coverage.

**4. Noise / Silence Negatives:**
Programmatically generated: white, pink, brown, babble, hum, silence.

### 4.2 Voice Roster — 84 Distinct Voices

| Voice Source | Description | Count |
|--------------|-------------|-------|
| en_US-amy-medium | US English, female | 1 |
| en_US-ryan-medium | US English, male | 1 |
| en_GB-alan-medium | British English, male | 1 |
| en_GB-southern_english_female-low | British English, female | 1 |
| en_US-kusal-medium | Sri Lankan-influenced English | 1 |
| en_US-libritts_r-medium | Multi-speaker, 904 speakers (75 randomly sampled) | 75 |
| en_US-l2arctic-medium | Real non-native speakers: 4 Hindi-L1 (ASI, RRBI, SVBI, TNI) | 4 |
| **Total** | | **84 distinct voices** |

**Why every voice produces BOTH positives AND negatives:**
If some voices only appeared in one class, the model could learn "this speaker = positive" as a shortcut. That shortcut works on synthetic data and **fails completely** on new real voices.

**Indian Accent Coverage:**
The 4 L2-ARCTIC Hindi-L1 speakers (ASI, RRBI, SVBI, TNI) are always included because they are the only genuinely Indian-accented voices in Piper's public voice set.

### 4.3 Phrase-List Hygiene (Automatic Safeguards)

1. **Sound-alike dedup**: Phrases that phonemize identically are collapsed (e.g., "Hey Cara" == "Hey Kara"). Text variety ≠ acoustic variety.

2. **Label-poisoning check**: Generation ABORTS if any negative phrase phonemizes identically to a positive. This would teach the model to reject its own wake word.

### 4.4 Fixed Duration + Random Placement
Every clip = exactly **2.0 seconds**. Content placed at a **random offset** (not always centered).

**Why fixed duration?** Without this, a duration-only classifier scored **61%** — classifying "short clip = positive" vs "long clip = negative" without listening to the audio.

**Why random placement?** A model trained on centered clips learns "wake word lives in the middle." In streaming inference, the sliding window sees the phrase at any offset.

---

## 5. Audio Augmentation System

### Script: scripts/augment.py (numpy/scipy only — no ffmpeg, no sox)

**15% of clips skip augmentation** (CLEAN_PROB = 0.15) — reference clips.
**85% receive randomized augmentation:**

| Augmentation | Parameter Range | Purpose |
|--------------|----------------|---------|
| Speed perturbation | 0.85x – 1.15x | Different speaking rates |
| Independent pitch shift | ±2.5 semitones | Decouple pitch from rate |
| Gain / loudness | -14 dB to +8 dB | Real-world level variation |
| Additive noise (SNR) | 5, 8, 12, 16, 20, 30 dB | Room/mic noise |
| Synthetic reverb | prob=0.35, decay 0.1–0.6s | Real room acoustics |
| Low-pass filter | prob=0.15, cutoff 1500–4000 Hz | Far-field/muffled mic |
| Soft clipping (tanh) | prob=0.05 | Cheap-mic distortion |
| Leading/trailing silence | 0–400 ms each end | Random context padding |

**Why SNR floor is 5 dB (not 0 dB):**
Hard negatives like "Hey Kara", "Hey Dara", "Hey Zara" differ from "Hey Tara" by ONE consonant. At 0 dB SNR they become acoustically identical to a positive while labeled negative — pure label noise.

**Key technical trick — rational resampling:**
Speed/pitch shifts use Fraction.limit_denominator(1000) to approximate the resample ratio. Without this, scipy.signal.resample_poly builds a filter with tens of thousands of taps, making generation ~100x slower.

---

## 6. Training Pipeline (train.py)

### Script: scripts/train.py

### 6.1 Training Data Sources
1. **Synthetic dataset** (data/) — ~4,700 clips from the generation pipeline
2. **Real team recordings** (orginal_recording/orginal dataset/member1/ ... member7/) — phone recordings from 7 team members
3. **LibriSpeech real speech negatives** — 3,000 clips from LibriSpeech (public-domain audiobook narration)
4. **Noise negatives** — programmatically generated at training time

### 6.2 THE MOST IMPORTANT FIX — Real Speech Negatives

**The bug:** Every keyword-spotting negative in typical pipelines is either silence, synthetic noise, or synthetic TTS. None of that is what a real room actually sounds like.

**Measured consequence BEFORE fix:**
Running the detector continuously against 5.4 hours of real audiobook speech it had never heard:
→ **120.6 false accepts per hour** (~once every 30 seconds)

**The fix:** Sample real speech (LibriSpeech, speaker-disjoint from test set) as negative training examples.

**Result AFTER fix:** **10.3 false accepts per hour** (~once every 6 minutes)
**Impact: ~12x reduction in false-accept rate**, measured on speakers the fix never saw.

### 6.3 Speaker-Disjoint Real Speech Sampling
The function build_real_speech_negatives():
- Shuffles LibriSpeech speaker directories and splits 80/20 (train/held-out)
- Only training-pool speakers contribute clips
- Held-out speakers are used to verify the fix works on unseen speakers

### 6.4 Partial-Word Negatives
**Insight:** In streaming inference, the sliding window often sees only PARTIAL words — "Hey Ta..." or "...ara" — not the complete phrase. Before this fix, the model produced confidence 0.62–0.66 even on fragments with only 13–40% of the phrase.

**Fix:** make_partial_crop() shifts a complete utterance so most of it falls OUTSIDE the 2-second window, leaving only a fragment at the edge. These fragments are labeled NEGATIVE. This trains the detector to require the WHOLE phrase before firing.

### 6.5 SpecAugment Regularization
spec_augment() randomly zeroes out small time/frequency bands during training:
- 1 frequency mask (up to 6 mel bins wide)
- 1 time mask (up to 20 frames wide)

Forces the model not to over-rely on any single narrow time/frequency region — critical with only 7 real speakers.

### 6.6 Two Honest Accuracy Numbers (LOSO)

| Metric | What It Measures |
|--------|-----------------|
| demo-voice fit (real_eval_*) | Deployed checkpoint trained on ALL 7 real speakers. "Does it work for people it has heard?" |
| LOSO (loso_*) | 7 separate throwaway models, each with ONE member held out. "Does it work for someone NEW?" |

**Results:**
| | Demo-voice fit | LOSO (unseen speaker) |
|-|---------------|----------------------|
| Accuracy | **98.6%** | ~71–75% |
| False-reject rate | 1.9% | ~30% |
| False-accept rate | 0.9% | ~21–35% |

The team **proactively reports the LOSO number** — volunteering the weaker honest number before a judge finds it.

### 6.7 Training Loop Details
- **Loss:** Binary Cross-Entropy with logits (BCEWithLogitsLoss)
- **Optimizer:** Adam, lr = 1e-3
- **Batch size:** 64
- **Early stopping:** Keeps epoch with best validation accuracy
- **Checkpoint:** scripts/wakeword_model.pt (includes state dict, feature mean/std, recommended_threshold, n_mels, target_sr, clip_duration_s, all eval metrics)

---

## 7. Feature Extraction (features.py)

### Script: scripts/features.py (numpy/scipy only — no librosa, no torchaudio)

### 7.1 Log-Mel Spectrogram Parameters
| Parameter | Value | Meaning |
|-----------|-------|---------|
| TARGET_SAMPLE_RATE | 16,000 Hz | Standard for KWS (Piper outputs 22,050 Hz, resampled down) |
| N_FFT | 400 | = 25ms at 16kHz |
| HOP_LENGTH | 160 | = 10ms at 16kHz (90% overlap) |
| N_MELS | 40 | Standard for low-latency KWS models |
| Clip duration | 2.0 s | Fixed for all clips |
| Output shape | (40, 198) | 40 mel bins x 198 time frames per 2-second clip |

### 7.2 Incremental Feature Extraction — IncrementalMelExtractor

**Problem:** live_demo.py used to recompute the log-mel spectrogram for the full 2-second window every 200ms poll — recomputing ~198 frames to get only ~20 new frames. 10x more work than necessary.

**Solution:** IncrementalMelExtractor keeps a rolling buffer of mel FRAMES instead of raw audio. On each poll:
1. Push only new audio chunks
2. Compute ONLY frames covering newly-arrived samples
3. Slide the frame window

**Result:** 2.6–3.5x less CPU work per poll vs. naive recompute.
**Verified:** Numerically identical to the naive version (max abs diff ~2e-6 float32).

**Implementation:** Uses numpy stride tricks to build all new overlapping frames as one zero-copy view, then FFTs them in a single batched np.fft.rfft() call — vectorized instead of a Python loop.

---

## 8. Live Demo System (live_demo.py)

### Script: scripts/live_demo.py
### This is the hackathon demo script.

### 8.1 Command-Line Flags

```bash
# Run with defaults:
.venv/bin/python3 scripts/live_demo.py

# With ASR streaming (full pipeline):
.venv/bin/python3 scripts/live_demo.py --stream-to-asr ws://127.0.0.1:8765

# More sensitive detection:
.venv/bin/python3 scripts/live_demo.py --threshold 0.4

# List available microphones:
.venv/bin/python3 scripts/live_demo.py --list-devices
```

| Flag | Default | Description |
|------|---------|-------------|
| --threshold | from checkpoint | Detection confidence threshold |
| --confirm-polls | 2 | Consecutive polls above threshold before firing |
| --cooldown-s | 1.5 | Minimum seconds between detections |
| --poll-interval-s | 0.2 | How often (seconds) to run inference |
| --stream-to-asr | off | WebSocket URL of cloud ASR server |
| --preroll-s | 1.5 | Seconds of pre-trigger audio to send to ASR |
| --asr-endpoint-silence-s | 0.7 | Stop streaming after this much trailing silence |
| --asr-vad-threshold | 0.5 | Silero VAD cutoff for gating audio to ASR |
| --dashboard-port | 8766 | Browser dashboard WebSocket port |
| --debug-dir | off | Save windows with confidence >0.2 as .wav files |

### 8.2 Main Inference Loop
1. Mic callback (sounddevice.InputStream) appends audio chunks to ring buffer + pending_chunks list
2. Every 200ms, the main thread:
   a. Drains pending_chunks → pushes to IncrementalMelExtractor (only new audio)
   b. Reads ring buffer for peak amplitude only (cheap max())
   c. Applies loudness gain in log-mel space (not raw waveform — avoids refeaturizing)
   d. Z-score normalizes features using stored training mean/std
   e. Runs TinyWakeWordNet inference
   f. Updates consecutive-hit counter

### 8.3 Consecutive Confirmation
Requires 2 consecutive polls above threshold before declaring detection.
- Genuine utterance: several consecutive high-confidence polls as it slides through the 2-second window
- Isolated noise spike: typically does NOT produce consecutive hits
- Adds ~200ms to minimum detection time but significantly reduces false positives

### 8.4 Loudness Normalization
**Why peak-based (not RMS-based):**
The earlier RMS version had no ceiling. A quiet room with even one sharp transient (click, breath, chair creak) could get amplified to/past 0 dBFS. The model scores higher on louder noise (0.397 at -50 dBFS → 0.548 at -30 dBFS), so aggressively amplifying quiet rooms causes false triggers.

**Current:** Peak-based (target_peak=0.7, max_gain_db=12). Skips near-silent windows (floor_peak=0.08).

**Applied in log-mel space** via apply_gain_to_log_mel() — mathematically exact:
raw_mel = exp(log_mel) - epsilon → new_log_mel = log(raw_mel * gain^2 + epsilon)

**Why not "just add a constant in log-space"?**
The additive approximation breaks near epsilon for near-silent frames, occasionally flipping the detector's decision near the operating threshold. The exact version has zero decision mismatches (verified against 60 real clips).

### 8.5 Pre-Roll Buffer
A continuous rolling 1.5-second buffer of raw audio is kept at all times. On detection:
1. Extract pre-roll audio from buffer
2. Silero VAD trims leading silence/room noise
3. This pre-trimmed audio is sent as the VERY FIRST FRAME to the ASR server

This is what makes ~205–225ms latency real — not measuring from a cold start.

---

## 9. Cloud ASR Server (asr_server.py)

### Script: scripts/asr_server.py
### ASR Engine: Moonshine (open-source, MIT license)

Moonshine is purpose-built for real-time, low-latency streaming transcription. No GPU required. Uses "Ergodic Streaming Encoder" architecture.

This process represents "the cloud" in the hybrid architecture. In the demo it runs locally on localhost, but pointing it at a remote server is a one-line change (just change the URL).

### 9.1 Wire Protocol (One persistent connection, many utterances)

**Client → Server:**
```
text:   {"event": "utt_start", "utt_id": N, "client_ts": T}
binary: raw PCM16LE mono 16kHz bytes (pre-roll first, then live audio)
text:   {"event": "utt_end", "utt_id": N, "client_ts": T}
```

**Server → Client:**
```
text: {"event": "first_frame_ack", "utt_id": N, "server_recv_ts": T}
text: {"event": "partial", "utt_id": N, "line_id": I, "text": "..."}
text: {"event": "final", "utt_id": N, "line_id": I, "text": "...", "moonshine_latency_ms": M}
```

**first_frame_ack is the latency measurement anchor:**
Sent BEFORE handing audio to Moonshine — decode time never leaks into the measurement. This makes the ~205–225ms claim honest.

### 9.2 Why Moonshine Over Alternatives?
| Alternative | Problem |
|-------------|---------|
| Vosk | Older Kaldi-based models, worse accuracy |
| Whisper | Fixed windows — NOT true low-latency streaming |
| Kyutai STT | Needs GPU; streaming only in Rust runtime (not Python) |
| Google Speech API | **PROPRIETARY — prohibited by hackathon rules** |

---

## 10. ASR Streaming Client (asr_client.py)

### Script: scripts/asr_client.py

**Design principle: Minimize demo-day failure surface.**

**Why synchronous WebSocket (not asyncio)?**
The websockets.sync.client API gives a persistent connection and thread-safe sends with none of asyncio's bookkeeping.

**Thread model:**
- Two threads share ONE connection
- Main thread SENDS (under a lock)
- Dedicated background thread ONLY READS (blocking for-loop)
- websockets sync client supports this by design

**One connection for the whole session:**
Opened ONCE at startup, stays open across every detection. Opening fresh per detection would put a TCP handshake in the latency-critical path.

### Key Methods:
```python
asr.start()                           # Open connection, start receive thread
asr.trigger_utterance(preroll, ts)    # Send utt_start + pre-roll immediately
asr.feed_audio(chunk)                 # Feed live audio while utterance is active
asr.end_utterance()                   # Send utt_end, reuse for next detection
asr.stop()                            # Close gracefully
```

---

## 11. Voice Activity Detection (vad.py)

### Script: scripts/vad.py
### Model: Silero VAD (MIT license, ~1MB ONNX)

**Problem it solves:**
A peak/energy threshold can't tell "someone talking quietly" from "the fan is on." This was the direct cause of **hallucinated transcripts** — Moonshine has no "no speech" output, so it transcribes whatever it receives, including background noise.

**What Silero VAD does:**
A real neural speech/non-speech classifier. Processes 32ms frames at 16kHz. Outputs speech probability (0–1). Used for:
1. Deciding WHAT audio to send to ASR (silence/background held back — never hallucinated)
2. Trimming leading silence off the pre-roll buffer
3. Endpointing — closing utterance after prolonged silence post-speech

**Implementation note:**
Talks directly to Silero's ONNX file via onnxruntime, bypassing the silero_vad pip package's Python wrapper (which imports torchaudio unconditionally, and that build fails to load due to a torch/torchaudio version mismatch). importlib.util.find_spec() locates the .onnx file without executing the broken __init__.py.

### VAD Parameters:
| Parameter | Value | Description |
|-----------|-------|-------------|
| FRAME_SAMPLES | 512 | Fixed: exactly 32ms at 16kHz |
| CONTEXT_SAMPLES | 64 | Preceding audio stitched onto each frame (REQUIRED — omitting produces near-zero probabilities) |
| DEFAULT_THRESHOLD | 0.5 | Silero's documented operating point |

---

## 12. UI Dashboard Broadcaster (ui_broadcast.py)

### Script: scripts/ui_broadcast.py

A tiny WebSocket SERVER (live_demo.py acts as the server here, opposite of the ASR client). Broadcasts real-time status to any browser dashboard tab.

**Read-only by design** — a dashboard should never feed commands back into the detection pipeline.
**Binds to 0.0.0.0** (not 127.0.0.1) so a dashboard on a different laptop on the same WiFi can connect.

### Message Schema:
```json
{"type": "listening", "conf": 0.02, "peak": 0.01}
{"type": "detected", "confidence": 0.96}
{"type": "streaming", "conf": 0.56, "vad": 0.87, "silence_s": 0.0}
{"type": "utterance_end"}
{"type": "asr_partial", "text": "what is the weath..."}
{"type": "asr_final", "text": "what is the weather", "moonshine_latency_ms": 349}
{"type": "latency", "ms": 212.0}
```

---

## 13. Noise Generation (noise_gen.py)

### Script: scripts/noise_gen.py (numpy/scipy only)

6 types of synthetic noise, used as both standalone negatives and augmentation:

| Type | Description | Implementation |
|------|-------------|----------------|
| white | Flat spectrum | Gaussian random |
| pink | 1/f spectrum | Spectral shaping in frequency domain |
| brown | 1/f^2 spectrum | Deeper, rumbling noise |
| babble | Crowd murmur | 6 band-limited noise voices summed (150-400 Hz to 1500-3200 Hz range per voice) |
| hum | Mains hum | Fundamental (50/60 Hz) + 2nd + 3rd harmonic + light Gaussian noise |
| silence | Near-silent room tone | -70 to -40 dBFS + random spectral tint + optional faint hum |

**Why silence is not digital zero:**
Real rooms are never quiet the same way twice. A single fixed Gaussian floor creates 40 copies of one sound — teaches the model nothing beyond that one noise shape.

---

## 14. Real Recordings & Evaluation

### Location: orginal_recording/orginal dataset/
### Script: scripts/ingest_real_recordings.py

**7 team members** each contributed phone recordings in two categories:
- Positive folder: full wake phrase "hey tara" (folder naming varies per member: "hey tara", "Hey_Tara", "heytara", etc.)
- Negative folder: "tara_alone" — the bare name with no "Hey"

**The ingest script is READ-ONLY** — never copies, moves, or modifies source audio.
- Auto-discovers member*/ folders (new contributors picked up without editing the script)
- Auto-flags clipped files (a run of >50 consecutive samples at the digital ceiling)
- Output: orginal_recording/manifest.csv — all rows with split = "real_eval"

### CRITICAL: Keep this set held-out FOREVER.
**Never train on it. Never fold it into synthetic train/val/test.**
It exists to catch what the synthetic test split CANNOT (shared vocoder artifacts). Filter to split=="real_eval" for evaluation only and skip rows with non-empty quality_flag (currently 2 clipped files from one member).

---

## 15. False-Accept Soak Test

### Script: scripts/false_accept_soak_test.py

**What it measures:** Running the deployed detector continuously against hours of real, never-heard speech and counting every false trigger.

**Why this matters more than test-set accuracy:**
Every other accuracy number comes from short isolated clips — none answer: "Left running next to a podcast or TV, how many times does it go off by mistake?"

**Test corpus:** LibriSpeech (real audiobook narration, public domain) — 5.4 hours total.

**Uses EXACT SAME inference path as live_demo.py:** IncrementalMelExtractor, loudness_gain/apply_gain_to_log_mel, confirm-polls, cooldown. A false-accept rate on a different code path would mean nothing.

### Results:
| Condition | False Accepts/Hour |
|-----------|-------------------|
| BEFORE fix (no real speech negatives) | **120.6 / hour** (~once every 30 seconds) |
| AFTER fix (LibriSpeech negatives added) | **10.3 / hour** (~once every 6 minutes) |
| Improvement | **~12x reduction** |

---

## 16. Key Results & Performance Numbers (ALL MEASURED, NOT ESTIMATED)

| Metric | Value | How Measured |
|--------|-------|-------------|
| Model size | **37.7 KB** (9,649 params), float32 | count_params() |
| Inference latency | **0.64 ms** per 2-second window | Timed on CPU |
| Incremental feature speedup | **2.6–3.5x** less CPU vs. naive | Bit-for-bit verified |
| Keyword-end → cloud-received latency | **~205–225 ms** | first_frame_ack timestamp vs. first_hit_ts |
| Demo-voice fit accuracy | **98.6%** | Real recordings, all 7 members in training |
| Demo-voice FRR (miss rate) | **1.9%** | |
| Demo-voice FAR (false trigger rate) | **0.9%** | |
| LOSO accuracy (unseen speaker) | **~71–75%** | 7-fold leave-one-speaker-out CV |
| False accepts on 5.4h real speech (before fix) | **120.6 / hour** | Soak test, LibriSpeech |
| False accepts on 5.4h real speech (after fix) | **10.3 / hour** | Same soak test |
| Improvement factor | **~12x** | Measured on unseen speakers |

---

## 17. Tech Stack Decisions (Why Each Was Chosen)

| Component | Choice | Reason |
|-----------|--------|--------|
| Edge model framework | **PyTorch (CPU)** | Custom CNN — trivial to export to TFLite/ONNX later |
| TTS voice data | **Piper TTS** | Open-source, no API cost, 100s of voices across accents |
| Cloud ASR | **Moonshine** (MIT) | Purpose-built streaming ASR; no GPU; better than Vosk |
| VAD | **Silero VAD** (MIT, ONNX, ~1MB) | Real neural speech classifier — fixed our hallucinated-transcript problem |
| Streaming transport | **WebSockets + raw PCM16** | Connection never in latency-critical path; simplest reliable option |
| Dashboard | **Browser app via local WebSocket** | Zero backend work; connects to read-only status feed |
| Hackathon compliance | **100% open-source** | PyTorch, Piper, Moonshine (MIT), Silero VAD (MIT), websockets |

**Why ONE laptop, not two:**
A second machine adds a live WiFi dependency. Venue networks often have client isolation (devices can't talk to each other). We verified everything on one machine. Visual impact is not worth an untestable WiFi dependency.

---

## 18. Known Limitations (State These Proactively)

The team states these BEFORE being asked — it looks far better than having a judge discover them.

1. **Unseen-speaker accuracy (~71–75%) is the weak point.** 7 real speakers is not enough to generalize perfectly. Fix: more speakers. Data quantity problem, not a broken model.

2. **One team member's data is incomplete** (13 clips vs. 50–75 for others) — makes that fold noisier in LOSO.

3. **ASR transcript quality on bare "Hey Tara"** is poor — Moonshine has almost no context on a two-word utterance. Works much better with a trailing command: "Hey Tara, what's the weather?"

4. **Not yet quantized to int8 or deployed on real hardware** (ESP32/Raspberry Pi). Model is small enough (37.7 KB float32) — not measured on-device yet.

5. **No dedicated final-checkpoint soak test.** Soak testing was done at intermediate points during iteration.

6. **Indian-English accent coverage is thin.** Only Hindi-L1 (ASI, RRBI, SVBI, TNI). Tamil, Telugu, Bengali, Punjabi-accented English are not represented.

7. **Noise beds are synthetic.** Production robustness needs real noise from MUSAN, DEMAND, ESC-50.

8. **75 of 80 voices from one TTS model** (libritts_r). Model can partly learn "Piper artifacts + hey-tara phonemes = positive" — vocoder memorization that the speaker-disjoint split cannot prevent.

---

## 19. Roadmap & What's Next

1. Record more real speakers (target 15–20+) to close the generalization gap.
2. Quantize to int8 and deploy to ESP32/Raspberry Pi for hardware-constrained measurement.
3. Add Moonshine per-word confidence scoring as a second transcript cleanup layer on top of VAD.
4. Opus audio compression to cut streaming data overhead.
5. Broader false-accept testing: TV audio, music with vocals, multiple accents.
6. Add real noise databases (MUSAN, DEMAND, ESC-50) as augmentation sources.
7. Add Common Voice / LibriSpeech segments as extra general-speech negatives.

---

## 20. Full File-by-File Reference

```
/home/pai/sih/
|
|-- README.md                        # Technical reference (dataset design, caveats)
|-- TEAM_BRIEFING.md                 # Pitch document (results, FAQ, talking points)
|-- requirements.txt                 # Python dependencies
|-- .venv/                           # Python virtual environment
|
|-- scripts/
|   |-- config.py                    # *** Master config: voices, phrases, augmentation ranges
|   |-- model.py                     # *** TinyWakeWordNet architecture (9,649 params)
|   |-- features.py                  # *** Log-mel spectrogram + IncrementalMelExtractor
|   |-- augment.py                   # *** Audio augmentation pipeline (numpy/scipy)
|   |-- noise_gen.py                 # *** Synthetic noise generators (6 types)
|   |-- tts_engine.py                # Piper TTS wrapper
|   |-- generate_dataset.py          # *** Full synthetic dataset generation pipeline
|   |-- validate_dataset.py          # Dataset integrity checks
|   |-- train.py                     # *** Full training pipeline (LOSO, real-speech negatives)
|   |-- infer.py                     # Simple inference on a single audio file
|   |-- live_demo.py                 # *** Live microphone demo — THE HACKATHON SCRIPT
|   |-- asr_server.py                # *** Moonshine WebSocket ASR server
|   |-- asr_client.py                # *** Persistent streaming client for asr_server
|   |-- vad.py                       # *** Silero VAD wrapper
|   |-- ui_broadcast.py              # *** Browser dashboard WebSocket broadcaster
|   |-- false_accept_soak_test.py    # *** Long-running false-accept measurement
|   |-- ingest_real_recordings.py    # Build manifest for team's phone recordings
|   |-- live_record.py               # Utility: record mic clips for team dataset
|   `-- wakeword_model.pt            # *** Trained model checkpoint (37.7 KB)
|
|-- data/
|   |-- manifest.csv                 # Full dataset manifest
|   |-- positive/wake_word/          # ~4,000+ positive .wav clips across 84 voices
|   `-- negative/
|       |-- confusable/              # Phonetically similar phrases
|       |-- other_wakeword/          # "Hey Google", "Alexa", etc.
|       |-- random_speech/           # General sentences
|       |-- noise/                   # white/pink/brown/babble/hum beds
|       `-- silence/                 # Near-silent room-tone clips
|
|-- voices/                          # Downloaded Piper TTS model files (.onnx + .json)
|   |-- en_US-amy-medium.onnx
|   |-- en_US-ryan-medium.onnx
|   |-- en_GB-alan-medium.onnx
|   |-- en_GB-southern_english_female-low.onnx
|   |-- en_US-kusal-medium.onnx
|   |-- en_US-libritts_r-medium.onnx   # 904-speaker model
|   `-- en_US-l2arctic-medium.onnx     # 4 Hindi-L1 speakers always included
|
|-- orginal_recording/
|   |-- manifest.csv                 # Real-eval manifest (never used for training)
|   `-- orginal dataset/
|       |-- member1/ ... member7/    # Phone recordings from 7 team members
|
`-- models/
    `-- download.moonshine.ai/       # Cached Moonshine ASR model files
```

### Python Dependencies (requirements.txt):
```
piper-tts         # Text-to-Speech synthesis (ships its own espeak-ng)
numpy             # Core array operations
scipy             # Signal processing (STFT, filters, resampling)
soundfile         # Read/write audio files
sounddevice       # Real-time microphone access
moonshine-voice   # Moonshine streaming ASR
websockets        # WebSocket server/client
```
*(torch is a transitive dependency of moonshine-voice, used directly for model inference)*
*(onnxruntime is a transitive dependency of silero-vad, used directly in vad.py)*

---

## 21. IAMH Website Project (Companion Web App)

This is a SEPARATE project at /home/pai/web/iamh/ — a professional website for the Indian Association for Mental Health (IAMH).

### What is IAMH?
Indian Association for Mental Health — a national umbrella organization for professionals from: Psychiatry, Psychology, Counselling, Social Work, Occupational Therapy, Psychiatric Nursing, Rehabilitation, Special Education, Neurosciences, and Holistic Therapies.

### Tech Stack:
| Layer | Technology |
|-------|-----------|
| Framework | React 19 + Vite 6 |
| Language | TypeScript |
| Styling | Tailwind CSS v4 |
| Routing | React Router v7 |
| Animations | Motion (Framer Motion v12) |
| Icons | Lucide React |
| AI Integration | @google/genai |
| Build | Vite (dev: npm run dev on port 3000) |

### Pages & Routes:
| Route | Component | Purpose |
|-------|-----------|---------|
| / | Home (in App.tsx) | Main IAMH landing page |
| /events/school-mental-health-conclave | SchoolConclavePage | School Mental Health Conclave 2026 |

### Main Landing Page Sections (in order):
1. Navbar — Sticky navigation
2. Hero — Hero banner
3. EventsAnnouncement — Upcoming event: School Mental Health Conclave 2026
4. About — Profile, Vision, Mission & Specialties (11 disciplines)
5. ValuesAndFeatures — Core values: Compassion, Collaboration, Inclusion, Ethics, Scientific Practice, Excellence
6. CommitteeMembers — Executive committee members
7. Membership — Benefits, "Why Join", live digital membership card generator
8. Contact — Contact details, form, interactive SVG maps
9. Footer — Institutional footer

### School Mental Health Conclave 2026 Page (17 sections):
1. Hero Section
2. Key Event Info Strip
3. About the Conclave
4. Strategic Objectives
5. Program Focus (Understanding the Child Behind the Behaviour)
6. Teacher Training (Behaviour, Emotions, Connections)
7. Positive Classroom Environment (Listen→Understand→Connect→Respond→Support)
8. Teacher Mental Health & Well-being
9. A Mentally Healthy School Ecosystem
10. School Participation Structure (delegation breakdown)
11. Participation Fee Section (Rs.21,000/- per school)
12. Certification & Recognition
13. Collaborating Organizations (IAMH & IMS-BHU Psychiatry CoE)
14. Venue (Tentative Notice Display)
15. Registration Form
16. Contact Section (Official Helpline)
17. Footer

### IAMH Mission (7 items):
1. Promote Mental Health Awareness
2. Encourage Research
3. Advance Mental Health Education
4. Support Advocacy
5. Promote Evidence-Based Practices
6. Foster Professional Collaboration
7. Enhance Community Well-being

---

## 22. FAQ — Likely Judge Questions & Answers

**Q: Why is your unseen-speaker accuracy only ~71-75%, not 95%+?**
Because we're reporting the HONEST number, not the flattering one. With only 7 real speakers, leave-one-out is inherently harder — that's a data quantity problem. The fix is more speakers, not a different architecture. Most teams won't have run this test at all and will show demo-voice-fit numbers without knowing the difference.

**Q: How do you know your false-accept rate is realistic?**
We didn't just claim it — we streamed 5.4 hours of real, continuous audiobook narration (never used in training) through the EXACT SAME live detection pipeline and counted every trigger.

**Q: Why Moonshine instead of Whisper / Vosk / Google Speech API?**
- Vosk: older Kaldi-based models, materially worse accuracy
- Whisper: fixed windows — NOT true low-latency streaming
- Google Speech API: PROPRIETARY AND COMMERCIAL — explicitly prohibited by hackathon rules
- Moonshine: open-source (MIT), purpose-built for real-time streaming, no GPU

**Q: What happens if cloud connection is down?**
The edge detector still fires and shows detection locally regardless of the ASR connection — keyword-spotting is fully independent of the cloud half. Streaming fails gracefully via exception handling in asr_client.py.

**Q: How does the pre-roll buffer avoid clipping the start of the command?**
We keep a continuous rolling 1.5-second buffer at all times. The moment the wake word is confirmed, that buffer — which already covers audio from just before the trigger — is sent as the first frame. The ~205-225ms latency is not from a cold start.

**Q: Why does it take ~200ms to detect, isn't that slow?**
That's a DELIBERATE design choice: we require 2 consecutive confident detections, which suppresses false positives from noise spikes. It's a tunable tradeoff (one flag), not a hard limitation.

**Q: Does this need special hardware?**
The edge model (37.7 KB) targets an ESP32 or Raspberry Pi — that port hasn't been done yet. Tonight's demo runs on a laptop as a stand-in for the edge device.

**Q: What was the hardest technical problem you solved?**
Two things:
1. Loudness normalization for streaming inference — the initial additive-shift approximation in log-mel space was wrong for near-silent frames, occasionally flipping detector decisions near the threshold. Fixed to the mathematically exact version (verified against 60 real clips with zero mismatches).
2. The false-accept fix — the bug was invisible until we specifically went looking for it with real, unheard speech. 120.6 false accepts/hour was hidden behind a 98%+ test-set accuracy number.

**Q: Is everything open-source?**
100% open-source top to bottom: PyTorch, Piper TTS, Moonshine (MIT), Silero VAD (MIT), websockets. No proprietary or commercial voice SDK anywhere in the pipeline.

**Q: How small is the model really?**
9,649 parameters. 37.7 KB as float32. Even int8 quantization (not yet done) would put it at ~10 KB — smaller than most JPEG images and nowhere close to the 256 KB hardware budget.

---

*This document was generated from a complete read of all source files in /home/pai/sih/. Every number cited here is from measured logs or code — not estimates. This file is complete: you do NOT need to open any other file to make a PPT from this.*
