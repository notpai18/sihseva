# Hey Tara — Hybrid Edge-Cloud Voice Assistant
### Team briefing: what we built, why, and how to talk about it

This is the reference document for the whole team. Read it before the pitch so
nobody gets caught off guard in Q&A. Every number in here is a number we
actually measured on this machine, not an estimate — if a judge asks "how do
you know that," the answer is always "we ran it and logged it."

---

## 1. The problem we're solving

Voice-controlled IoT devices that send every second of audio to the cloud are
slow, expensive, and a privacy risk (the mic is always streaming somewhere).
The fix is a hybrid architecture: a tiny, cheap model runs locally and does
nothing but wait for one keyword. The moment it hears that keyword, THEN it
wakes up the expensive, accurate cloud model and streams audio to it. The cloud
never hears anything before the wake word.

Our custom keyword: **"Hey Tara."**

## 2. Architecture, end to end

```
 [Microphone] → [Edge: wake-word model, always running]
                        │
                 hears "Hey Tara"
                        │
                        ▼
        [Pre-roll buffer sent instantly over a
         persistent WebSocket, already open]
                        │
                        ▼
        [Cloud: Moonshine ASR transcribes the
         command that follows]
                        │
                        ▼
        [Live dashboard shows confidence, state,
         and the transcript in real time]
```

Two processes, one laptop for today's demo (see §9 for why we deliberately
did NOT split this across two laptops tonight):

| Process | File | Role |
|---|---|---|
| Edge detector | `scripts/live_demo.py` | Mic capture, wake-word model, voice-activity gating, triggers streaming |
| Cloud ASR | `scripts/asr_server.py` | Receives streamed audio, transcribes with Moonshine |
| Dashboard feed | `scripts/ui_broadcast.py` | Broadcasts live status/transcript to a browser dashboard |

## 3. What we actually built (feature list)

- **Custom keyword-spotting model**, trained from scratch, not a pre-trained
  generic wake word. Depthwise-separable CNN ("Hello Edge" family), **9,649
  parameters, 37.7 KB as float32** — comfortably inside the 256 KB budget with
  room to spare even before quantization.
- **Synthetic + real training pipeline.** ~84 synthetic TTS voices (Piper,
  open-source) across US/UK/South-Asian/Hindi-accented speakers, plus real
  phone recordings from all 7 team members, plus a `hey_other` negative
  category (so the model can't shortcut on the word "Hey" alone).
- **Real-speech negative training data.** Added 3,000 clips sampled from
  LibriSpeech (real audiobook narration, public-domain, never synthetic) as
  negative examples — see §5, this is the single biggest accuracy fix we made.
- **Incremental feature extraction.** The spectrogram computation only
  processes NEW audio each poll instead of recomputing the whole 2-second
  window from scratch every time — verified bit-for-bit identical to the naive
  version, 2.6–3.5x less CPU work depending on poll rate.
- **Honest generalization testing (leave-one-speaker-out).** Instead of
  reporting one accuracy number, we train 7 separate throwaway models, each
  with one team member held out, and test only on the person that model never
  saw. This is the only number in the project that measures "will it work for
  someone new," and we report it even though it's the less flattering number.
- **A real false-accept measurement, not a claim.** We ran the detector
  continuously against **5.4 hours of real human speech it had never heard**
  (LibriSpeech) and counted every false trigger — see §5.
- **Streaming ASR with a pre-roll buffer.** The client keeps a rolling 1.5s
  buffer of audio at all times, so the instant the wake word is confirmed, the
  audio already sent to the cloud covers the moment speech started — this is
  what makes the measured latency small and honest, not an artifact of sending
  audio starting fresh from the trigger.
- **Real voice-activity detection (Silero VAD), not a volume threshold.** Gates
  what actually gets sent to the cloud, so background noise between words
  doesn't get transcribed into garbage. See §5 for the concrete before/after.
- **A live browser dashboard** showing detection state, confidence, VAD
  activity, live transcript, and the measured latency, built for exactly this
  presentation.

## 4. Tech stack, and why each piece was chosen

| Component | Choice | Why |
|---|---|---|
| Edge model framework | PyTorch (CPU) | Small custom CNN, trivial to export to TFLite/ONNX later for real MCU deployment |
| Synthetic voice data | Piper TTS | Open-source, no API cost, many accents/voices for free |
| Cloud ASR | **Moonshine** (small-streaming, MIT license) | Purpose-built for real-time low-latency streaming ASR, no GPU required, meaningfully more accurate than Vosk (the old-Kaldi-based standard choice) |
| Voice-activity detection | **Silero VAD** (MIT, ONNX, ~1MB) | A real neural speech/non-speech classifier, not a volume threshold — this is what fixed our hallucinated-transcript problem |
| Streaming transport | WebSockets, persistent connection, raw PCM16 | Connection setup is never in the latency-critical path; simplest reliable option given the time we had |
| Dashboard | Browser app (built with Google AI Studio) fed by a local WebSocket broadcaster | Zero backend work needed; connects to a read-only status feed |

**Why not Vosk?** Older Kaldi-based acoustic models, noticeably worse
accuracy. **Why not Kyutai STT** (a strong alternative we seriously evaluated)?
It needs a GPU to hit its quoted latency, and its standout feature (built-in
semantic voice detection) only exists in their Rust server, not the Python
path we'd actually use — so we'd carry the extra weight without the benefit.

## 5. The two headline results (lead with these)

### Result A: we found and fixed a real, serious flaw

Every keyword-spotting negative-training example in typical pipelines
(including our first version) is either silence, synthetic noise, or
synthetic TTS speech. **None of that is what a real room actually sounds
like.** We proved this mattered:

| | Before fix | After fix |
|---|---|---|
| False accepts on 5.4 hours of real, unheard speech | **120.6 / hour** (~once every 30 seconds) | **10.3 / hour** (~once every 6 minutes) |

The fix: sample real speech (we used part of the same LibriSpeech corpus,
speaker-disjoint from the test set) as negative training examples. **A ~12x
reduction**, measured on speakers the fix never saw.

### Result B: two honest accuracy numbers, not one flattering one

| | Demo-voice fit (people who recorded training data) | Unseen speaker (leave-one-out) |
|---|---|---|
| Accuracy | 98.6% | ~71–75% |
| False-reject rate | 1.9% | ~30% |
| False-accept rate | 0.9% | ~21–35%* |

\* at a fixed diagnostic threshold of 0.5; the deployed threshold (0.23) is
separately calibrated to minimize combined error.

**We say this out loud, unprompted, in the pitch.** The gap between these two
numbers is expected and explainable (7 real speakers isn't enough data to
generalize perfectly yet — this is a data quantity problem, not a broken
model), and volunteering it before a judge finds it looks far better than
having it discovered in Q&A.

### Supporting numbers

| Metric | Value |
|---|---|
| Model size | 37.7 KB (9,649 params), float32 |
| Inference latency | 0.64 ms per 2-second window |
| Feature extraction | Incremental, 2.6–3.5x less CPU than naive recompute |
| Keyword-end → cloud-received latency | **~205–225 ms**, measured on the real mic path with pre-roll buffering |

## 6. Advantages over the "obvious" approach

A naive team's submission looks like: download a KWS tutorial, swap in a new
keyword, train once, report one training-set accuracy number, done. Ours
differs in exactly the ways that matter for a real deployment:

1. **Speaker-disjoint everything.** Synthetic train/val/test splits never mix
   voices across splits, and neither does the real-recording evaluation.
2. **We measure generalization, not memorization** (leave-one-speaker-out).
3. **We measure the failure mode that actually matters in production**
   (false accepts per hour on real, continuous, never-heard speech) instead of
   trusting a clean test-set number.
4. **We found a real bug in our own methodology, fixed it, and proved the fix**
   with a controlled before/after on data the fix never trained on.
5. **The latency claim is architected to be true**, not just measured once: a
   persistent connection and a pre-roll buffer are what make "low latency"
   real instead of an artifact of a lucky test run.

## 7. Known limitations — say these before you're asked

- **Unseen-speaker accuracy (~71–75%) is still the weak point.** More real
  speakers is the direct fix; we only have 7 today. This is the single most
  important "what's next" answer.
- **One team member's data is incomplete** (13 clips vs. 50–75 for others),
  which makes that speaker's individual fold noisier.
- **ASR transcript quality is poor on a bare "Hey Tara" with nothing after
  it** — Moonshine has almost no linguistic context on a two-word utterance.
  It works much better with a real trailing command ("Hey Tara, what's the
  weather"), which is what the demo should actually say.
- **Not yet quantized to int8 or deployed on real microcontroller hardware**
  (ESP32/Raspberry Pi). The model is small enough (37.7 KB float32) that this
  should be straightforward, but it hasn't been measured on-device yet.
- **No dedicated hours-long soak test on the exact final checkpoint/threshold
  combination** — our soak testing was done at intermediate points during
  iteration, not as one final end-to-end certification run.

## 8. Roadmap / what's next

1. Record more real speakers (target 15–20+) to close the generalization gap.
2. Quantize to int8 and deploy to an actual ESP32/Raspberry Pi for the formal
   hardware-constrained RAM/CPU measurement.
3. Add Moonshine's per-word confidence scoring as a second layer of transcript
   cleanup, on top of the VAD fix.
4. Opus audio compression to cut streaming data overhead further.
5. Broader false-accept testing: TV audio, music with vocals, multiple accents
   and languages, not just audiobook narration.

## 9. Why the demo runs on one laptop, not two

We built a browser dashboard and initially considered running it on a second
laptop for visual impact. We deliberately chose ONE laptop for the actual
pitch: a second machine adds a live WiFi dependency (shared/venue networks
often block device-to-device traffic for security — "client isolation" — and
we can't test that in advance), for a purely cosmetic benefit. Everything
verified end-to-end lives on one machine; that's what we're showing.

## 10. Hackathon rule compliance (say this explicitly if asked)

- **Open-source only**, top to bottom: PyTorch, Piper TTS, Moonshine (MIT),
  Silero VAD (MIT), `websockets`. No proprietary or commercial voice SDK
  anywhere in the pipeline.
- **Custom keyword, trained from scratch** — "Hey Tara" is not a pre-trained
  generic assistant wake word.
- **Model is tiny** — 37.7 KB, nowhere close to the 256 KB ceiling, no
  transformer, no heavy pre-trained backbone.

---

## Appendix: FAQ — likely judge questions and how to answer them

**Q: Why is your unseen-speaker accuracy only ~71–75%, not 95%+?**
A: Because we're reporting the honest number, not the flattering one. With
only 7 real speakers, leave-one-out testing is inherently harder than testing
on speakers the model has seen — that's a data quantity problem, and the fix
is more speakers, not a different architecture. Most teams won't have run
this test at all and will show a demo-voice-fit number without knowing the
difference.

**Q: How do you know your false-accept rate is realistic?**
A: We didn't just claim it — we streamed 5.4 hours of real, continuous
audiobook narration (never used in training) through the live detection
pipeline and counted every trigger. That's the same code path the demo uses,
not a simplified offline classifier.

**Q: Why Moonshine instead of [Whisper / Vosk / Google's Speech API]?**
A: Vosk's Kaldi-based models are meaningfully less accurate. Whisper-family
models process audio in fixed windows, not true low-latency streaming.
Google's API is a proprietary/commercial SDK, which the rules explicitly
prohibit. Moonshine is open-source (MIT), purpose-built for real-time
streaming, and needs no GPU.

**Q: What happens if the wake word is said but the connection to the cloud is
down?**
A: The edge detector still fires and shows the detection locally regardless
of the ASR connection — the keyword-spotting half is fully independent of the
cloud half. [If asked for more detail: streaming would fail gracefully via the
existing exception handling in `asr_client.py`; a retry/reconnect loop is on
the roadmap, not yet built.]

**Q: How does the pre-roll buffer avoid clipping the start of the command?**
A: We keep a continuous rolling 1.5-second buffer of raw audio at all times,
independent of detection state. The moment the wake word is confirmed, that
buffer — which already contains audio from just before the trigger — is sent
as the first frame, before any new audio has even arrived. That's what makes
the ~205–225ms latency number real: it's not measuring from a cold start.

**Q: Why does it take ~200ms to detect, isn't that slow?**
A: That number is mostly a deliberate design choice: we require two
consecutive confident detections before triggering, which suppresses false
positives from a single noise spike. It's a tunable trade-off (one flag), not
a hard limitation — we chose reliability over shaving off ~200ms for tonight.

**Q: Does this need special hardware?**
A: The edge model is small enough (37.7 KB) to target a real microcontroller
like an ESP32 or Raspberry Pi, per the brief's hardware requirement — that
port hasn't been done yet, tonight's demo runs on a laptop as a stand-in for
the edge device, with the cloud ASR representing the actual cloud server on
the other side of that boundary.

**Q: What was the hardest technical problem you solved?**
A: Two things, and they're worth naming specifically because they're the
kind of subtle bug that's easy to miss entirely: (1) our loudness
normalization for streaming inference needed to exactly reproduce what
happens when you renormalize the raw waveform, without actually touching the
raw waveform each time — we initially built an approximation that was wrong
in near-silent regions and occasionally flipped detection decisions near the
threshold; we caught it by testing against 60 real clips before shipping it.
(2) The false-accept fix itself (§5) — the bug was invisible until we
specifically went looking for it with real, unheard speech.
