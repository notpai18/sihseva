# "Hey Tara" Wake-Word Dataset Generator

Synthetic dataset pipeline for training/evaluating a "Hey Tara" wake-word
(keyword-spotting) detector, in the spirit of "Hey Google" / "Hey Siri".
It generates **positives** (the wake word, many voices/prosodies), **hard
negatives** (phonetically confusable phrases), **other-wake-word negatives**,
**general-speech negatives**, and **pure noise/silence negatives**, all with
randomized audio augmentation, plus a manifest CSV for training.

Everything runs offline after a one-time voice-model download — no ffmpeg,
sox, or espeak system packages required (piper-tts ships its own espeak-ng
phonemizer and ONNX runtime).

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# one-time: download the TTS voices used (~few hundred MB total)
.venv/bin/python3 -m piper.download_voices --download-dir voices \
  en_US-amy-medium en_US-ryan-medium en_GB-alan-medium \
  en_GB-southern_english_female-low en_US-kusal-medium en_US-libritts_r-medium

# generate the dataset (defaults produce ~4,700 clips in a few minutes)
.venv/bin/python3 scripts/generate_dataset.py --clean
```

`--clean` wipes `--out-dir` first. Always use it when regenerating after
changing `scripts/config.py` or the generator itself — the script does not
overwrite-in-place, so without `--clean` a rerun's new files land alongside
whatever old files are still sitting there.

Output lands in `data/`:

```
data/
  manifest.csv
  positive/wake_word/...
  negative/confusable/...       # "Hey Sara", "Hey Tarot", "Hey Tar", ...
  negative/other_wakeword/...   # "Hey Google", "Hey Siri", "Alexa", ...
  negative/random_speech/...    # generic sentences, unrelated to the wake word
  negative/noise/...            # synthetic white/pink/brown/babble/hum beds
  negative/silence/...          # near-silent room-tone clips
```

`manifest.csv` columns: `filepath, label, category, text, voice_id,
speaker_id, sample_rate, duration_sec, split, augmentation` (the last is a
JSON blob of exactly which augmentations were applied to that file).

Tune scale via CLI flags, e.g.:

```bash
.venv/bin/python3 scripts/generate_dataset.py --clean \
  --positive-variants-per-combo 4 \
  --confusable-per-voice 15 --negative-variants-per-combo 3 \
  --noise-clips-per-kind 100
```

Run `.venv/bin/python3 scripts/generate_dataset.py --help` for the full list
(per-voice phrase-sample counts for each negative category, variants per
combo, noise clip counts, train/val/test fractions, seed).

## Design

### Voices
Five single-speaker Piper voices (US female/male, British male/female,
South-Asian-influenced English) plus 75 speaker IDs sampled from the
904-speaker `en_US-libritts_r-medium` model, plus 4 Hindi-accented voices
(see below) — 84 distinct voices total from 7 downloaded models. Edit
`scripts/config.py` (`SINGLE_SPEAKER_VOICES`, `MULTI_SPEAKER_NUM_IDS_TO_SAMPLE`)
to add more.

**Every voice produces both positives and negatives.** Positives use all 84
voices × all wake-word phrases; each negative category uses the same 84-voice
roster, each voice contributing a random subsample of that category's phrase
list (sized by `--confusable-per-voice` etc.). This matters: if some voices
only ever appeared in one class, the model could learn "this speaker equals
positive" instead of "this word equals positive" — a shortcut that works
great on synthetic held-out data and fails completely on a real voice it's
never heard.

### Indian-accented voices (L2-ARCTIC)
`en_US-l2arctic-medium` is built from the L2-ARCTIC corpus: real non-native
English speakers reading English in their own accent (not synthetic accent
imitation). 4 of its 24 speakers — `ASI`, `RRBI`, `SVBI`, `TNI` — are Hindi-L1
(Indian-accented English), and are always included in the roster (not
randomly sampled, unlike the libritts_r pool) since they were deliberately
chosen for accent coverage. This is the only genuinely Indian-accented voice
source in the pipeline; `en_US-kusal-medium` above is Sri Lankan-influenced,
not Indian. Still limited: only Hindi L1 (not Tamil/Telugu/Bengali/Punjabi-
accented English), and the source recordings are careful read-speech from an
accent-research corpus, not casual conversation — real recordings from Indian
users remain the more reliable signal for anything accent-related.

### What counts as positive
**Only the full phrase "Hey Tara".** Bare `"Tara"`, `"Okay Tara"`, `"Hi Tara"`
and `"Hey"` alone are all negatives — the detector fires on the wake phrase,
not on the name appearing in ordinary conversation. The positives are that one
phrase rendered with different punctuation (`config.POSITIVE_PHRASES`) so the
TTS front-end varies prosody and pause placement across voices.

**A note on pronunciation.** espeak-ng phonemizes the wake word identically for
every voice — `hˈeɪ tˈɑːɹɹə` ("TAH-ruh", *a* as in *father*). It is the
phonemizer, not the voice model, that decides this, so adding more voices or
accents does **not** add pronunciation variety. Users who say "TAIR-uh" or
"TARR-uh" are unrepresented. `"Hey Terra"`, `"Hey Tarra"` and `"Hey Tarah"` are
therefore excluded from the negatives too: labeling them negative would train
the detector to actively reject those speakers. Decide the canonical
pronunciation with the team, then either add the variants to
`POSITIVE_PHRASES` (detector accepts them) or to `CONFUSABLE_PHRASES`
(detector rejects them) — but do it deliberately.

### Phrase-list hygiene (automatic, at generation time)
Two guards run before any audio is produced:
- **Sound-alike dedup** — phrases that phonemize identically (`"Hey Kara"` vs
  `"Hey Cara"`, `"Hey Tora"` vs `"Hey Torah"`) are collapsed. Text variety is
  not acoustic variety; without this you get hidden duplicates under distinct
  filenames that inflate the apparent dataset size.
- **Label-poisoning check** — generation **aborts** if any negative phrase
  phonemizes identically to a positive. Nothing downstream would ever surface
  that as an error, and it would teach the model to reject its own wake word.

### Negative categories
- **confusable** — the most important category: phrases that sound close to
  "Hey Tara" (*Hey Sara, Hey Tarot, Hey Terra, Hey Tar, Tara, A Tara, ...*).
  These are what actually teach the model a tight decision boundary instead
  of just "any speech vs. no speech".
- **other_wakeword** — "Hey Google", "Hey Siri", "Alexa", etc., so the model
  doesn't fire on competing products' wake words.
- **random_speech** — everyday sentences (weather, commands, numbers, days)
  for broad phoneme/vocabulary coverage.
- **noise / silence** — synthetic white/pink/brown/babble/hum beds and
  near-silent room tone, so the model learns not to fire on non-speech audio.

### Augmentation (`scripts/augment.py`, numpy/scipy only)
A `CLEAN_PROB` fraction of clips (15% by default, decided independently per
clip — not tied to which copy number it is) skip augmentation entirely and
keep the raw TTS/noise-generator output as a reference. Everything else gets:
random speed perturbation, independent pitch shift, gain (±6 dB — preserved
all the way to the output file; only a clip that would actually clip gets
peak-limited), additive noise at a randomized SNR (0–30 dB), probabilistic
synthetic reverb, probabilistic low-pass filtering (far-field/muffled-mic
effect), occasional soft clipping (cheap-mic distortion), and randomized
leading/trailing silence padding. Resampling (speed/pitch) uses a rational
approximation of the resample ratio internally, not the raw sample counts —
otherwise `scipy.signal.resample_poly` builds a filter with tens of thousands
of taps and generation is ~2 orders of magnitude slower for no audible gain
in quality.

### Fixed clip duration, random placement
Every clip — positive, negative, noise, silence alike — is padded/cropped to
exactly `config.CLIP_DURATION_S` (2.0s), with the content placed at a **random
offset** inside the window rather than centered. Two separate reasons:
- Uniform duration stops the model classifying by clip length (positives are
  short phrases, some negatives are full sentences — before this was fixed, a
  duration-only classifier scored 61%).
- Random placement stops it learning "the wake word sits in the middle of the
  buffer", which is true here but false under streaming inference, where a
  sliding window sees the phrase at an arbitrary offset.

### Validating the output
```bash
.venv/bin/python3 scripts/validate_dataset.py
```
Checks manifest/disk agreement, orphaned files from earlier runs, NaN/silent
clips, speaker leakage across splits, whether every voice contributes to both
classes, and whether the label is predictable from duration, loudness, SNR, or
"clean" fraction alone. Exits non-zero on any hard failure. Run it after every
regeneration — these are exactly the failures that are invisible when you spot
check a few clips by ear.

### Speaker-disjoint splits
Train/val/test (80/10/10 by default) are assigned **per voice**, not per
sample — every clip from a given synthetic speaker stays entirely within one
split. This matters: splitting randomly at the file level lets the same
voice appear in both train and test, which inflates measured accuracy
because the model partly learns to recognize that TTS voice rather than the
word. Noise/silence clips (which have no voice) are split independently per
file at the same train/val/test proportions.

## Important caveats — read before training on this alone

1. **The held-out split cannot detect this dataset's main failure mode — do
   not trust its accuracy number.** 75 of the 80 voices come from a single
   model (`libritts_r`); all 80 come from just 6. The speaker-disjoint split
   prevents *speaker* memorization but not *vocoder* memorization — train and
   test share the same synthesis artifacts. So the model can partly learn "this
   is Piper-generated audio containing /tɑːɹə/" and still score ~96-98% on
   test. That high number is not evidence it will work in a room. No quantity
   of additional Piper voices fixes this, because the artifacts come from the
   vocoder, not the speaker.

   **Mitigation: record a real eval set.** Even 20-30 phone recordings of
   teammates saying "Hey Tara" (plus some near-misses and background chatter),
   held out entirely from training, is the only measurement here that means
   anything. Treat every metric from the synthetic test split as a
   pipeline sanity check — it tells you training is wired up correctly, not
   that the detector works.
2. **Indian-English accent coverage is thin.** Piper doesn't ship a
   dedicated Indian-English voice; `en_US-kusal-medium` (Sri Lankan-accented)
   is the closest available stand-in here. If your deployment target is
   Indian users, real recordings matter even more for this project.
3. **Noise beds are synthetic**, not real recorded noise. For production
   robustness, swap in real datasets like MUSAN, DEMAND, or ESC-50 as
   additional negative/augmentation noise sources (drop wav files in a
   folder and point `noise_gen` at them, or extend `AUGMENT_RANGES` mixing
   logic) — the interface is generator-based so replacing the noise source
   is straightforward.

5. **No real speech appears anywhere, including in the negatives.** Real false
   accepts come from TV dialogue, music with vocals, overlapping conversation
   and half-heard words — none of which this dataset contains. Expect the
   deployed false-accept rate to be far worse than the test split suggests.
   Adding a general speech corpus (Common Voice, LibriSpeech) as extra
   negatives is the single highest-value addition after real positives.

6. **No vocal-effort variation.** People shout at devices from across a room;
   raised voice changes the spectrum, not just the level (Lombard effect).
   Every TTS clip here is calm and close-mic'd, so this variation is absent
   and cannot be synthesized by gain augmentation.

7. **Not bit-reproducible.** `--seed` controls phrase sampling, augmentation
   and splits, but the VITS sampling inside onnxruntime is not seeded by it,
   so audio differs slightly run to run. The manifest records what was applied.
4. **Positive:negative ratio.** Defaults land roughly 1:4 by file count,
   weighted toward confusable/random-speech negatives; typical KWS setups
   use anywhere from 1:3 to 1:10 depending on how much false-accept
   tolerance you need. Adjust `--confusable-per-voice`,
   `--other-wakeword-per-voice`, `--random-speech-per-voice`, and the
   `--*-variants-per-combo` flags to rebalance.

## Real recordings (held-out eval set)

`orginal_recording/orginal dataset/` holds the team's real phone recordings —
`member1/`, `member2/`, ... each with a "hey tara" folder (positive) and a
"tara_alone" folder (negative, the bare name with no "Hey"). Folder-naming
conventions differ per member (`hey tara` vs `Hey_Tara` vs `heytara`, etc.) —
don't assume a fixed pattern when working with this data.

Build its manifest (same schema as the synthetic one, plus a `quality_flag`
column) with:
```bash
.venv/bin/python3 scripts/ingest_real_recordings.py
```
This is read-only — it never copies, moves, or modifies the source audio.
It auto-discovers `member*/` folders (so new contributors are picked up
without editing the script) and auto-flags likely-clipped files (a run of
>50 consecutive samples at the digital ceiling) rather than hardcoding which
files are bad, so a re-recorded file clears its own flag on the next run.
Output: `orginal_recording/manifest.csv`, every row's `split` set to
`"real_eval"`.

**Keep this set held out.** It exists specifically to catch what the
synthetic test split cannot (see the caveat above about shared vocoder
artifacts) — never train on it, and never fold it into the synthetic
train/val/test split logic. Training code should filter to `split=="real_eval"`
for evaluation only, and should probably skip rows with a non-empty
`quality_flag` (currently 2 files from one member, both clipped) until
they're re-recorded and the flag clears on its own.
- Add more confusable phrases, other-wake-words, or sentences: edit the
  lists in `scripts/config.py`.
- Add more/different voices: extend `SINGLE_SPEAKER_VOICES` in
  `scripts/config.py` (browse available Piper voices at the
  [Piper voices samples page](https://rhasspy.github.io/piper-samples/)),
  then `python -m piper.download_voices <name>`.
- Add real recordings later: drop them under `data/positive/wake_word/` (or
  a new `data/positive/real/` subfolder) with matching manifest rows —
  the manifest schema doesn't assume synthetic origin.
