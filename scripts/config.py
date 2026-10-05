"""Central word lists, voice roster, and augmentation ranges for the 'Hey Tara'
wake-word dataset generator. Edit this file to extend phrase coverage or swap voices
-- generate_dataset.py just consumes these lists.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VOICES_DIR = REPO_ROOT / "voices"
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"  # downloaded ASR models (e.g. Moonshine) for scripts/asr_server.py

# --------------------------------------------------------------------------- #
# TTS voices
# --------------------------------------------------------------------------- #
# Single-speaker piper models used as-is, plus one large multi-speaker model
# (libritts_r, 904 speakers) that we sample from to maximize voice diversity
# without downloading dozens of separate models.
SINGLE_SPEAKER_VOICES = [
    "en_US-amy-medium",
    "en_US-ryan-medium",
    "en_GB-alan-medium",
    "en_GB-southern_english_female-low",
    "en_US-kusal-medium",  # South-Asian-influenced English accent
]

MULTI_SPEAKER_VOICE = "en_US-libritts_r-medium"
MULTI_SPEAKER_NUM_IDS_TO_SAMPLE = 75  # out of 904 available speaker IDs

# L2-ARCTIC corpus voice: real non-native English speakers reading English in
# their own accent (not synthetic accent imitation). 4 of its 24 speakers are
# Hindi-L1 (Indian-accented English) -- the only genuinely Indian-accented
# voices available in Piper's public voice set, addressing a real gap: none
# of the voices above are Indian-accented (kusal is Sri Lankan-influenced).
# Always included by code, not randomly sampled like the libritts_r pool,
# since they were deliberately chosen for accent coverage.
# Caveat: only covers Hindi L1 (not Tamil/Telugu/Bengali/Punjabi-accented
# English, which sound quite different), and the source recordings are
# careful read-speech from an accent-research corpus, not casual conversation.
L2ARCTIC_VOICE = "en_US-l2arctic-medium"
L2ARCTIC_HINDI_SPEAKER_CODES = ["ASI", "RRBI", "SVBI", "TNI"]

# Total roster = len(SINGLE_SPEAKER_VOICES) + MULTI_SPEAKER_NUM_IDS_TO_SAMPLE
#              + len(L2ARCTIC_HINDI_SPEAKER_CODES).
# Every voice in the roster is used for BOTH positives and negatives (see
# generate_dataset.py) so the model can't learn "this speaker -> positive".

# --------------------------------------------------------------------------- #
# Positive phrase text variants (same wake word, different punctuation/casing
# so the TTS front-end produces slightly different prosody/pauses)
# --------------------------------------------------------------------------- #
# Only the full phrase "Hey Tara" is a positive. Bare "Tara"/"Tarra", "Okay
# Tara", "Hi Tara" etc. are negatives (see CONFUSABLE_PHRASES) -- the detector
# should fire on the wake phrase, not on the name in ordinary conversation.
#
# PRONUNCIATION: the name "Tara" has (at least) three common pronunciations
# and the team decided to accept ALL THREE as positive, since real users
# (especially a mixed Indian/international user base) will say it different
# ways:
#   /ɑː/ "TAH-rah" (father vowel) -- UK English AND the traditional Hindi/
#        Sanskrit pronunciation of तारा. espeak-ng's default reading of the
#        literal spelling "Tara". Most likely default for Indian users.
#   /æ/  "TARR-uh" (cat vowel) -- US English. espeak-ng does NOT produce this
#        from the spelling "Tara"; the spelling "Tarra" is used purely to
#        steer the phonemizer to this vowel (double consonant after a single
#        vowel is a standard English spelling cue for a short vowel) -- it is
#        not a typo and is not shown to end users anywhere.
#   /ɛ/  "TAIR-uh" (bed vowel) -- read via the spelling "Terra". Also accepted.
#
# If the phonemizer or voice set changes, re-verify with
# scripts/tts_engine.phonemize() that "Tarra"/"Terra" still yield /æ/ and /ɛ/
# respectively before relying on this again.
POSITIVE_PHRASES = [
    "Hey Tara",
    "Hey, Tara",
    "Hey Tara.",
    "Hey Tara!",
    "Hey Tara?",
    "Hey Tarra",
    "Hey, Tarra",
    "Hey Tarra.",
    "Hey Tarra!",
    "Hey Tarra?",
    "Hey Terra",
    "Hey, Terra",
    "Hey Terra.",
    "Hey Terra!",
    "Hey Terra?",
]

# --------------------------------------------------------------------------- #
# Hard negatives: phonetically close to "Hey Tara" -- these are the most
# important negatives for teaching the detector a tight decision boundary.
# --------------------------------------------------------------------------- #
# Removed as exact phoneme duplicates of an entry already here: "Hey Cara"
# (== Hey Kara), "Hey Taro" (== Hey Tarot), "Hey Torah" (== Hey Tora).
#
# All three pronunciations of "Tara" (/ɑː/, /æ/, /ɛ/) are positive now -- see
# POSITIVE_PHRASES -- so none of them appear here as pronunciation negatives.
CONFUSABLE_PHRASES = [
    # one phoneme away from the target
    "Hey Sara", "Hey Kara", "Hey Zara", "Hey Dara", "Hey Lara", "Hey Farah",
    # share the stressed "TAR" onset
    "Hey Tarot", "Hey Tiara", "Hey Tamra", "Hey Tanya", "Hey Tata",
    "Hey Tyra", "Hey Tora", "Hey Tarun", "Hey Tarzan", "Hey Tarragon",
    "Hey Tarp", "Hey Target", "Hey Tarmac", "Hey Tardy",
    # the wake word embedded in, or adjacent to, ordinary words
    "Sahara", "Mascara", "Hey Barbara", "Hey guitar",
    "The Sahara desert", "She wore mascara", "Play the guitar",
    # right phrase shape, wrong carrier word -- all three accepted pronunciations
    "Okay Tara", "Hi Tara", "A Tara",
    "Okay Tarra", "Hi Tarra", "A Tarra",
    "Okay Terra", "Hi Terra", "A Terra", "Hey Cara Lee",
    # the name alone (all three pronunciations) and the carrier alone -- none
    # of these should trigger; the full phrase "Hey Tara"/"Tarra"/"Terra" is
    # required
    "Tara", "Tarra", "Terra", "Hey", "Hey there", "Hey Tar",
]

# --------------------------------------------------------------------------- #
# Other product wake words -- negatives that must not falsely trigger "Tara"
# --------------------------------------------------------------------------- #
OTHER_WAKEWORDS = [
    "Hey Google", "OK Google", "Hey Siri", "Alexa", "Hey Cortana",
    "Computer", "Hey Bixby", "Jarvis", "Hey Assistant", "Wake up",
    "Hello Google", "Hey Alexa",
]

# --------------------------------------------------------------------------- #
# General speech negatives -- everyday sentences with no relation to the
# wake word, for broad phoneme/vocabulary coverage.
# --------------------------------------------------------------------------- #
RANDOM_SPEECH_SENTENCES = [
    "What's the weather like today?",
    "Please turn off the lights in the kitchen.",
    "Can you set a timer for ten minutes?",
    "I think it's going to rain later this evening.",
    "Let's meet at the coffee shop around noon.",
    "Turn up the volume on the television.",
    "Remind me to call my brother tomorrow.",
    "How far is it to the train station?",
    "Add milk and eggs to the shopping list.",
    "Play some music while I make dinner.",
    "The meeting has been moved to three o'clock.",
    "Can you translate this sentence into French?",
    "I left my keys on the kitchen counter.",
    "What time does the store close tonight?",
    "Book a table for four at the Italian restaurant.",
    "The kids are watching cartoons in the living room.",
    "Send a message to Priya about the project update.",
    "I need to charge my phone before we leave.",
    "Traffic on the highway is really bad this morning.",
    "Could you please repeat that a little slower?",
    "Set an alarm for six thirty tomorrow morning.",
    "The package should arrive sometime this afternoon.",
    "Switch the fan to a lower speed, it's too loud.",
    "Let's watch a movie after dinner tonight.",
    "My favorite season is autumn because of the colors.",
    "Can you check if it's going to snow this weekend?",
    "I forgot my umbrella at the office again.",
    "The bus was ten minutes late this morning.",
    "How do you make a good cup of masala chai?",
    "Please close the door, it's getting cold in here.",
    "We should plan a trip to the mountains next month.",
    "That new restaurant downtown has great reviews.",
    "I can't find my charger anywhere in this room.",
    "The Wi-Fi has been really slow all day.",
    "Could you help me move this table to the corner?",
    "Let's go for a walk before it gets dark.",
    "Did you finish reading that book yet?",
    "The battery on my laptop is almost dead.",
    "I'll pick you up from the airport at five.",
    "This song always reminds me of summer.",
    "one two three four five six seven eight nine ten",
    "Monday Tuesday Wednesday Thursday Friday Saturday Sunday",
    "January February March April May June July",
    "open close start stop pause resume next previous",
    "yes no maybe later not now go ahead",
]

# --------------------------------------------------------------------------- #
# Synthetic background-noise categories used both as standalone negatives and
# as additive augmentation beds for positives/negatives.
# --------------------------------------------------------------------------- #
NOISE_KINDS = ["white", "pink", "brown", "babble", "hum", "silence"]

# --------------------------------------------------------------------------- #
# Augmentation parameter ranges
# --------------------------------------------------------------------------- #
AUGMENT_RANGES = {
    "speed_factor": (0.85, 1.15),       # tempo + pitch perturbation
    "pitch_semitones": (-2.5, 2.5),      # independent pitch shift
    # Widened from the original (-6, 6) after comparing against the team's real
    # phone recordings: real "Hey Tara" clips vary 2x more in RMS loudness than
    # +/-6 dB produced (std 5.8 dB vs 2.5 dB), and reach as quiet as -34 dBFS,
    # well below the old floor of -26.7 dBFS. Skewed toward quieter because
    # that's the direction the real data extended further past the old range.
    "gain_db": (-14.0, 8.0),
    # SNR floor is deliberately 5 dB, not 0 dB. Several negatives ("Hey Kara",
    # "Hey Dara", "Hey Zara") differ from the wake word by a single consonant;
    # buried at 0 dB SNR they become acoustically identical to a positive while
    # carrying the opposite label, which is just label noise. The floor applies
    # to ALL categories -- applying a gentler floor only to hard negatives would
    # make "heavily degraded" itself a cue for the positive class.
    "snr_db_choices": [5, 8, 12, 16, 20, 30],
    "reverb_prob": 0.35,
    "reverb_decay_s": (0.1, 0.6),
    "reverb_drr_db": (0.0, 15.0),        # direct-to-reverberant ratio, real rooms
    "lowpass_prob": 0.15,               # simulate far-field / muffled mic
    "lowpass_cutoff_hz": (1500, 4000),
    "leading_silence_ms": (0, 400),
    "trailing_silence_ms": (0, 400),
    "clip_prob": 0.05,                  # simulate cheap-mic clipping distortion
}

TARGET_SAMPLE_RATE = 16000  # standard for KWS models; piper outputs 22050, we resample down

# Every clip (positive, negative, noise, silence) is padded/cropped to exactly
# this duration. Fixing duration dataset-wide prevents a model from learning
# "short clip -> positive, long clip -> negative" as a shortcut instead of the
# actual acoustic content. Downstream streaming inference should use a sliding
# window rather than assuming utterances arrive pre-segmented at this length.
CLIP_DURATION_S = 2.0

# Fraction of clips (positive AND negative alike) that skip augmentation
# entirely and keep the raw TTS/noise-generator output. Applied by a coin flip
# per copy rather than "copy 0 of N", so the clean fraction doesn't drift when
# positive/negative variants-per-combo counts differ.
CLEAN_PROB = 0.15
