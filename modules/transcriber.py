import torch
import os
import gc
import warnings
from faster_whisper import WhisperModel

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE + MODEL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

def _select_model_config() -> tuple[str, str, str]:
    """
    Choose the best model size and compute type for available hardware.

    VRAM budget (faster-whisper int8 usage):
      tiny   ~200MB  — emergency fallback, poor JP accuracy
      base   ~370MB  — acceptable for clean vocals
      small  ~750MB  — recommended for 2GB VRAM, good JP accuracy
      medium ~1.5GB  — requires 2GB+ with NO other GPU load
      large  ~3.1GB  — requires 4GB+ VRAM

    For a 2GB VRAM card (HP Pavilion 15):
      - small + int8_float16 fits at ~750MB, leaving headroom for PyTorch overhead
      - medium WILL fit numerically but PyTorch overhead + driver reservation
        (~300-400MB) pushes you to ~1.9GB and risks OOM mid-transcription
      - If CUDA OOM occurs, _run_with_fallback() retries on CPU automatically

    Returns: (device, compute_type, model_size)
    """
    if not torch.cuda.is_available():
        return "cpu", "int8", "small"

    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb    = vram_bytes / (1024 ** 3)
    print(f"[WHISPER] Detected GPU VRAM: {vram_gb:.1f} GB")

    if vram_gb >= 5.0:
        return "cuda", "float16", "large-v3"
    elif vram_gb >= 3.0:
        return "cuda", "int8_float16", "medium"
    elif vram_gb >= 1.5:
        # Your HP Pavilion 15 lands here — small is the correct choice
        return "cuda", "int8_float16", "small"
    else:
        # Very low VRAM (integrated / shared memory) — CPU is safer
        return "cpu", "int8", "base"


DEVICE, COMPUTE_TYPE, MODEL_SIZE = _select_model_config()

# ─────────────────────────────────────────────────────────────────────────────
# TRANSCRIPTION PROMPT
# ─────────────────────────────────────────────────────────────────────────────

# A longer initial_prompt dramatically reduces hallucinations in Whisper.
# It primes the model on:
#   - The language register (lyric-style Japanese, not conversational)
#   - The domain (Touhou doujin music)
#   - Common hallucination targets in music (silence → fake subtitles)
#
# Rule of thumb: use a snippet that LOOKS like what the output should look like.
# This prompt is styled like an actual LRC line — Whisper continues in that style.
INITIAL_PROMPT = (
    "東方Project アレンジ 歌詞 ボーカル曲。"
    "誰も知らない 光見えない 奈落の果てを遡る 想い集めながら"
)

# ─────────────────────────────────────────────────────────────────────────────
# HALLUCINATION FILTER
# ─────────────────────────────────────────────────────────────────────────────

# Whisper is prone to outputting these strings during silence or non-vocal audio.
# Any segment whose text matches these (after stripping) is discarded.
_HALLUCINATION_FRAGMENTS = {
    "視聴",
    "ご視聴",
    "ありがとうございます",
    "チャンネル登録",
    "字幕",
    "翻訳",
    "制作",
    "エンコード",
    "subtitles",
    "subscribe",
    "thank you for watching",
    "amara.org",
}


def _is_hallucination(text: str) -> bool:
    """
    Return True if the segment text is a known Whisper hallucination.

    Checks:
      1. Too short (< 2 chars) — usually noise artifacts
      2. Exact match against known hallucination strings
      3. Contains known hallucination fragments (for longer variants)
    """
    t = text.strip().lower()
    if len(t) < 2:
        return True
    if t in {h.lower() for h in _HALLUCINATION_FRAGMENTS}:
        return True
    for fragment in _HALLUCINATION_FRAGMENTS:
        if fragment.lower() in t:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# CORE TRANSCRIPTION
# ─────────────────────────────────────────────────────────────────────────────

def _transcribe_with_model(
    audio_path: str,
    device: str,
    compute_type: str,
    model_size: str,
) -> list[dict]:
    """
    Load model, transcribe, free VRAM, return segment list.

    Returns list of dicts:
        {
            "start": float,   # segment start in seconds
            "end":   float,   # segment end in seconds
            "text":  str,     # cleaned segment text
            "words": list,    # word-level timestamps (if available)
        }

    Word-level timestamps (word_timestamps=True):
        Whisper can return per-word timing within each segment.
        This gives your reconciler much finer-grained alignment data —
        instead of mapping one official lyric line to a 3-second segment,
        it can map individual words or phrases.
        Stored in entry["words"] as list of {"word", "start", "end"}.
        If word timestamps are unavailable (older faster-whisper), the key
        is present but empty, so downstream code can always check safely.
    """
    print(f"[WHISPER] Loading {model_size!r} on {device} ({compute_type})")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    print(f"[WHISPER] Transcribing: {os.path.basename(audio_path)}")
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        language="ja",
        vad_filter=True,            # skip non-vocal segments
        vad_parameters={
            "min_silence_duration_ms": 500,  # don't split on brief pauses
            "speech_pad_ms": 200,            # small pad around speech regions
        },
        word_timestamps=True,        # enable per-word timing
        initial_prompt=INITIAL_PROMPT,
        condition_on_previous_text=False,   # prevents error snowballing
        no_speech_threshold=0.6,     # discard low-confidence silent segments
        log_prob_threshold=-1.0,     # discard low overall confidence segments
    )

    print(f"[WHISPER] Detected language: {info.language!r} "
          f"(probability: {info.language_probability:.2f})")

    results = []
    for seg in segments:
        text = seg.text.strip()

        if _is_hallucination(text):
            print(f"[WHISPER] Hallucination filtered: {text!r}")
            continue

        # Collect word-level timestamps (faster-whisper returns Word objects)
        words = []
        if seg.words:
            for w in seg.words:
                words.append({
                    "word":  w.word.strip(),
                    "start": round(w.start, 3),
                    "end":   round(w.end, 3),
                })

        results.append({
            "start": round(seg.start, 2),
            "end":   round(seg.end, 2),
            "text":  text,
            "words": words,
        })

    # CRITICAL: Free VRAM immediately — next stage (LLM via Groq) needs memory
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    print(f"[WHISPER] Done. {len(results)} segments transcribed.")
    return results


def _run_with_fallback(audio_path: str) -> list[dict]:
    """
    Try CUDA transcription first; fall back to CPU on OOM.

    On a 2GB VRAM card, CUDA OOM is possible if:
      - Another process is using GPU memory
      - The audio is very long (>10 min) and batch processing spills
      - Driver overhead leaves less than expected

    On OOM, we immediately free VRAM and retry on CPU with the same model size.
    CPU with small/int8 is slower (~5-10× real time) but will always complete.
    """
    if DEVICE == "cpu":
        # Already on CPU — no fallback needed
        return _transcribe_with_model(audio_path, "cpu", "int8", MODEL_SIZE)

    try:
        return _transcribe_with_model(audio_path, DEVICE, COMPUTE_TYPE, MODEL_SIZE)

    except torch.cuda.OutOfMemoryError:
        warnings.warn(
            f"[WHISPER] CUDA OOM with {MODEL_SIZE!r} on {DEVICE}. "
            f"Falling back to CPU with 'small' model. "
            f"This will be slower but will complete."
        )
        torch.cuda.empty_cache()
        gc.collect()

        # Drop one model size on CPU to ensure it finishes
        fallback_size = "base" if MODEL_SIZE == "small" else "small"
        return _transcribe_with_model(audio_path, "cpu", "int8", fallback_size)

    except Exception as e:
        # Non-OOM error — re-raise so app.py can surface it to the user
        raise RuntimeError(f"[WHISPER] Transcription failed: {e}") from e


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def run_transcription(audio_path: str) -> list[dict]:
    """
    Transcribe a vocal audio file and return timed segment list.

    Entry point called by app.py Stage 3.

    Parameters:
        audio_path : path to the separated vocal track (wav/mp3/flac)

    Returns:
        List of segment dicts:
        [
            {
                "start": 12.5,
                "end":   15.2,
                "text":  "誰も知らない　光見えない",
                "words": [
                    {"word": "誰も", "start": 12.5, "end": 13.1},
                    {"word": "知らない", "start": 13.1, "end": 13.8},
                    ...
                ]
            },
            ...
        ]

        Returns [] if the file doesn't exist or no speech is detected.

    Note on full-width spaces in lyrics:
        Whisper does NOT insert full-width spaces 　(U+3000) — those come from
        the official DB lyrics. Whisper output will be plain text like
        '誰も知らない光見えない' (no internal spacing). The reconciler's job
        is to match this against the official lyric line that HAS the spaces.
    """
    if not os.path.exists(audio_path):
        warnings.warn(f"[WHISPER] Audio file not found: {audio_path!r}")
        return []

    return _run_with_fallback(audio_path)