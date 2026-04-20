import os
import glob
import subprocess
import warnings
import torch
import yt_dlp


# ─────────────────────────────────────────────────────────────────────────────
# DEMUCS MODEL SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def _select_demucs_model() -> tuple[str, list[str]]:
    """
    Pick the Demucs model and extra flags based on available VRAM.

    VRAM requirements (approximate):
      htdemucs          ~3-4 GB   — best quality, transformer hybrid
      htdemucs_ft       ~3-4 GB   — fine-tuned variant, slightly better on music
      mdx_extra         ~2.5 GB   — good quality, CNN-based
      mdx               ~1.5 GB   — recommended for 2GB VRAM cards
      mdx     +segment  ~0.8 GB   — chunked processing, fits on 1GB VRAM

    For HP Pavilion 15 with 2GB VRAM:
      mdx with --segment 8 keeps peak usage under 1.5GB and always completes.
      htdemucs WILL OOM mid-separation and leave no output file.

    Returns: (model_name, extra_flags)
    """
    if not torch.cuda.is_available():
        # CPU: use mdx with small segments to manage RAM
        return "mdx", ["--segment", "6"]

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"[DEMUCS] Detected VRAM: {vram_gb:.1f} GB")

    if vram_gb >= 6.0:
        return "htdemucs_ft", []
    elif vram_gb >= 3.0:
        return "mdx_extra", []
    elif vram_gb >= 1.5:
        # Your card — mdx with chunked segments
        return "mdx", ["--segment", "8"]
    else:
        # Very low VRAM — force CPU via --device flag
        return "mdx", ["--segment", "4", "--device", "cpu"]


DEMUCS_MODEL, DEMUCS_EXTRA_FLAGS = _select_demucs_model()


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO PATH RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────

def _find_downloaded_audio(output_dir: str) -> str | None:
    """
    Find the audio file yt-dlp actually wrote, regardless of extension.

    yt-dlp requests wav but the source container (webm, opus, m4a) determines
    what FFmpeg actually produces. Even with FFmpegExtractAudio + preferredcodec
    wav, some streams come out as .wav.webm or the extension is inferred wrong.

    Search priority: .wav → .mp3 → .flac → .m4a → .opus → .webm
    Returns the first match, or None if nothing found.
    """
    for ext in ("wav", "mp3", "flac", "m4a", "opus", "webm"):
        pattern = os.path.join(output_dir, f"input_audio.{ext}")
        if os.path.exists(pattern):
            return pattern

    # Fallback: glob for anything named input_audio.*
    matches = glob.glob(os.path.join(output_dir, "input_audio.*"))
    if matches:
        return matches[0]

    return None


def _find_vocal_track(output_dir: str, model: str) -> str | None:
    """
    Find the Demucs vocals output file.

    Demucs writes to: output_dir / model_name / stem_name / vocals.wav
    The stem name matches the input filename without extension.
    Returns path if found, None if Demucs didn't produce output.
    """
    # Standard path
    standard = os.path.join(output_dir, model, "input_audio", "vocals.wav")
    if os.path.exists(standard):
        return standard

    # Glob fallback — catches renamed stems or alternate model output dirs
    matches = glob.glob(
        os.path.join(output_dir, "**", "vocals.wav"), recursive=True
    )
    if matches:
        return matches[0]

    return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def download_and_separate(video_url: str, output_dir: str) -> tuple[str | None, dict]:
    """
    Stage 1: Download audio from YouTube and separate vocals with Demucs.

    Returns:
        (vocal_path, metadata)

        vocal_path : path to isolated vocals.wav, or None if separation failed.
                     Caller (app.py) must handle None — if separation fails,
                     Whisper falls back to transcribing the full mixed audio.

        metadata   : dict with keys:
                       title       — YouTube video title (used by query agent)
                       description — video description (used by query agent)
                       channel     — uploader channel name (often = circle name)
                       tags        — list of video tags
                       uploader    — uploader display name
    """

    # ── Step 1: Download ──────────────────────────────────────────────────────
    ydl_opts = {
        "format": "bestaudio/best",
        "nocheckcertificate": True,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        # Fixed stem name so _find_downloaded_audio() knows where to look
        "outtmpl": os.path.join(output_dir, "input_audio"),
    }

    print("[DOWNLOADER] Fetching audio and metadata from YouTube...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)

    # Fan uploads: channel = random fan account, not the circle.
    # Circle name lives in the VIDEO TITLE (e.g. 「kimino-museum」, 【SYNC.ART'S】).
    # _detect_artist() slugs the full title, so no injection needed.
    # Tags are appended in case they contain circle mentions.
    desc_raw = info.get("description", "") or ""
    tags     = info.get("tags", []) or []

    enriched_description = "\n".join(filter(None, [desc_raw, " ".join(tags)]))

    metadata = {
        "title":       info.get("title", "Unknown"),
        "description": enriched_description,
        "tags":        tags,
    }
    print(f"[DOWNLOADER] Title : {metadata['title']!r}")

    # FIX: Verify the audio file actually exists before calling Demucs
    audio_path = _find_downloaded_audio(output_dir)
    if not audio_path:
        warnings.warn("[DOWNLOADER] yt-dlp completed but no audio file found in output dir.")
        return None, metadata

    print(f"[DOWNLOADER] Audio saved: {os.path.basename(audio_path)}")

    # ── Step 2: Vocal Separation ──────────────────────────────────────────────
    print(f"[DEMUCS] Separating vocals with model={DEMUCS_MODEL!r} ...")
    cmd = [
        "demucs",
        "-n", DEMUCS_MODEL,
        "--out", output_dir,
        *DEMUCS_EXTRA_FLAGS,   # e.g. ['--segment', '8'] for low VRAM
        audio_path,
    ]

    try:
        # FIX: Don't use capture_output=True — pipe stderr separately so we
        # can print Demucs progress AND still capture errors for the warning.
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.stderr:
            # Demucs writes progress to stderr — show it so user sees activity
            print(result.stderr[-500:])  # last 500 chars to avoid flooding

    except subprocess.CalledProcessError as e:
        # FIX: Extract the actual error message from stderr, not just the exit code
        err_detail = e.stderr[-800:] if e.stderr else "no stderr output"
        warnings.warn(
            f"[DEMUCS] Separation failed (exit code {e.returncode}).\n"
            f"  Model: {DEMUCS_MODEL}  Flags: {DEMUCS_EXTRA_FLAGS}\n"
            f"  Error: {err_detail}\n"
            f"  Falling back to full mixed audio for transcription."
        )
        # FIX: Fall back to the original mixed audio instead of returning None.
        # Whisper on a mixed track is imperfect but far better than crashing.
        return audio_path, metadata

    except FileNotFoundError:
        warnings.warn(
            "[DEMUCS] 'demucs' command not found. "
            "Install with: pip install demucs\n"
            "Falling back to full mixed audio."
        )
        return audio_path, metadata

    # ── Step 3: Locate vocal output ───────────────────────────────────────────
    vocal_path = _find_vocal_track(output_dir, DEMUCS_MODEL)

    if not vocal_path:
        warnings.warn(
            f"[DEMUCS] Separation ran but vocals.wav not found under "
            f"{output_dir!r}. Falling back to mixed audio."
        )
        return audio_path, metadata

    print(f"[DEMUCS] Vocals isolated: {os.path.relpath(vocal_path, output_dir)}")
    return vocal_path, metadata