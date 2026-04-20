import re
import warnings


# ─────────────────────────────────────────────────────────────────────────────
# ASS HEADER
# ─────────────────────────────────────────────────────────────────────────────

# Font note: Meiryo is a Windows font (not bundled on Linux/Mac servers).
# If rendering on a Linux server, install via: apt install fonts-meiryo
# or substitute with 'Noto Sans CJK JP' which is freely available.
ASS_HEADER = """\
[Script Info]
Title: EchoSync AI Generated Subtitles
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Main,Meiryo,60,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,3,2,2,10,10,80,1
Style: Ruby,Meiryo,28,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,10,10,80,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

# ─────────────────────────────────────────────────────────────────────────────
# TIME HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def to_seconds(t_str: str) -> float:
    """
    Parse LRC timestamp 'MM:SS.xx' → float seconds.
    Raises ValueError on bad input so the caller knows something went wrong,
    rather than silently returning 0.0 and corrupting all subsequent timings.
    """
    try:
        m, s = t_str.strip().split(":", 1)
        return float(m) * 60 + float(s)
    except Exception:
        raise ValueError(f"Invalid LRC timestamp: {t_str!r}")


def to_ass_time(sec: float) -> str:
    """
    Convert float seconds → ASS timestamp 'H:MM:SS.xx'.
    ASS uses centiseconds (2 decimal places), not milliseconds.
    Clamps to 0 to avoid negative timestamps.
    """
    sec = max(0.0, sec)
    h   = int(sec // 3600)
    m   = int((sec % 3600) // 60)
    s   = sec % 60
    return f"{h}:{m:02d}:{s:05.2f}"


# ─────────────────────────────────────────────────────────────────────────────
# RUBY / FURIGANA PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

# Input format expected from furigana_logic.py:  漢字(かんじ)
# i.e. kanji immediately followed by reading in ASCII parentheses.
_RUBY_PATTERN = re.compile(r"([一-龠々〇〻ヶ]+)\(([^)]+)\)")


def _build_ruby_dialogue(base_text: str, start_sec: float, end_sec: float) -> list[str]:
    """
    Produce correct ASS ruby (furigana) rendering for a single lyric line.

    ASS has no native ruby tag. The standard workaround used by production
    tools (Aegisub, Substation Alpha) is TWO dialogue lines per lyric line:

      Layer 0  — the kanji text (Main style, bottom-center)
      Layer 1  — the furigana text, positioned above the kanji using \\an8
                 (top-center) with a higher layer number so it renders on top

    This is how fansub groups handle ruby and it works in all major players
    (mpv, VLC, MPC-HC, Aegisub preview).

    The \\fs35 approach you had puts the reading on a NEW LINE below via \\N,
    which looks like:
        reading
        漢字
    instead of proper ruby:
        reading   ← small, above
        漢字      ← normal size, below

    Parameters:
        base_text  : lyric line, may contain 漢字(かんじ) annotations
        start_sec  : line start in seconds
        end_sec    : line end in seconds

    Returns list of ASS Dialogue strings (1 line if no ruby, 2 if ruby present).
    """
    start_ass = to_ass_time(start_sec)
    end_ass   = to_ass_time(end_sec)

    has_ruby = bool(_RUBY_PATTERN.search(base_text))

    if not has_ruby:
        # Simple line — no ruby annotations present
        return [f"Dialogue: 0,{start_ass},{end_ass},Main,,0,0,0,,{base_text}"]

    # Strip ruby annotations for the base (kanji) layer
    kanji_only = _RUBY_PATTERN.sub(lambda m: m.group(1), base_text)

    # Build furigana-only string with zero-width spaces as spacers.
    # Each kanji group is replaced by its reading; non-annotated chars become spaces
    # so the reading aligns horizontally above the correct kanji.
    # This is a best-effort approximation — perfect pixel alignment requires
    # the renderer to know glyph widths, which ASS text cannot do.
    reading_parts = []
    last_end = 0
    for match in _RUBY_PATTERN.finditer(base_text):
        # Characters between ruby groups → replace with spaces for alignment
        gap = base_text[last_end:match.start()]
        # Approximate: each full-width char ≈ 1 space at Main fontsize
        reading_parts.append(" " * len(gap))
        reading_parts.append(match.group(2))    # the reading
        last_end = match.end()
    # Remainder after last ruby group
    reading_parts.append(" " * len(base_text[last_end:]))
    furigana_text = "".join(reading_parts)

    # Layer 0: kanji (Main style, alignment 2 = bottom-center)
    base_line = f"Dialogue: 0,{start_ass},{end_ass},Main,,0,0,0,,{kanji_only}"

    # Layer 1: furigana (Ruby style, \\an8 = top-center so it sits above kanji)
    # MarginV 80 in Ruby style pushes it up from the bottom.
    # \\an8 overrides alignment to top-center within the margin area.
    ruby_line = (
        f"Dialogue: 1,{start_ass},{end_ass},Ruby,,0,0,0,,"
        f"{{\\an8}}{furigana_text}"
    )

    return [base_line, ruby_line]


# ─────────────────────────────────────────────────────────────────────────────
# LRC PARSER
# ─────────────────────────────────────────────────────────────────────────────

_LRC_LINE = re.compile(r"^\[([0-9]+:[0-9]+\.[0-9]+)\]\s*(.*)")


def parse_lrc(lrc_content: str) -> list[dict]:
    """
    Parse LRC content into a list of timed entries.

    Returns list of dicts:
        {"start": float, "text": str}

    Rules:
      - Lines without a valid [MM:SS.xx] timestamp are skipped.
      - Lines with a timestamp but empty text are skipped (blank verse
        separators don't need to become subtitle events — the timing gap
        between the previous and next line already creates the visual pause).
      - Duplicate timestamps (LLM sometimes outputs two lines with the same
        start) are deduplicated by keeping the first occurrence, with a warning.
    """
    entries    = []
    seen_times = {}     # timestamp → index in entries, for dedup

    for lineno, raw in enumerate(lrc_content.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue    # blank line — verse separator, skip for events

        match = _LRC_LINE.match(line)
        if not match:
            continue    # metadata tags, comments, malformed lines — skip

        try:
            start_sec = to_seconds(match.group(1))
        except ValueError as e:
            warnings.warn(f"LRC line {lineno}: {e} — skipped")
            continue

        text = match.group(2).strip()
        if not text:
            continue    # timestamped blank line — skip

        # Deduplicate: if same timestamp seen before, keep first, warn
        key = round(start_sec, 2)
        if key in seen_times:
            warnings.warn(
                f"LRC line {lineno}: duplicate timestamp {match.group(1)!r} — "
                f"keeping first occurrence, skipping this line"
            )
            continue

        seen_times[key] = len(entries)
        entries.append({"start": start_sec, "text": text})

    return entries


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONVERTER
# ─────────────────────────────────────────────────────────────────────────────

def lrc_to_ass_final(
    lrc_content: str,
    output_ass_path: str,
    tail_duration: float = 4.0,
) -> None:
    """
    Convert an LRC string to an ASS subtitle file.

    Parameters:
        lrc_content      : LRC file content as string
        output_ass_path  : path to write the .ass file
        tail_duration    : seconds to display the last lyric line (default 4.0)
                           Increase for songs with long outros.

    Ruby/furigana handling:
        If the lyric text contains 漢字(かんじ) annotations (inserted by
        furigana_logic.py), _build_ruby_dialogue() produces two ASS Dialogue
        lines per lyric — one for the kanji, one for the reading above it.
        Lines without annotations produce a single Dialogue line as normal.

    End time logic:
        Each line's end time = the NEXT line's start time.
        This creates clean cuts with no overlap between lines.
        The final line ends at start + tail_duration seconds.
    """
    parsed = parse_lrc(lrc_content)

    if not parsed:
        warnings.warn("lrc_to_ass_final: no valid LRC lines found — writing empty ASS file")

    dialogue_lines = []

    for i, entry in enumerate(parsed):
        start_sec = entry["start"]
        text      = entry["text"]

        # End time: next line's start, or tail for the last line
        if i + 1 < len(parsed):
            end_sec = parsed[i + 1]["start"]
        else:
            end_sec = start_sec + tail_duration

        # Guard: end must be strictly after start (handles identical consecutive
        # timestamps that slipped through dedup)
        if end_sec <= start_sec:
            end_sec = start_sec + 0.5

        dialogue_lines.extend(
            _build_ruby_dialogue(text, start_sec, end_sec)
        )

    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write(ASS_HEADER)
        f.write("\n".join(dialogue_lines))
        f.write("\n")