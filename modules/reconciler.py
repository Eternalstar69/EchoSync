import os
import re
from groq import Groq

# Do NOT initialize the client at module level —
# avoids errors during import before load_dotenv() runs in app.py.

def get_client():
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("GROQ API_KEY not found. Ensure load_dotenv() is called in app.py")
    return Groq(api_key=api_key)


# ─────────────────────────────────────────────────────────────────────────────
# LYRIC PRE-PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_lyrics(raw_lyrics: str) -> str:
    """
    Normalize raw DB lyrics before sending to the LLM.

    Problems this fixes:
      1. Double (or more) blank lines → single blank line.
         Your files use blank lines as verse separators. The LLM needs
         consistent formatting to understand structure, but 2+ blank lines
         add no extra information and inflate the token count.

      2. Trailing whitespace on each line — stripped.

      3. Full-width spaces 　(U+3000) within lines are LEFT INTACT.
         '誰も知らない　光見えない' stays as-is because:
         - They are part of the original lyric phrasing/rhythm.
         - Whisper will return this as one segment (one timestamp).
         - The LLM should map the whole line to one timestamp.
         - Splitting on 　 would require sub-segment timestamp interpolation
           which Whisper cannot provide natively.

    Returns cleaned lyrics string.
    """
    lines = raw_lyrics.splitlines()

    # Strip trailing whitespace per line
    lines = [line.rstrip() for line in lines]

    # Collapse runs of 2+ blank lines into a single blank line
    cleaned = []
    prev_blank = False
    for line in lines:
        is_blank = (line.strip() == "")
        if is_blank and prev_blank:
            continue    # skip consecutive blank lines
        cleaned.append(line)
        prev_blank = is_blank

    # Strip leading/trailing blank lines from the whole block
    result = "\n".join(cleaned).strip()
    return result


def format_whisper_for_prompt(whisper_results: list[dict]) -> str:
    """
    Format Whisper segments into a readable block for the LLM prompt.

    Each segment becomes:   [MM:SS.xx] text

    Whisper note on full-width spaces:
      Whisper segments by audio pause detection. A line like
      '誰も知らない　光見えない' will typically appear as ONE segment
      because the full-width space 　 is a typographic pause, not a
      silence long enough for Whisper to split on. The LLM should
      treat the entire Whisper segment text as one mappable unit.
    """
    lines = []
    for s in whisper_results:
        start = s["start"]
        mm    = int(start // 60)
        ss    = start % 60
        text  = s["text"].strip()
        lines.append(f"[{mm:02d}:{ss:05.2f}] {text}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1: LRC GENERATION (Whisper + DB Lyrics → Timestamped LRC)
# ─────────────────────────────────────────────────────────────────────────────

def run_main_reconciliation(whisper_results: list[dict], db_lyrics: str | None = None) -> str:
    """
    Map official DB lyrics to Whisper timestamps and produce LRC format.

    Two modes:
      - db_lyrics provided → map official lyrics to Whisper timestamps (preferred)
      - db_lyrics is None  → clean and format Whisper STT output directly

    LRC format rules enforced in the prompt:
      [MM:SS.xx] lyric line
      Blank lines between verses are kept as blank LRC lines (no timestamp).
      Full-width spaces 　 within a line are preserved as-is.
      Each non-blank lyric line gets exactly one timestamp.
    """
    client      = get_client()
    stt_context = format_whisper_for_prompt(whisper_results)

    if db_lyrics:
        lyrics_clean = preprocess_lyrics(db_lyrics)

        system_prompt = (
            "You are a professional LRC synchronization agent. "
            "Your only job is to produce a valid LRC file. "
            "Output ONLY the LRC content — no explanations, no markdown, no code blocks."
        )

        user_prompt = f"""TASK: Map the OFFICIAL LYRICS to the WHISPER TIMESTAMPS and output valid LRC format.

═══════════════════════════════
OFFICIAL LYRICS (source of truth for text):
═══════════════════════════════
{lyrics_clean}

═══════════════════════════════
WHISPER STT (source of truth for timing):
═══════════════════════════════
{stt_context}

═══════════════════════════════
STRICT OUTPUT RULES:
═══════════════════════════════
1. FORMAT: Each lyric line must be: [MM:SS.xx] lyric text
2. TIMESTAMPS: Take timestamps from the Whisper STT. Match each official lyric
   line to the closest Whisper segment by meaning/phonetics.
3. BLANK LINES: If the official lyrics have a blank line between verses,
   output a blank line in the LRC too (with NO timestamp). This preserves
   verse structure for subtitle renderers.
4. FULL-WIDTH SPACES: Lines containing　(U+3000 full-width space) like
   '誰も知らない　光見えない' must be output as a SINGLE LRC line with ONE
   timestamp. Do NOT split them into two lines.
5. USE OFFICIAL TEXT: The LRC text must come from the OFFICIAL LYRICS exactly.
   Do not use the Whisper text (it may contain transcription errors).
6. COVERAGE: Every non-blank official lyric line must appear in the output.
7. NO EXTRA CONTENT: Output nothing except the LRC lines and blank separators.

Example correct output:
[00:12.50] 誰も知らない　光見えない
[00:16.20] 奈落の果てを遡る
[00:19.80] 想い集めながら

[00:24.10] 忘れ去られた　時の彼方に"""

    else:
        # Fallback: no DB lyrics — clean and reformat Whisper output directly
        system_prompt = (
            "You are a professional LRC synchronization agent. "
            "Output ONLY the LRC content — no explanations, no markdown, no code blocks."
        )

        user_prompt = f"""TASK: Clean and reformat this Whisper STT output into valid LRC format.

═══════════════════════════════
WHISPER STT:
═══════════════════════════════
{stt_context}

═══════════════════════════════
STRICT OUTPUT RULES:
═══════════════════════════════
1. FORMAT: [MM:SS.xx] lyric text
2. Keep timestamps exactly as given. Only clean the text (fix obvious STT errors).
3. Group lines into verses with blank lines between natural lyric sections.
4. Do NOT merge or split lines — one Whisper segment = one LRC line.
5. Output nothing except LRC lines and blank separators."""

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        model="llama-3.3-70b-versatile",
        temperature=0,          # deterministic — this is a structured mapping task
        max_tokens=4096,
    )

    raw = response.choices[0].message.content.strip()
    return _clean_llm_lrc_output(raw)


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2: THAI TRANSLATION
# ─────────────────────────────────────────────────────────────────────────────

def run_thai_translation(lrc_content: str) -> str:
    """
    Translate Japanese LRC lyrics into natural, poetic Thai.

    Preserves:
      - All timestamps exactly ([MM:SS.xx])
      - Blank lines between verses
      - LRC structure (one line per timestamp)

    Translation target: natural spoken Thai, not literal. Poetic where possible.
    """
    client = get_client()

    system_prompt = (
        "You are a professional Japanese-to-Thai song translator. "
        "You produce natural, poetic Thai lyrics that preserve the emotional "
        "tone of the original. Output ONLY the translated LRC — no explanations."
    )

    user_prompt = f"""TASK: Translate the Japanese lyrics in this LRC file into natural Thai.

═══════════════════════════════
INPUT LRC (Japanese):
═══════════════════════════════
{lrc_content}

═══════════════════════════════
STRICT OUTPUT RULES:
═══════════════════════════════
1. TIMESTAMPS: Copy every timestamp [MM:SS.xx] EXACTLY as-is. Do not change them.
2. TRANSLATION: Replace only the Japanese lyric text with Thai translation.
3. BLANK LINES: Keep all blank lines exactly where they are (verse separators).
4. TONE: Natural, poetic Thai. Not word-for-word literal. Preserve emotion and rhythm.
5. ONE LINE IN = ONE LINE OUT: Do not merge or split lines.
6. NO EXTRA CONTENT: Output only the translated LRC. No notes, no explanations.

Example:
  Input:  [00:12.50] 誰も知らない　光見えない
  Output: [00:12.50] ไม่มีใครรู้　แสงสว่างที่มองไม่เห็น"""

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        model="openai/gpt-oss-120b",    # better multilingual than openai/gpt-oss-120b on Groq
        temperature=0.4,        # slight creativity for natural translation, not fully deterministic
        max_tokens=4096,
    )

    raw = response.choices[0].message.content.strip()
    return _clean_llm_lrc_output(raw)


# ─────────────────────────────────────────────────────────────────────────────
# POST-PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def _clean_llm_lrc_output(raw: str) -> str:
    """
    Strip common LLM output contamination from LRC content.

    LLMs sometimes wrap output in markdown code blocks or add preamble text
    even when told not to. This strips those safely without touching the LRC.

    Also normalises:
      - Windows line endings → Unix
      - Multiple consecutive blank lines → single blank line
      - Leading/trailing whitespace
    """
    # Strip markdown code fences (```lrc ... ``` or ``` ... ```)
    raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"^```$",         "", raw, flags=re.MULTILINE)

    # Strip any line that doesn't look like LRC or a blank separator
    # (catches preambles like "Here is the LRC output:")
    lines         = raw.splitlines()
    lrc_pattern   = re.compile(r"^\[\d{2}:\d{2}\.\d{2}\]")
    filtered      = []
    in_lrc        = False

    for line in lines:
        if lrc_pattern.match(line):
            in_lrc = True           # found first real LRC line
        if in_lrc:
            filtered.append(line)   # keep everything from first LRC line onward

    result = "\n".join(filtered)

    # Normalize line endings and collapse multiple blank lines
    result = result.replace("\r\n", "\n").replace("\r", "\n")
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()