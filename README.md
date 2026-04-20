<p align="center">
  <img src="https://github.com/user-attachments/assets/16a36dc5-57e0-4b59-a093-7dbc5e578ff8" width="400" alt="EchoSync Logo">
</p>
# EchoSync — Agentic Lyric Synchronization System

> A 6-stage AI pipeline that converts a YouTube music URL into professionally timed subtitle files — standard Japanese, furigana-annotated, and Thai-translated — with zero manual intervention.

Built as a 3rd-year Computer Engineering project. Designed around a specialized Touhou Project doujin music dataset, but the architecture generalizes to any Japanese vocal music with a lyrics database.

---

## What it does

Paste a YouTube URL → get three download-ready subtitle files:

| Output | Format | Description |
|--------|--------|-------------|
| Standard JP | `.ass` + `.lrc` | Japanese lyrics, timestamped |
| Furigana | `.ass` + `.lrc` | Kanji annotated with hiragana readings above |
| Thai | `.ass` + `.lrc` | Translated Thai lyrics, same timestamps |

---

## Pipeline architecture

```
YouTube URL
     │
     ▼
[Stage 1] downloader.py
  yt-dlp      → WAV audio + metadata (title, description, tags)
  Demucs mdx  → vocals.wav  (VRAM-aware, OOM fallback to CPU)
     │
     ├──────────────────────────────────┐
     ▼                                  ▼
[Stage 2] query_agent.py        [Stage 3] transcriber.py
  TouhouQueryAgent                    Faster-Whisper (language=ja)
  Pass 0 → deterministic slug scan    VAD filter + word timestamps
  Pass 1 → ChromaDB vector search     VRAM-aware model selection
  Pass 2 → Groq LLM translation       OOM fallback to CPU
     │                                  │
     └────────────────┬─────────────────┘
                      ▼
             [Stage 4] reconciler.py
               run_main_reconciliation()
               Llama 3.3 (Groq) — maps DB lyrics
               to Whisper timestamps → LRC base
                      │
           ┌──────────┴──────────┐
           ▼                     ▼
   [Stage 5a]               [Stage 5b]
   furigana_logic.py        reconciler.py
   pykakasi — annotates     run_thai_translation()
   kanji with hiragana      Llama 3.3 (Groq)
   readings inline          → Thai LRC
           │                     │
           └──────────┬──────────┘
                      ▼
             [Stage 6] lrctoass.py
               lrc_to_ass_final() × 3
               Two-layer ASS ruby rendering
               (Layer 0 kanji + Layer 1 furigana)
                      │
                      ▼
         .ass + .lrc  ×  3 variants
         (served via st.session_state,
          survives Streamlit reruns)
```

---

## RAG retrieval — how the query agent works

The hardest problem in this project is matching a noisy YouTube title like
`【東方Vocal】 At the End of Spring 「kimino-museum」`
to the correct database entry `君の美術館 - 春の湊に`.

The agent runs three passes in order, stopping at the first success:

**Pass 0 — Deterministic title scan (no vectors, no API call)**

When the circle is identified via slug matching, the agent scans every song in that circle's pre-cached `circle_title_index` using three tiers:
- Tier 1: exact slug match — `get_slug(music_name) == get_slug(query)` — handles `Hydrangea == Hygrangea`, EN titles stored as EN
- Tier 2: substring slug match — either direction — handles partial titles
- Tier 3: word overlap ≥ 50% — weakest signal, last resort

The circle name is stripped from `clean_title` before scanning — `Sound Online - Hydrangea` becomes query `Hydrangea` — so the comparison hits only the bare song name.

**Pass 1 — ChromaDB vector search**

Embeds the noise-stripped title with `all-MiniLM-L6-v2` and queries the vector store. When the circle is confirmed, the query is scoped with a `where` filter so only that circle's songs are candidates.

**Pass 2 — Groq LLM translation**

When vector distance > 0.60 (EN↔JP cross-lingual mismatch is the main cause), calls `llama3-70b-8192` to translate the English title to Japanese, then re-queries ChromaDB. Results from Pass 1 and Pass 2 are merged and re-ranked by a composite score. An `is_structural_match` flag on exact string identity between `music_name` and the translated title bypasses all distance thresholds — if the strings match, it is the correct song.

---

## Furigana logic

`furigana_logic.py` uses [pykakasi](https://github.com/miurahr/pykakasi) — a pure-Python Japanese text converter — to annotate each kanji group with its hiragana reading inline in the LRC:

```
Input  : [00:12.50] 誰も知らない　光見えない
Output : [00:12.50] 誰(だれ)も知(し)らない　光(ひかり)見(み)えない
```

The `lrctoass.py` converter then renders this as proper two-layer ASS ruby — the kanji on Layer 0 (Main style) and the reading on Layer 1 above it (`\an8`, Ruby style), which displays correctly in mpv, VLC, and MPC-HC.

---

## Hardware requirements

Designed to run on consumer hardware. Tested on HP Pavilion 15 with 2 GB VRAM.

| | Minimum | Recommended |
|--|---------|-------------|
| VRAM | 2 GB | 6 GB+ |
| RAM | 8 GB | 16 GB |
| Storage | 5 GB | 10 GB |
| OS | Windows 10 / Ubuntu 20.04 / macOS 12 | — |

VRAM-aware auto-selection:

| VRAM | Demucs model | Whisper model |
|------|-------------|---------------|
| < 1.5 GB | `mdx --segment 4 --device cpu` | `base` (CPU) |
| 1.5–3 GB | `mdx --segment 8` | `small` |
| 3–6 GB | `mdx_extra` | `medium` |
| 6 GB+ | `htdemucs_ft` | `large-v3` |

Both components fall back to CPU automatically on `OutOfMemoryError`.

---

## Setup

### 1. System dependencies

```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows — download from https://ffmpeg.org and add to PATH
```

### 2. Clone and install

```bash
git clone https://github.com/your-username/echosync.git
cd echosync
```

Install PyTorch first (choose one):

```bash
# CUDA 12.1 (GPU)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then install the rest:

```bash
pip install -r requirements.txt
```

### 3. Environment variables

Create a `.env` file in the project root:

```env
API_KEY=your_groq_api_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

### 4. Add your lyrics database

Place lyric `.txt` files in `lyrics_pool/` following this naming convention exactly:

```
lyrics_pool/
  SoundOnline - Hydrangea.txt
  君の美術館 - 春の湊に.txt
  SYNC.ART'S - Lost and Found.txt
  豚乙女 - 緑の華.txt
```

**Format:** `Circle Name - Song Title.txt`

**Content:** raw lyric text, one line per sung phrase, blank lines between verses. Full-width spaces `　` within a line are supported and preserved. Example:

```
誰も知らない　光見えない
奈落の果てを遡る
想い集めながら

忘れ去られた　時の彼方に
残る貴女の影を
今も追いかけ続けている
```

### 5. Build the vector database

```bash
python bulk_importer.py
```

The importer will:
- Create `lyrics_pool/` automatically if it does not exist and prompt you to add files
- Upsert all `.txt` files into ChromaDB with metadata `{title, artist, source}`
- Print a summary of total records indexed

Re-run this script whenever you add new files. Existing entries are upserted (updated, not duplicated).

**To fully reset the database**, uncomment the `shutil.rmtree` block at the bottom of `bulk_importer.py` before running.

### 6. Run the app

```bash
streamlit run app.py
```

---

## Project structure

```
echosync/
├── app.py                  # Streamlit UI — pipeline orchestration + session_state downloads
├── bulk_importer.py        # One-time / incremental ChromaDB ingestion
├── requirements.txt
├── .env                    # Groq API key (not committed)
│
├── lyrics_pool/            # Raw .txt lyric files  ← you provide these (not committed)
├── touhou_vectordb/        # ChromaDB persistent store (not committed, auto-created)
│
└── modules/
    ├── __init__.py
    ├── downloader.py       # yt-dlp audio download + Demucs vocal separation
    ├── transcriber.py      # Faster-Whisper transcription (VRAM-aware + OOM fallback)
    ├── query_agent.py      # TouhouQueryAgent — 3-pass RAG retrieval
    ├── reconciler.py       # LRC alignment (Llama 3.3) + Thai translation
    ├── furigana_logic.py   # pykakasi — kanji → kanji(reading) annotation
    └── lrctoass.py         # LRC → .ASS subtitle converter (two-layer ruby)
```

---

## .gitignore

```gitignore
# Environment / secrets
.env

# Lyrics and database — copyright content + large binary files
lyrics_pool/
touhou_vectordb/

# Python
__pycache__/
*.pyc
*.pyo
.venv/
venv/

# Streamlit
.streamlit/secrets.toml

# OS
.DS_Store
Thumbs.db
```

> `lyrics_pool/` is excluded because lyric text is copyrighted. Populate your own from licensed sources. `touhou_vectordb/` is excluded because it is auto-generated from `lyrics_pool/` — there is no reason to version it.

---

## Known limitations

| Issue | Cause | Workaround |
|-------|-------|------------|
| Song not found | Not in `lyrics_pool/` | Add the `.txt` file and re-run `bulk_importer.py` |
| Wrong song returned | LLM translation guessed a different title (e.g. `春の終わりに` vs `春の湊に`) | Add the correct file; Pass 0 will find it before LLM is called |
| Timestamps off by one line | Whisper segmented differently from lyric line breaks | Reduce `min_silence_duration_ms` in `transcriber.py` |
| Furigana misread | pykakasi context-free, homophone kanji | Known limitation of rule-based reading — no fix without a full NLP parser |
| Ruby misaligned in ASS | Renderer cannot know glyph widths | Accepted; correct in mpv and MPC-HC, approximate in others |

---

## Tech stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| UI | Streamlit | Web interface + file downloads |
| Audio download | yt-dlp | YouTube audio extraction + metadata |
| Vocal separation | Demucs (`mdx` / `htdemucs_ft`) | Remove music, isolate vocals |
| Speech recognition | Faster-Whisper | Timestamped Japanese transcription |
| Embeddings | `all-MiniLM-L6-v2` (via ChromaDB default) | Lyric vector search |
| Vector database | ChromaDB | Persistent lyric store + similarity search |
| Furigana | pykakasi | Rule-based kanji reading annotation |
| LLM inference | Groq API — `llama-3.3-70b-versatile` | LRC alignment + Thai translation + title translation |
| Subtitle output | Custom (`lrctoass.py`) | LRC → ASS with two-layer ruby |

---

## License

MIT — see `LICENSE`.

Lyric content in `lyrics_pool/` is not included in this repository and remains the property of the respective copyright holders.
