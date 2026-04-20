import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

from modules.downloader import download_and_separate
from modules.transcriber import run_transcription
from modules.query_agent import TouhouQueryAgent
from modules.reconciler import run_main_reconciliation, run_thai_translation
from modules.furigana_logic import run_local_furigana_logic
from modules.lrctoass import lrc_to_ass_final

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(page_title="EchoSync AI", layout="wide")

@st.cache_resource
def init_agent():
    return TouhouQueryAgent()

agent = init_agent()

# ── Session State Init ────────────────────────────────────────────────────────
# FIX: Store all pipeline outputs in session_state so they survive reruns
# triggered by clicking download buttons. Without this, clicking any button
# causes a rerun → tempfile.TemporaryDirectory exits → all files deleted.
if "pipeline_outputs" not in st.session_state:
    st.session_state.pipeline_outputs = None

# ── User Input Section ────────────────────────────────────────────────────────
st.title("EchoSync: Multimodal AI Lyric Agent")
st.markdown("### 3rd Year Engineering Project | Agentic Workflow")

with st.container():
    st.subheader("Input and Configuration")
    video_url = st.text_input(
        "Enter YouTube Music Video URL:",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Paste the link to the music video here.",
    )
    start_btn = st.button("Start Agentic Pipeline")

# ── Execution Logic ───────────────────────────────────────────────────────────
if start_btn:
    if not video_url:
        st.warning("Please enter a URL first.")
    else:
        # Clear previous run's outputs before starting a new pipeline
        st.session_state.pipeline_outputs = None

        with tempfile.TemporaryDirectory() as tmp_dir:

            # Stage 1: Audio Extraction
            with st.status("Stage 1: Downloading and Extracting Metadata...") as status:
                vocal_path, yt_meta = download_and_separate(video_url, tmp_dir)
                st.write(f"Target: {yt_meta.get('title')}")
                status.update(label="Metadata Extracted", state="complete")

            # Stage 2: Database Search
            with st.status("Stage 2: Searching Database...") as status:
                db_result = agent.search_lyrics(yt_meta)
                db_lyrics = db_result["lyrics"] if db_result else None
                if db_result:
                    st.write(
                        f"DB Match: **{db_result['meta'].get('artist')} — "
                        f"{db_result['meta'].get('title')}** "
                        f"(method: {db_result.get('method')}, "
                        f"reason: {db_result.get('reason', 'n/a')})"
                    )
                else:
                    st.write("No DB match found — Whisper transcript will be used as source.")
                status.update(label="Database Search Complete", state="complete")

            # Stage 3: Transcription
            with st.status("Stage 3: Transcribing Vocals (Whisper)...") as status:
                whisper_results = run_transcription(vocal_path)
                status.update(label="Transcription Complete", state="complete")

            # Stage 4: LLM Alignment
            with st.status("Stage 4: LLM Alignment (Llama 3.3)...") as status:
                lrc_base = run_main_reconciliation(whisper_results, db_lyrics)
                status.update(label="LRC Generated", state="complete")

            # Stage 5: Multi-language Post-Processing
            with st.status("Stage 5: Generating Multi-Language Versions...") as status:
                lrc_furi = run_local_furigana_logic(lrc_base)
                lrc_thai = run_thai_translation(lrc_base)
                status.update(label="All Versions Ready", state="complete")

            # Stage 6: ASS Formatting
            # FIX: Read all file bytes into memory NOW, while tmp_dir still exists.
            # After this with-block exits the directory is deleted — we must have
            # everything in RAM (session_state) before that happens.
            ass_paths = {
                "std":  os.path.join(tmp_dir, "std.ass"),
                "furi": os.path.join(tmp_dir, "furi.ass"),
                "thai": os.path.join(tmp_dir, "thai.ass"),
            }
            lrc_to_ass_final(lrc_base, ass_paths["std"])
            lrc_to_ass_final(lrc_furi, ass_paths["furi"])
            lrc_to_ass_final(lrc_thai, ass_paths["thai"])

            # Read bytes from disk into session_state before tmp_dir is wiped
            st.session_state.pipeline_outputs = {
                # ASS files — read as bytes for st.download_button
                "std_ass":  open(ass_paths["std"],  "rb").read(),
                "furi_ass": open(ass_paths["furi"], "rb").read(),
                "thai_ass": open(ass_paths["thai"], "rb").read(),
                # LRC files — already strings, store directly
                "std_lrc":  lrc_base,
                "furi_lrc": lrc_furi,
                "thai_lrc": lrc_thai,
                # Keep meta for display
                "song_title": yt_meta.get("title", "output"),
            }
        # TemporaryDirectory is now deleted — but all data is safe in session_state

# ── Output Section ────────────────────────────────────────────────────────────
# FIX: Render download buttons OUTSIDE the pipeline block and OUTSIDE the
# tempfile context. They read from session_state, which persists across reruns.
# Clicking a button now just reruns the script — session_state still has the data.
if st.session_state.pipeline_outputs is not None:
    out = st.session_state.pipeline_outputs
    song = out["song_title"]

    st.divider()
    st.success("Pipeline execution successful. Download files below:")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("#### Standard (JP)")
        st.download_button(
            label="⬇ Download Standard ASS",
            data=out["std_ass"],
            file_name=f"{song}_standard_jp.ass",
            mime="text/plain",
            key="dl_std_ass",
        )
        st.download_button(
            label="⬇ Download Standard LRC",
            data=out["std_lrc"],
            file_name=f"{song}_standard_jp.lrc",
            mime="text/plain",
            key="dl_std_lrc",
        )

    with col2:
        st.success("#### Furigana (Reading)")
        st.download_button(
            label="⬇ Download Furigana ASS",
            data=out["furi_ass"],
            file_name=f"{song}_furigana_jp.ass",
            mime="text/plain",
            key="dl_furi_ass",
        )
        st.download_button(
            label="⬇ Download Furigana LRC",
            data=out["furi_lrc"],
            file_name=f"{song}_furigana_jp.lrc",
            mime="text/plain",
            key="dl_furi_lrc",
        )

    with col3:
        st.warning("#### Thai (Translated)")
        st.download_button(
            label="⬇ Download Thai ASS",
            data=out["thai_ass"],
            file_name=f"{song}_thai_translation.ass",
            mime="text/plain",
            key="dl_thai_ass",
        )
        st.download_button(
            label="⬇ Download Thai LRC",
            data=out["thai_lrc"],
            file_name=f"{song}_thai_translation.lrc",
            mime="text/plain",
            key="dl_thai_lrc",
        )

    st.caption(
        "Workspace purged. No temporary data remains on the host machine. "
        "Outputs above are held in session memory until you run the pipeline again."
    )