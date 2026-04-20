import chromadb
import re
import os
from groq import Groq
from chromadb.utils import embedding_functions


class TouhouQueryAgent:
    def __init__(self, db_path="./touhou_vectordb", groq_api_key: str | None = None):
        # ── ChromaDB ──────────────────────────────────────────────────────────
        self.client = chromadb.PersistentClient(path=db_path)
        self.ef = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_collection(
            name="touhou_lyrics_vault",
            embedding_function=self.ef,
        )

        # ── Groq Client ───────────────────────────────────────────────────────
        # Reads GROQ_API_KEY from env if not passed explicitly
        self.groq = Groq(api_key=groq_api_key or os.environ.get("API_KEY"))
        self.groq_model = "llama-3.3-70b-versatile"   # best Groq model for JP translation

        # ── Thresholds ────────────────────────────────────────────────────────
        self.THRESHOLD_SKIP_LLM    = 0.35   # below → confident match, skip Groq entirely
        self.THRESHOLD_TRIGGER_LLM = 0.60   # above → likely EN/JP mismatch, call Groq
        self.THRESHOLD_SCOPED      = 0.55   # max accept score when circle is known
        self.THRESHOLD_BROAD       = 0.35   # max accept score for broad fallback

        # ── Alias Map ─────────────────────────────────────────────────────────
        # All values are PRE-SLUGGED — must match what get_slug() returns
        # for the canonical DB artist name. Use debug_slug() to verify.
        self.ALIAS_MAP = {
            # 君の美術館
            "君の美術館":        "kiminomiujiamu",
            "キミノミュージアム": "kiminomiujiamu",
            "kiminomuseum":      "kiminomiujiamu",
            # ゼッケン屋
            "ゼッケン屋":        "zekkenuya",
            "sekkenya":          "zekkenuya",
            # いえろ～ぜぶら
            "いえろぜぶら":      "ierozemubura",
            "yellowzebra":       "ierozemubura",
            # 豚乙女
            "豚乙女":            "butaotome",
            "butao":             "butaotome",
            # SYNC.ART'S
            "syncarts":          "syncarts",
            # Alstroemeria Records
            "alstro":            "alstroemeriarecords",
            "alstroemeria":      "alstroemeriarecords",
            # XL Project
            "xlproj":            "xlproject",
            "xlp":               "xlproject",
            # SoundOnline
            "soundonline":       "soundonline",
            # MN-logic24
            "mnlogic":           "mnlogic24",
            # Liz Triangle
            "liz":               "liztriangle",
            "liztri":            "liztriangle",
            # C-CLAYS
            "cclays":            "cclays",
        }

        # ── Noise Patterns ────────────────────────────────────────────────────
        # Applied in ORDER to clean_title before vector search / Pass 0.
        # Goal: strip YT upload boilerplate while keeping the bare song title.
        # Fan upload patterns handled:
        #   【東方Vocal】  「kimino-museum」  [Touhou]  (Full ver.)  東方アレンジ
        self._NOISE_PATTERNS = [
            r"[\[【][^\]】]*[\]】]",      # [tags] 【tags】 — catches 【東方Vocal】, [Touhou]
            r"「[^」]*」",                   # 「circle name」 — most common fan upload bracket
            r"『[^』]*』",                   # 『...』 double corner quotes
            r"\([^)]*\)",                  # (Full ver.) (2024) (official) etc.
            r"\bfeat\.?\s+[\w\s・]+",   # feat. artist / ft. vocalist
            r"\b(cv|vo|vocal|arrange|arr|music|lyrics?|words?|lyric)\s*[:\.]?\s*[\w\s]+",
            r"\b(HQ|HD|MV|PV|Full|Short|ver\.?|version|off\s*vocal|instrumental|piano)\b",
            r"\b(Reitaisai|Comiket|C\d{2,3}|M3|博麗神社例大祭)\s*\d*",  # event tags
            r"\b(project)\b",              # strips "Project" from "XL Project" noise
            r"東方",                          # 東方 / 東方Project prefix (very common in fan titles)
            r"\bTouhou\b",                 # English "Touhou" prefix
            r"\bアレンジ\b",               # アレンジ (arrange) tag
            r"\bボーカル\b",               # ボーカル (vocal) tag  
            r"[♪♫★☆◆◇▶►♥「」『』]",        # stray decorative/bracket chars
            r"\s{2,}",                      # collapse leftover whitespace
        ]

        self._cache_artists()

    # ─────────────────────────────────────────────────────────────────────────
    # CORE UTILITIES
    # ─────────────────────────────────────────────────────────────────────────

    def _cache_artists(self):
        """
        Cache two structures at startup:

        1. db_artist_slug_map : {slug -> artist_name}
           Used by _detect_artist() for circle identification.

        2. circle_title_index : {artist_name -> [{id, music_name, music_slug, doc, meta}]}
           Pre-sliced title roster per circle. Used by _deterministic_scan() in
           Pass 0 — exact/fuzzy string match over the circle song list before any
           vector search or LLM call. O(n) over ~5-30 songs, perfectly reliable
           when the circle is already confirmed.
        """
        all_data = self.collection.get(include=["metadatas", "documents"])
        all_meta = all_data["metadatas"]
        all_docs = all_data["documents"]
        all_ids  = all_data["ids"]

        unique_artists = list(set(m["artist"] for m in all_meta))
        self.db_artist_slug_map: dict[str, str] = {
            self.get_slug(a): a for a in unique_artists
        }

        # Build per-circle title index with pre-sliced music names
        self.circle_title_index: dict[str, list[dict]] = {}
        for id_, meta, doc in zip(all_ids, all_meta, all_docs):
            artist     = meta["artist"]
            music_name = self._slice_music_name(meta["title"])
            entry = {
                "id":         id_,
                "music_name": music_name,
                "music_slug": self.get_slug(music_name),
                "doc":        doc,
                "meta":       meta,
            }
            self.circle_title_index.setdefault(artist, []).append(entry)

        total = sum(len(v) for v in self.circle_title_index.values())
        print(f"[INIT] Cached {len(unique_artists)} artists, {total} songs in title index.")

    def get_slug(self, text: str) -> str:
        """
        Strip all non-alphanumeric/non-CJK chars and lowercase.
        'MN-logic24' → 'mnlogic24' | "SYNC.ART'S" → 'syncarts'
        """
        if not text:
            return ""
        return re.sub(
            r"[^a-zA-Z0-9\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]",
            "",
            text,
        ).lower().strip()

    def extract_core_title(self, yt_title: str) -> str:
        """Strip YouTube noise patterns while preserving the real song title."""
        title = yt_title
        for pattern in self._NOISE_PATTERNS:
            title = re.sub(pattern, " ", title, flags=re.IGNORECASE)
        return re.sub(r"\s{2,}", " ", title).strip(" -–—")

    def _detect_artist(self, yt_context_slug: str) -> str | None:
        """Deterministic circle detection via slug + alias matching."""
        for db_slug, artist_name in self.db_artist_slug_map.items():
            if db_slug and db_slug in yt_context_slug:
                return artist_name
        for alias, target_slug in self.ALIAS_MAP.items():
            alias_slug = self.get_slug(alias)
            if alias_slug and alias_slug in yt_context_slug:
                canonical = self.db_artist_slug_map.get(target_slug)
                if canonical:
                    return canonical
        return None

    @staticmethod
    def _slice_music_name(db_title: str) -> str:
        """
        Extract the bare music name from a DB title formatted as 'Circle - Music Name'.

        Your DB format is always 'Circle - Name', e.g.:
          '君の美術館 - ラストリモート'      → 'ラストリモート'
          'Alstroemeria Records - Real Intension' → 'Real Intension'
          'ラストリモート'                    → 'ラストリモート'  (no delimiter → full string)

        Uses split(' - ', 1) so only the FIRST delimiter is used, allowing
        music names that themselves contain ' - ' to survive intact.
        """
        parts = db_title.split(" - ", 1)
        return parts[1].strip() if len(parts) == 2 else db_title.strip()

    def _rank_candidates(
        self,
        results: dict,
        yt_title: str,
        clean_title: str,
        detected_artist: str | None,
        translated_title: str | None = None,
    ) -> list[tuple[float, bool, dict, str]]:
        """
        Composite re-ranking with title-slice matching. Lower score = better match.

        Return type: list of (score, is_structural_match, meta, doc)

        is_structural_match = True when the sliced music_name exactly equals
        either the translated_title or clean_title (case-insensitive). This flag
        is the "ground truth" signal — _validate_and_return uses it to bypass
        the vector threshold entirely, because an exact string comparison is more
        reliable than any embedding distance for this format.

        Scoring flow per candidate:
          1. Start with raw vector distance as base score.
          2. Slice DB title ('Circle - Music Name') → isolate bare music_name.
          3. Run exact-match checks → set is_structural_match + apply score bonus.
          4. Run substring / fuzzy checks for non-exact cases.
          5. Apply artist confirmation and word-overlap bonuses.

        Score bonuses (structural exact matches also set the flag):
        ┌──────────────────────────────────────────────────────────┬────────┬───────┐
        │ Condition                                                │ Bonus  │ Flag  │
        ├──────────────────────────────────────────────────────────┼────────┼───────┤
        │ music_name == translated_title  (exact, case-insensitive)│ -0.80  │  ✓   │
        │ music_name == clean_title       (exact, case-insensitive)│ -0.50  │  ✓   │
        │ music_name in translated_title  (substring)              │ -0.25  │       │
        │ music_name in clean_title       (substring)              │ -0.15  │       │
        │ full DB title in raw YT title   (legacy fallback)        │ -0.20  │       │
        │ full DB title in clean_title    (legacy fallback)        │ -0.10  │       │
        │ confirmed circle artist                                  │ -0.10  │       │
        │ word overlap ≥ 2 words          (per word)               │ -0.05× │       │
        └──────────────────────────────────────────────────────────┴────────┴───────┘
        """
        scored      = []
        yt_lower    = yt_title.lower()
        clean_lower = clean_title.lower()
        trans_lower = translated_title.lower() if translated_title else ""

        for meta, dist, doc in zip(
            results["metadatas"][0],
            results["distances"][0],
            results["documents"][0],
        ):
            score                = dist
            db_title_lower       = meta["title"].lower()
            is_structural_match  = False   # ← innocent until proven guilty

            # ── TITLE SLICING ─────────────────────────────────────────────────
            # DB format confirmed from bulkimporter.py:
            #   meta["title"] = f"{artist} - {title}"   e.g. '君の美術館 - ラストリモート'
            # Slicing strips the circle prefix so comparisons hit only the song name.
            music_name = self._slice_music_name(meta["title"]).lower()

            # ── EXACT MATCH (structural) ──────────────────────────────────────
            if trans_lower and music_name == trans_lower:
                # Perfect JP translation match — string identity, not vector similarity.
                # Vector distance is irrelevant here; this IS the correct song.
                is_structural_match = True
                score -= 0.80
                print(
                    f"  [SLICE] *** Structural JP match *** "
                    f"{music_name!r} == {trans_lower!r}  "
                    f"(raw_dist={dist:.3f}, flag=TRUE)"
                )

            elif clean_lower and music_name == clean_lower:
                # Perfect EN clean-title match — equally definitive.
                is_structural_match = True
                score -= 0.50
                print(
                    f"  [SLICE] *** Structural EN match *** "
                    f"{music_name!r} == {clean_lower!r}  "
                    f"(raw_dist={dist:.3f}, flag=TRUE)"
                )

            # ── SUBSTRING / FUZZY (non-structural) ───────────────────────────
            elif trans_lower and music_name in trans_lower:
                score -= 0.25
            elif clean_lower and music_name in clean_lower:
                score -= 0.15

            # ── LEGACY WHOLE-TITLE FALLBACK ───────────────────────────────────
            # Fires only when none of the slice checks matched — handles the rare
            # case where ' - ' delimiter is absent in DB title.
            else:
                if db_title_lower in yt_lower:
                    score -= 0.20
                elif db_title_lower in clean_lower:
                    score -= 0.10

            # ── ARTIST CONFIRMATION ───────────────────────────────────────────
            # meta["artist"] is the bare circle name per bulkimporter.py
            if detected_artist and meta["artist"] == detected_artist:
                score -= 0.10

            # ── WORD OVERLAP DISAMBIGUATION ───────────────────────────────────
            # Prevents "Last Remote" from matching "Last Day Never Knows"
            db_words    = set(music_name.split())
            clean_words = set(clean_lower.split())
            overlap     = len(db_words & clean_words)
            if overlap >= 2:
                score -= 0.05 * overlap

            scored.append((score, is_structural_match, meta, doc))

        # Sort by score ascending; structural matches naturally bubble to top
        # because of the large bonuses, but the flag is what _validate_and_return
        # actually acts on — not the final score alone.
        return sorted(scored, key=lambda x: x[0])

    # ─────────────────────────────────────────────────────────────────────────
    # PASS 0: DETERMINISTIC TITLE SCAN
    # ─────────────────────────────────────────────────────────────────────────

    def _strip_circle_from_title(self, clean_title: str, detected_artist: str) -> str:
        """
        Remove the circle name from clean_title wherever it appears.

        Fan upload titles put the circle name ANYWHERE:
          Left prefix : 'SoundOnline - Hydrangea'
          Right suffix: 'At the End of Spring kimino-museum'
          Middle      : '東方 SoundOnline Hydrangea arrange'
          Already gone: 'At the End of Spring'  (brackets stripped by extract_core_title)

        Strategy:
          1. Try removing literal artist name from anywhere (not left-anchored).
          2. Try removing slug form to handle spacing variants.
          3. If nothing matched, return the title unchanged — the bare query
             will still work for Pass 0 Tier 1 if the song name is all that remains.

        Examples:
          'Sound Online - Hydrangea'        + 'SoundOnline'  -> 'Hydrangea'
          'At the End of Spring'            + '君の美術館'    -> 'At the End of Spring'
          'Touhou kimino-museum Spring Song'+ '君の美術館'    -> 'Touhou  Spring Song'
          'Hydrangea SoundOnline arrange'   + 'SoundOnline'  -> 'Hydrangea  arrange'
        """
        title = clean_title.strip()

        # Step 1: Remove literal artist name from anywhere in the string
        pattern = re.compile(
            r'\s*' + re.escape(detected_artist) + r'\s*',
            re.IGNORECASE
        )
        stripped = pattern.sub(' ', title).strip(' -–—')
        if stripped and stripped != title:
            # Collapse multiple spaces left by removal
            return re.sub(r' {2,}', ' ', stripped).strip()

        # Step 2: Slug-based removal for spacing variants
        # e.g. 'Sound Online' vs 'SoundOnline' — slugs both to 'soundonline'
        # We rebuild the original string minus the matched slug span
        artist_slug = self.get_slug(detected_artist)
        if artist_slug:
            result_chars = []
            i = 0
            while i < len(title):
                # Check if slug of title[i:i+window] matches artist_slug
                # Use a sliding window of ±5 chars around len(detected_artist)
                window = len(detected_artist)
                for w in range(max(1, window - 5), window + 6):
                    chunk = title[i:i + w]
                    if self.get_slug(chunk) == artist_slug:
                        i += w  # skip this chunk
                        break
                else:
                    result_chars.append(title[i])
                    i += 1
            rebuilt = re.sub(r' {2,}', ' ', ''.join(result_chars)).strip(' -–—')
            if rebuilt and rebuilt != title:
                return rebuilt

        return title  # no match — return unchanged; Pass 0 will try full clean_title

    def _deterministic_scan(
        self, clean_title: str, detected_artist: str
    ) -> dict | None:
        """
        Pass 0 — Scan ALL titles for the confirmed circle deterministically.

        This bypasses vector search entirely and does three tiers of string
        matching against the pre-cached circle_title_index:

        Tier 1  Exact slug match
                get_slug(music_name) == get_slug(query)
                Handles: 'Hydrangea' == 'Hydrangea', 'hygrangea' == 'Hygrangea'
                Also handles EN title in DB matching EN title in YT.

        Tier 2  Substring slug match
                get_slug(query) in get_slug(music_name)  OR  vice versa
                Handles: 'spring' in 'springharbor', partial matches.

        Tier 3  Word overlap (≥ 50% of query words found in music_name)
                Handles: 'End of Spring' partially matching 'Spring Harbor'
                (intentionally weak — only fires if no tier 1/2 hit)

        All tiers strip the circle prefix first via _strip_circle_from_title()
        so 'Sound Online - Hydrangea' correctly becomes query='Hydrangea'.

        Returns the same dict shape as search_lyrics result, or None.
        """
        songs = self.circle_title_index.get(detected_artist, [])
        if not songs:
            return None

        # Strip circle prefix from clean_title to get bare query
        bare_query      = self._strip_circle_from_title(clean_title, detected_artist)
        bare_query_slug = self.get_slug(bare_query)

        print(f"[PASS0] Scanning {len(songs)} titles for '{detected_artist}'")
        print(f"[PASS0] Bare query: {bare_query!r}  (slug: {bare_query_slug!r})")

        tier1_hit = None
        tier2_hit = None
        tier3_hit = None
        tier3_best_overlap = 0

        for entry in songs:
            db_slug = entry["music_slug"]   # pre-computed at cache time

            # Tier 1: exact slug match
            if db_slug == bare_query_slug:
                tier1_hit = entry
                break   # can't get better than this

            # Tier 2: substring slug match (either direction)
            if bare_query_slug and (
                bare_query_slug in db_slug or db_slug in bare_query_slug
            ):
                if tier2_hit is None:   # keep first (they're sorted by insertion)
                    tier2_hit = entry

            # Tier 3: word overlap >= 50%
            query_words = set(bare_query.lower().split())
            db_words    = set(entry["music_name"].lower().split())
            if query_words and db_words:
                overlap = len(query_words & db_words)
                ratio   = overlap / max(len(query_words), len(db_words))
                if ratio >= 0.5 and overlap > tier3_best_overlap:
                    tier3_best_overlap = overlap
                    tier3_hit = entry

        # Return best tier hit
        for tier_name, hit in [
            ("tier1-exact",     tier1_hit),
            ("tier2-substring", tier2_hit),
            ("tier3-overlap",   tier3_hit),
        ]:
            if hit:
                print(
                    f"[PASS0] *** Hit ({tier_name}): ''{hit['music_name']}'' ***"
                )
                return {
                    "lyrics": hit["doc"],
                    "method": f"deterministic_{tier_name}",
                    "meta":   hit["meta"],
                    "score":  0.0,
                    "reason": "structural-match",
                }

        print(f"[PASS0] No deterministic match found for {bare_query!r}")
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # LLM TRANSLATION LAYER (Groq)
    # ─────────────────────────────────────────────────────────────────────────

    def llm_query(self, prompt: str) -> str:
        """Send a prompt to Groq and return raw text. Low temp for accuracy."""
        response = self.groq.chat.completions.create(
            model=self.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,    # deterministic → less hallucination on titles
            max_tokens=128,     # song titles are short, no need for more
        )
        return response.choices[0].message.content.strip()

    def translate_title_to_japanese(
        self, clean_title: str, detected_artist: str | None
    ) -> str | None:
        """
        Ask Groq to translate an English Touhou song title to its Japanese equivalent.

        Why this works:
          LLaMA-70b has strong Touhou music knowledge. Given the English title
          and circle name, it can often return the correct katakana/kanji title
          (e.g. 'Last Remote' + 'SoundOnline' → 'ラストリモート'), which then
          produces a much better embedding match in ChromaDB.

        Fix — two-layer contamination defence:
          Layer 1: Strict prompt explicitly prohibits circle name + brackets in output.
          Layer 2: Python post-processing strips any bracket content and circle name
                   that slipped through regardless, before the result hits ChromaDB.

        Returns None if the LLM is not confident or the result is unusable.
        """
        # FIX: Pass artist name explicitly so the prompt can tell the LLM what to EXCLUDE
        artist_label = detected_artist or "unknown circle"
        artist_ctx   = (
            f"by the Touhou doujin circle '{detected_artist}'"
            if detected_artist
            else "from the Touhou Project doujin scene"
        )

        # FIX: Prompt is now extremely strict — prohibits circle name, brackets,
        # romanization, and any non-title content. Artist name is injected explicitly
        # into both the context AND the exclusion rules so the LLM knows exactly
        # what NOT to include.
        prompt = f"""You are a Touhou Project doujin music database assistant.

TASK: Return the Japanese title of this song.
- Song (English title): "{clean_title}"
- Circle: "{artist_label}"

STRICT OUTPUT RULES — violating any rule makes your answer wrong:
1. Output the Japanese song title ONLY. Nothing else.
2. DO NOT include the circle name "{artist_label}" or any part of it.
3. DO NOT use any brackets: 「」『』【】[]（）() are all forbidden.
4. DO NOT add romanization, translations, explanations, or punctuation.
5. If you do not know the answer with high confidence, output exactly: UNKNOWN

Examples of CORRECT output:  ラストリモート
Examples of WRONG output:    ラストリモート「SoundOnline」  /  Last Remote  /  ラストリモート (Last Remote)

Japanese song title:"""

        try:
            result = self.llm_query(prompt)
        except Exception as e:
            print(f"[LLM] Groq API error: {e}")
            return None

        print(f"[LLM] Raw output for {clean_title!r}: {result!r}")

        # ── POST-PROCESSING GUARDRAIL ─────────────────────────────────────────
        # Layer 2 safety: strip contamination even if the LLM ignored the prompt.

        # Step A: Remove any bracket-enclosed content (all bracket styles)
        result = re.sub(r"「[^」]*」", "", result)    # JP corner quotes
        result = re.sub(r"『[^』]*』", "", result)    # JP double corner quotes
        result = re.sub(r"【[^】]*】", "", result)    # JP thick brackets
        result = re.sub(r"\[[^\]]*\]", "", result)   # ASCII square brackets
        result = re.sub(r"\([^)]*\)",  "", result)   # ASCII parentheses
        result = re.sub(r"（[^）]*）", "", result)   # full-width parentheses

        # Step B: Remove the circle name itself if it leaked into the output.
        # Check both the raw artist string and its slug to catch partial matches.
        if detected_artist:
            # Literal artist name (case-insensitive)
            result = re.sub(re.escape(detected_artist), "", result, flags=re.IGNORECASE)
            # Also strip slug form (e.g. 'soundonline' if artist is 'SoundOnline')
            artist_slug_pattern = re.escape(self.get_slug(detected_artist))
            result = re.sub(artist_slug_pattern, "", result, flags=re.IGNORECASE)

        # Step C: Strip any stray bracket chars, leading/trailing whitespace & dashes
        result = re.sub(r"[「」『』【】\[\]（）()]", "", result)
        result = result.strip(" 　-–—・/|")       # includes full-width space 　

        # ── VALIDATION ───────────────────────────────────────────────────────
        # Discard if: gave up, empty after cleaning, too long, or purely Latin
        if (
            not result
            or result.upper() == "UNKNOWN"
            or len(result) > 60
            or re.fullmatch(r"[a-zA-Z0-9\s\-'.,!?]+", result)   # purely Latin → useless
        ):
            print(f"[LLM] Translation discarded after cleaning: {result!r}")
            return None

        print(f"[LLM] Cleaned translation: {clean_title!r} → {result!r}")
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # SEARCH HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _run_query(
        self, query_text: str, detected_artist: str | None, n_results: int = 5
    ) -> dict:
        """Run a scoped (circle-filtered) or broad ChromaDB query."""
        if detected_artist:
            return self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where={"artist": {"$eq": detected_artist}},
                include=["documents", "metadatas", "distances"],
            )
        return self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

    def _merge_results(self, results_1: dict, results_2: dict | None) -> dict:
        """
        Merge two ChromaDB result dicts, deduplicating by document ID.
        When the same ID appears twice (EN + JP query both hit it), keep
        the lower distance value — it represents the better embedding match.
        """
        if not results_2:
            return results_1

        seen: dict[str, int] = {}
        ids_, docs_, metas_, dists_ = [], [], [], []

        pairs = zip(
            results_1["ids"][0]       + results_2["ids"][0],
            results_1["documents"][0] + results_2["documents"][0],
            results_1["metadatas"][0] + results_2["metadatas"][0],
            results_1["distances"][0] + results_2["distances"][0],
        )
        for id_, doc, meta, dist in pairs:
            if id_ in seen:
                idx = seen[id_]
                if dist < dists_[idx]:      # prefer the better-matching query
                    dists_[idx] = dist
            else:
                seen[id_] = len(ids_)
                ids_.append(id_)
                docs_.append(doc)
                metas_.append(meta)
                dists_.append(dist)

        return {
            "ids":       [ids_],
            "documents": [docs_],
            "metadatas": [metas_],
            "distances": [dists_],
        }

    def _validate_and_return(
        self,
        ranked: list[tuple[float, bool, dict, str]],
        yt_title: str,
        detected_artist: str | None,
        method: str,
    ) -> dict | None:
        """
        Gate the top-ranked candidate and return lyrics or None.

        Decision priority (first match wins):
          1. is_structural_match == True
               → IMMEDIATE SUCCESS. Bypasses threshold entirely.
                 A string-level exact match on the sliced music name is ground truth.
                 The vector distance is irrelevant — we know it's the right song.

          2. best_score < threshold
               → Fuzzy vector match was good enough within tolerance.
                 Threshold: 0.55 (circle known) / 0.35 (broad search).

          3. title_substring_hit
               → Full DB title string appears verbatim inside the raw YT title.

          4. None of the above → FAIL, return None.

        Log output clearly states which path fired so you can trace the decision
        in the console without reading the code.
        """
        if not ranked:
            print("[FAIL] Ranked list is empty.")
            return None

        best_score, is_structural_match, best_meta, best_doc = ranked[0]
        threshold           = self.THRESHOLD_SCOPED if detected_artist else self.THRESHOLD_BROAD
        title_substring_hit = best_meta["title"].lower() in yt_title.lower()

        # ── PATH 1: Structural match — bypass threshold unconditionally ───────
        if is_structural_match:
            print(
                f"[SUCCESS] {best_meta['artist']!r} — {best_meta['title']!r}\n"
                f"          method={method}  reason=structural-match  "
                f"raw_dist≈{best_score + 0.80:.3f} (threshold bypassed)"
            )
            return {
                "lyrics": best_doc,
                "method": method,
                "meta":   best_meta,
                "score":  best_score,
                "reason": "structural-match",
            }

        # ── PATH 2 & 3: Fuzzy vector / substring fallback ────────────────────
        if best_score < threshold or title_substring_hit:
            reason = "substring-hit" if title_substring_hit else "threshold"
            print(
                f"[SUCCESS] {best_meta['artist']!r} — {best_meta['title']!r}\n"
                f"          method={method}  reason={reason}  "
                f"score={best_score:.3f}  threshold={threshold}"
            )
            return {
                "lyrics": best_doc,
                "method": method,
                "meta":   best_meta,
                "score":  best_score,
                "reason": reason,
            }

        # ── PATH 4: Nothing matched ───────────────────────────────────────────
        print(
            f"[FAIL] No match for {yt_title!r}\n"
            f"       Best candidate: {best_meta['artist']!r} — {best_meta['title']!r}\n"
            f"       score={best_score:.3f}  threshold={threshold}  "
            f"structural=False  substring=False"
        )
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def search_lyrics(self, yt_metadata: dict) -> dict | None:
        """
        Full 7-Stage Retrieval Pipeline
        ─────────────────────────────────────────────────────────────────────
        Stage 1   Noise stripping      extract_core_title() cleans YT title
        Stage 2   Artist detection     slug + alias match (cached, O(n))
        Stage 3   First vector search  scoped by circle or broad fallback
        Stage 4a  Early exit           dist < 0.35 → confident, skip Groq
        Stage 4b  LLM trigger check    dist > 0.60 OR (circle known + dist > 0.35)
        Stage 5   Groq translation     EN song title → Japanese title guess
        Stage 6   Second vector search query ChromaDB with JP translated title
        Stage 7   Merge + re-rank      combine both result sets, composite score
        Stage 8   Threshold gate       accept / reject best candidate
        ─────────────────────────────────────────────────────────────────────
        Real example this solves:
          YT title   : "君の美術館 - Last Remote"
          DB entry   : meta["title"]   = "君の美術館 - ラストリモート"
                       meta["artist"]  = "君の美術館"
          Pass 1 dist: ~1.90  ← all-MiniLM-L6-v2 fails EN→JP cross-lingual
          Groq guess : "ラストリモート"
          Pass 2 dist: ~1.85  ← still high (DB stores "Circle - Title", not bare title)
          Slice check: music_name="ラストリモート" == trans_lower="ラストリモート" → TRUE
          Result     : is_structural_match=True → threshold bypassed → [SUCCESS]
        """
        yt_title: str = yt_metadata["title"]
        yt_desc: str  = yt_metadata.get("description", "")

        yt_context_slug = self.get_slug(yt_title + " " + yt_desc)
        clean_title     = self.extract_core_title(yt_title)

        print(f"\n{'='*60}")
        print(f"[SEARCH] Original : {yt_title!r}")
        print(f"[SEARCH] Cleaned  : {clean_title!r}")

        # ── STAGE 2: Artist Detection ─────────────────────────────────────────
        detected_artist = self._detect_artist(yt_context_slug)
        print(f"[STAGE2] Circle   : {detected_artist!r}")

        # ── PASS 0: Deterministic title scan (circle known only) ────────────────
        # Runs BEFORE vector search. Scans every song for the detected circle
        # with exact/substring/overlap string matching on the bare music name.
        # This handles both cases that break vector+LLM:
        #   - EN title in DB ('Hydrangea') matched by EN YT title → tier1 exact
        #   - JP title in DB ('春の湊に') partially matched by EN YT title → tier3
        # Cost: O(n) over ~5-30 songs — essentially free vs a vector call.
        if detected_artist:
            pass0_result = self._deterministic_scan(clean_title, detected_artist)
            if pass0_result:
                return pass0_result

        # ── STAGE 3: First Vector Search ─────────────────────────────────────
        query_text = clean_title if len(clean_title) > 2 else yt_title
        results_1  = self._run_query(query_text, detected_artist)

        if not results_1["ids"][0]:
            print("[FAIL] DB returned no candidates on first search.")
            return None

        best_dist_1 = results_1["distances"][0][0]
        print(f"[STAGE3] Best dist (pass 1): {best_dist_1:.3f}")

        # ── STAGE 4a: Early Exit ──────────────────────────────────────────────
        if best_dist_1 < self.THRESHOLD_SKIP_LLM:
            print(f"[STAGE4] Confident match (dist < {self.THRESHOLD_SKIP_LLM}) — skipping Groq")
            ranked = self._rank_candidates(
                results_1, yt_title, clean_title, detected_artist
            )
            method = "scoped_direct" if detected_artist else "vector_direct"
            return self._validate_and_return(ranked, yt_title, detected_artist, method)

        # ── STAGE 4b: LLM Trigger ─────────────────────────────────────────────
        # Only trigger LLM if Pass 0 + vector both failed.
        # This prevents wasting a Groq call on EN-titled songs like 'Hydrangea'
        # that Pass 0 already handles, and avoids wrong JP guesses like
        # '春の終わりに' when the real title is '春の湊に'.
        should_translate = best_dist_1 > self.THRESHOLD_TRIGGER_LLM or (
            detected_artist and best_dist_1 > self.THRESHOLD_SKIP_LLM
        )

        translated_title: str | None = None
        results_2: dict | None = None

        if should_translate:
            print(f"[STAGE4] Triggering Groq translation (dist={best_dist_1:.3f})")
            translated_title = self.translate_title_to_japanese(clean_title, detected_artist)

            # ── STAGE 5 + 6: JP Query ────────────────────────────────────────
            if translated_title:
                results_2 = self._run_query(translated_title, detected_artist)
                if results_2["ids"][0]:
                    print(f"[STAGE6] Best dist (pass 2 / JP): {results_2['distances'][0][0]:.3f}")
                else:
                    print("[STAGE6] JP query returned no results.")
                    results_2 = None
        else:
            print(f"[STAGE4] Groq skipped (dist={best_dist_1:.3f})")

        # ── STAGE 7: Merge + Re-rank ──────────────────────────────────────────
        merged = self._merge_results(results_1, results_2)
        ranked = self._rank_candidates(
            merged, yt_title, clean_title, detected_artist, translated_title
        )

        # ── STAGE 8: Threshold Gate ───────────────────────────────────────────
        if translated_title:
            method = "llm_translation"
        elif detected_artist:
            method = "scoped_slug"
        else:
            method = "vector_fallback"

        return self._validate_and_return(ranked, yt_title, detected_artist, method)

    # ─────────────────────────────────────────────────────────────────────────
    # MAINTENANCE / DEBUG
    # ─────────────────────────────────────────────────────────────────────────

    def refresh_artist_cache(self):
        """Re-cache artists after adding new songs to DB at runtime."""
        self._cache_artists()

    def debug_slug(self, text: str) -> str:
        """
        Print and return what get_slug produces for any string.
        Use this to correctly fill ALIAS_MAP values.

        Example:
            agent.debug_slug("君の美術館")   # → 'kiminomiujiamu'
            agent.debug_slug("Liz Triangle") # → 'liztriangle'
        """
        slug = self.get_slug(text)
        print(f"  Input : {text!r}")
        print(f"  Slug  : {slug!r}")
        return slug