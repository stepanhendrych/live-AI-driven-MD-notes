import json
import queue
import re
import subprocess
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

from .vector_memory import VectorMemory


class NotesProcessorWorker(threading.Thread):
    """Consumes transcripts, asks Ollama for structured notes, appends non-duplicates."""

    # Maximum characters of a similar-notes snippet included in the LLM prompt
    _SIMILAR_NOTES_MAX_CHARS = 500

    def __init__(
        self,
        transcript_queue,
        stop_event,
        notes_file,
        file_lock,
        ollama_model="llama3.2:3b",
        context_lines=50,
        heading_lines=10,
        ollama_base_url="http://localhost:11434",
        ollama_timeout=30,
        ollama_retries=2,
        retry_backoff_seconds=2.0,
        ollama_keep_alive="5m",
        ollama_temperature=0.2,
        ollama_top_p=0.9,
        ollama_top_k=40,
        ollama_repeat_penalty=1.1,
        ollama_num_predict=768,
        ollama_num_ctx=4096,
        ollama_seed=42,
        structured_facts_extraction=True,
        feedback_preview_chars=180,
        max_context_chars=6000,
        min_bullet_points=6,
        max_section_lines=60,
        prompt_mode="balanced",
        two_phase_extraction=True,
        max_required_facts=12,
        enforce_fact_coverage=True,
        preserve_note_style=True,
        adaptive_speed_mode=True,
        short_transcript_chars=220,
        coverage_repair_min_missing=3,
        coverage_repair_min_missing_ratio=0.35,
        quality_repair_min_chars=140,
        enable_vector_memory=True,
        vector_memory_persist_dir=".notes_vector_db",
        vector_memory_similarity_threshold=0.05,
        vector_memory_context_results=3,
        enable_wiki_links=True,
    ):
        super().__init__(daemon=True)
        self.transcript_queue = transcript_queue
        self.stop_event = stop_event
        self.notes_file = Path(notes_file)
        self.file_lock = file_lock
        self.ollama_model = ollama_model
        self.context_lines = context_lines
        self.heading_lines = heading_lines
        self.ollama_base_url = str(ollama_base_url).rstrip("/")
        self.ollama_timeout = ollama_timeout
        self.ollama_retries = max(0, int(ollama_retries))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self.ollama_keep_alive = str(ollama_keep_alive)
        self.ollama_temperature = float(ollama_temperature)
        self.ollama_top_p = float(ollama_top_p)
        self.ollama_top_k = int(ollama_top_k)
        self.ollama_repeat_penalty = float(ollama_repeat_penalty)
        self.ollama_num_predict = int(ollama_num_predict)
        self.ollama_num_ctx = int(ollama_num_ctx)
        self.ollama_seed = int(ollama_seed)
        self.structured_facts_extraction = bool(structured_facts_extraction)
        self.feedback_preview_chars = max(60, int(feedback_preview_chars))
        self.max_context_chars = max(500, int(max_context_chars))
        self.min_bullet_points = max(1, int(min_bullet_points))
        self.max_section_lines = max(10, int(max_section_lines))
        self.prompt_mode = str(prompt_mode).strip().lower()
        self.two_phase_extraction = bool(two_phase_extraction)
        self.max_required_facts = max(4, int(max_required_facts))
        self.enforce_fact_coverage = bool(enforce_fact_coverage)
        self.preserve_note_style = bool(preserve_note_style)
        self.adaptive_speed_mode = bool(adaptive_speed_mode)
        self.short_transcript_chars = max(80, int(short_transcript_chars))
        self.coverage_repair_min_missing = max(1, int(coverage_repair_min_missing))
        self.coverage_repair_min_missing_ratio = min(1.0, max(0.0, float(coverage_repair_min_missing_ratio)))
        self.quality_repair_min_chars = max(80, int(quality_repair_min_chars))
        self.enable_wiki_links = bool(enable_wiki_links)
        self.recent_outputs = deque(maxlen=20)
        self._metrics = {
            "total_transcripts": 0,
            "phase1_success": 0,
            "phase1_fail": 0,
            "phase2_success": 0,
            "phase2_fail": 0,
            "single_phase_success": 0,
            "single_phase_fail": 0,
            "repair_success": 0,
            "repair_fail": 0,
            "quality_ok_initial": 0,
            "quality_ok_after_repair": 0,
            "duplicates_skipped": 0,
            "appended": 0,
            "total_bullets": 0,
            "total_chars": 0,
            "coverage_repairs": 0,
            "coverage_repair_fail": 0,
        }
        if enable_vector_memory:
            self._vector_memory = VectorMemory(
                persist_dir=vector_memory_persist_dir,
                collection_name="notes",
            )
            self._vector_similarity_threshold = float(vector_memory_similarity_threshold)
            self._vector_context_results = max(1, int(vector_memory_context_results))
        else:
            self._vector_memory = None
            self._vector_similarity_threshold = 0.05
            self._vector_context_results = 3

    def _clean_transcript(self, transcript):
        text = transcript or ""
        # Remove common speech fillers to improve signal-to-noise for the LLM.
        text = re.sub(r"\b(ehm+|hmm+|jakoby|prost[eě]|vlastn[eě]|tak[eé])\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\.{2,}", ". ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _strip_think_and_fences(self, text):
        cleaned = text or ""
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"^```(?:markdown)?\s*", "", cleaned.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.IGNORECASE)
        return cleaned.strip()

    def _normalize_markdown(self, text):
        lines = (text or "").splitlines()
        converted_bold_headings = []
        normalized = []
        for line in lines:
            # Normalize list marker to '-' and keep heading spacing consistent.
            line = re.sub(r"^\s*\*\s+", "- ", line)
            line = re.sub(r"^\s*\+\s+", "- ", line)
            bold_heading = re.match(r"^\s*\*\*([^*][^*]*?)\*\*\s*$", line)
            if bold_heading:
                converted_bold_headings.append(bold_heading.group(1).strip())
                line = bold_heading.group(1).strip()
            line = re.sub(r"^(#{2,6})([^\s#])", r"\1 \2", line)
            normalized.append(line.rstrip())

        # If model produced bold pseudo-headings, convert them to markdown headings.
        has_h2 = any(re.match(r"^##\s+", l) for l in normalized)
        if converted_bold_headings:
            rebuilt = []
            first_promoted = False
            for line in normalized:
                candidate = line.strip()
                if candidate in converted_bold_headings:
                    if not has_h2 and not first_promoted:
                        rebuilt.append(f"## {candidate}")
                        first_promoted = True
                    else:
                        rebuilt.append(f"### {candidate}")
                    continue
                rebuilt.append(line)
            normalized = rebuilt

        return "\n".join(normalized).strip()

    def _build_style_constraints(self, context):
        if not self.preserve_note_style:
            return "- Drz konzistentni markdown styl s dosavadnimi poznamkami."

        heading_names = []
        for raw in (context or "").splitlines():
            line = raw.strip()
            if not re.match(r"^#{2,3}\s+", line):
                continue
            name = re.sub(r"^#{2,3}\s+", "", line).strip()
            if name and name not in heading_names:
                heading_names.append(name)

        if not heading_names:
            return "- Drz konzistentni markdown styl s dosavadnimi poznamkami (nazvy sekci i ton)."

        preferred = heading_names[-8:]
        preferred_list = "\n".join(f"- {item}" for item in preferred)
        return (
            "- Dodrz styl uz existujicich poznamek a navazuj na jejich strukturu.\n"
            "- Pouzivej skutecne markdown nadpisy (##, ###), ne samostatne tucne radky jako pseudo-nadpisy.\n"
            "- Pro nove bloky pouzij `##` pro hlavni tema a `###` pro podsekce; nepouzivej `#` pro nove casti.\n"
            "- Pouzivej stejne nebo velmi podobne nazvy nadpisu jako v kontextu.\n"
            "- Vyhni se generickym sekcim typu Klicove body/Kontext, pokud v poznamkach nejsou bezne.\n"
            "- Zachovej stejny jazyk jako transcript (neprepinat mezi cestinou a anglictinou).\n"
            "- Preferovane nazvy sekci z existujicich poznamek:\n"
            f"{preferred_list}"
        )

    def _extract_heading_levels(self, lines):
        levels = []
        for raw in lines:
            m = re.match(r"^(#{1,6})\s+", raw.strip())
            if m:
                levels.append(len(m.group(1)))
        return levels

    def _apply_style_profile(self, text, context):
        """Align generated heading hierarchy with heading conventions from existing notes."""
        if not self.preserve_note_style:
            return text

        lines = (text or "").splitlines()
        if not lines:
            return text

        context_lines = (context or "").splitlines()
        context_levels = self._extract_heading_levels(context_lines)
        context_h3 = sum(1 for lvl in context_levels if lvl == 3)
        context_h2 = sum(1 for lvl in context_levels if lvl == 2)

        # In appended sections, top-level '#' headings are usually wrong; promote to '##'.
        adjusted = [re.sub(r"^#\s+", "## ", ln) for ln in lines]

        # If notes usually use a 2-level hierarchy (## + ###) and model produced only ##,
        # keep first ## as section title and map following ## headings to ### subsections.
        generated_h3 = sum(1 for ln in adjusted if re.match(r"^###\s+", ln.strip()))
        generated_h2_indices = [idx for idx, ln in enumerate(adjusted) if re.match(r"^##\s+", ln.strip())]
        prefers_hierarchy = context_h3 >= max(2, context_h2 // 2)
        if prefers_hierarchy and generated_h3 == 0 and len(generated_h2_indices) >= 2:
            first = generated_h2_indices[0]
            for idx in generated_h2_indices[1:]:
                if idx > first:
                    adjusted[idx] = re.sub(r"^##\s+", "### ", adjusted[idx])

        return "\n".join(adjusted).strip()

    def _normalize_for_match(self, text):
        base = unicodedata.normalize("NFKD", text or "")
        base = "".join(ch for ch in base if not unicodedata.combining(ch))
        base = re.sub(r"\s+", " ", base).strip().lower()
        return base

    def _get_wiki_link_targets(self):
        """Scan the notes directory and build a mapping of link terms to wiki-link targets.

        Returns a list of ``(compiled_pattern, link_str)`` tuples sorted longest-term first,
        ready for use in ``_add_wiki_links``.  Each pattern matches whole words case-insensitively.
        """
        raw_targets = {}  # term_lower -> link_str
        notes_dir = self.notes_file.parent
        try:
            md_files = sorted(notes_dir.glob("*.md"))
        except Exception:
            return []

        for md_file in md_files:
            try:
                if md_file.resolve() == self.notes_file.resolve():
                    continue
            except Exception:
                continue

            stem = md_file.stem
            link = f"[[{stem}]]"

            # Match the full stem as-is (e.g. "Shakespeare")
            if len(stem) >= 4:
                raw_targets[stem.lower()] = link

            # Also produce human-readable sub-words by splitting on dashes/underscores
            # and on CamelCase boundaries (e.g. "HumanismusRenesance" -> ["Humanismus", "Renesance"])
            parts = re.split(r"[-_]", stem)
            for part in parts:
                sub_parts = re.sub(r"([A-Z][a-z]+)", r" \1", part).split()
                for sub in sub_parts:
                    if len(sub) >= 4 and sub.lower() not in raw_targets:
                        raw_targets[sub.lower()] = link

        # Pre-compile patterns, sorted longest term first to prevent partial-match clobbering
        compiled = []
        for term_lower in sorted(raw_targets, key=len, reverse=True):
            pattern = re.compile(r"\b" + re.escape(term_lower) + r"\b", re.IGNORECASE)
            compiled.append((pattern, raw_targets[term_lower]))

        return compiled

    def _add_wiki_links(self, text):
        """Post-process generated markdown to add ``[[Wiki-links]]`` for known note files.

        Only the first occurrence of each link target per generated block is linked.
        Heading lines and lines that already contain ``[[`` are left untouched.
        """
        if not self.enable_wiki_links:
            return text

        targets = self._get_wiki_link_targets()
        if not targets:
            return text

        lines = text.splitlines()
        result = []
        already_linked = set()  # link strings already inserted in this block

        for line in lines:
            # Skip markdown headings and lines that already contain wiki-links
            if re.match(r"^#{1,6}\s", line) or "[[" in line:
                result.append(line)
                continue

            for pattern, link in targets:
                if link in already_linked:
                    continue
                if pattern.search(line):
                    line = pattern.sub(link, line, count=1)
                    already_linked.add(link)

            result.append(line)

        return "\n".join(result)

    def _extract_required_facts(self, transcript):
        text = self._clean_transcript(transcript)
        if not text:
            return []

        facts = []

        # Years are high-signal and easy to verify in output.
        for year in re.findall(r"\b(?:1\d{3}|20\d{2})\b", text):
            facts.append(year)

        # Quoted phrases often carry key terminology or famous citations.
        for quoted in re.findall(r'"([^"\n]{3,120})"', transcript or ""):
            cleaned = re.sub(r"\s+", " ", quoted).strip()
            if cleaned:
                facts.append(cleaned)

        # Capture title-like entities and works (1-5 words in title case).
        title_pattern = (
            r"\b([A-ZA-Z\u00C0-\u017F][a-zA-Z\u00C0-\u017F\-']*"
            r"(?:\s+[a-zA-Z\u00C0-\u017F]{1,3})?"
            r"(?:\s+[A-ZA-Z\u00C0-\u017F][a-zA-Z\u00C0-\u017F\-']*){0,4})\b"
        )
        for entity in re.findall(title_pattern, transcript or ""):
            cleaned = re.sub(r"\s+", " ", entity).strip(" .,:;!?\t\r\n")
            if len(cleaned) >= 4:
                facts.append(cleaned)

        # Preserve first occurrence order and trim noise.
        seen = set()
        result = []
        skip = {
            "novy transcript",
            "transcript",
            "teacher",
            "ucitel",
        }
        for fact in facts:
            norm = self._normalize_for_match(fact)
            if not norm or norm in seen or norm in skip:
                continue
            seen.add(norm)
            result.append(fact)
            if len(result) >= self.max_required_facts:
                break

        return result

    def _find_missing_required_facts(self, markdown_text, required_facts):
        if not required_facts:
            return []

        doc = self._normalize_for_match(markdown_text)
        missing = []
        for fact in required_facts:
            norm_fact = self._normalize_for_match(fact)
            if not norm_fact:
                continue

            if norm_fact in doc:
                continue

            # Fallback token matching for longer facts to avoid false negatives.
            tokens = [t for t in re.findall(r"[a-z0-9]{3,}", norm_fact) if t not in {"the", "and", "pro", "proti"}]
            if not tokens:
                missing.append(fact)
                continue

            matched = sum(1 for token in tokens if token in doc)
            if matched < max(1, int(len(tokens) * 0.7)):
                missing.append(fact)

        return missing

    def _markdown_quality(self, text):
        lines = (text or "").splitlines()
        heading_count = sum(1 for l in lines if re.match(r"^##\s+", l))
        bullet_count = sum(1 for l in lines if re.match(r"^-\s+", l))
        char_count = len((text or "").strip())
        return {
            "heading_count": heading_count,
            "bullet_count": bullet_count,
            "char_count": char_count,
            "ok": heading_count >= 1 and bullet_count >= self.min_bullet_points and char_count >= 180,
        }

    def _preview(self, text):
        cleaned = re.sub(r"\s+", " ", text or "").strip()
        return cleaned[: self.feedback_preview_chars]

    def _extract_think_preview(self, text):
        match = re.search(r"<think>(.*?)</think>", text or "", re.IGNORECASE | re.DOTALL)
        if not match:
            return ""
        return self._preview(match.group(1))

    def _tokenize(self, text):
        normalized = re.sub(r"\s+", " ", text.lower()).strip()
        return set(re.findall(r"[a-zA-Z0-9_\u00C0-\u017F]+", normalized))

    def _jaccard_similarity(self, a, b):
        if not a or not b:
            return 0.0
        intersection = len(a & b)
        union = len(a | b)
        if union == 0:
            return 0.0
        return intersection / union

    def _get_context(self):
        if not self.notes_file.exists():
            return ""

        with self.file_lock:
            with self.notes_file.open("r", encoding="utf-8") as handle:
                lines = handle.readlines()

        if self.context_lines <= 0:
            recent = ""
        else:
            recent = "".join(lines[-self.context_lines :])

        if self.heading_lines <= 0:
            outline = ""
        else:
            headings = [line for line in lines if line.startswith("#")]
            outline = "".join(headings[-self.heading_lines :])

        context = f"OUTLINE:\n{outline}\n\nPOSLEDNICH {self.context_lines} RADKU:\n{recent}"
        if len(context) <= self.max_context_chars:
            return context

        clipped = context[-self.max_context_chars :]
        return "...[TRUNCATED CONTEXT FOR SPEED]...\n" + clipped

    def _is_duplicate(self, new_content, context):
        if new_content in context:
            return True

        candidate_tokens = self._tokenize(new_content)
        if not candidate_tokens:
            return True

        for previous in self.recent_outputs:
            score = self._jaccard_similarity(candidate_tokens, self._tokenize(previous))
            if score >= 0.98:
                return True

        # Context-wide dedupe is intentionally omitted: it caused useful details
        # to be dropped when the topic was similar but facts were new/specific.

        # Semantic duplicate check via vector memory (tight threshold)
        if self._vector_memory is not None:
            if self._vector_memory.is_semantic_duplicate(new_content, threshold=self._vector_similarity_threshold):
                return True

        return False

    def _build_prompt(self, context, transcript, similar_notes=None):
        detail_instruction = {
            "fast": "Bud strucny a preferuj rychlost. Jen klicove body.",
            "detailed": "Bud podrobny, uved minimum obecnych tvrzeni a maximalne konkretni fakta.",
            "balanced": "Vyvaz rychlost a detail."
        }.get(self.prompt_mode, "Vyvaz rychlost a detail.")
        style_constraints = self._build_style_constraints(context)

        semantic_section = ""
        semantic_dedup_rule = "- Pokud je bod uz v poznamkach, neopakuj ho doslova, ale NEVYNECHEJ nove detaily a upresneni"
        if similar_notes:
            snippets = "\n---\n".join(doc[:self._SIMILAR_NOTES_MAX_CHARS] for doc, _ in similar_notes)
            semantic_section = (
                "\nSEMANTICKY PODOBNY EXISTUJICI OBSAH "
                "(nepakuj tyto informace – piš pouze nove detaily a upresneni):\n"
                f"{snippets}\n"
            )
            semantic_dedup_rule = (
                "- Pokud je informace jiz obsazena v sekci "
                "\"SEMANTICKY PODOBNY EXISTUJICI OBSAH\" nebo v poznamkach, neopakuj ji – uvad jen nove detaily a upresneni"
            )

        return f"""EXISTUJICI POZNAMKY:
{context}
{semantic_section}
NOVY TRANSCRIPT OD UCITELE:
\"{self._clean_transcript(transcript)}\"

PRAVIDLA:
- Priorita je ZACHYTIT vsechny dulezite informace z transcriptu (jmena, roky, dila, pojmy, definice)
{semantic_dedup_rule}
- Markdown format STRICTNE (## nadpisy, - bullet points)
- Min {self.min_bullet_points} bullet pointu, pokud transcript obsahuje dostatek informaci
- Max ~{self.max_section_lines} radku na sekci
- Zadne emoji
- Zadne tabulky (jen bullet-pointy)
- Jednoradkove definice
- Pouzij cistou cestinu (bez azbuky), oprav zjevne preklepy a nesmyslne nazvy sekci
{style_constraints}
- Pokud transcript neobsahuje nic noveho/relevantniho, vrat prazdny radek
- {detail_instruction}

ODPOVED (pouze text k appendnuti):"""

    def _build_extraction_prompt(self, context, transcript, similar_notes=None):
        """Phase 1 prompt: extract raw facts from transcript as a numbered list."""
        semantic_section = ""
        semantic_dedup_rule = "Pokud je cast informace uz v poznamkach, stejne zachyt nove detaily, upresneni a souvislosti."
        if similar_notes:
            snippets = "\n---\n".join(doc[:self._SIMILAR_NOTES_MAX_CHARS] for doc, _ in similar_notes)
            semantic_section = (
                "\nSEMANTICKY PODOBNY EXISTUJICI OBSAH "
                "(tyto informace jiz existuji – zachyt pouze nove detaily a upresneni):\n"
                f"{snippets}\n"
            )
            semantic_dedup_rule = (
                "Pokud je informace jiz obsazena v sekci "
                "\"SEMANTICKY PODOBNY EXISTUJICI OBSAH\" nebo v poznamkach, "
                "nezachycuj ji znovu – zachyt pouze nove detaily, upresneni a souvislosti."
            )

        return f"""EXISTUJICI POZNAMKY (kontext co uz bylo zmíneno):
{context}
{semantic_section}
NOVY TRANSCRIPT OD UCITELE:
\"{self._clean_transcript(transcript)}\"

UKOL: Zachyt vsechny dulezite informace z transcriptu.
{semantic_dedup_rule}
Nevynechavej konkretni fakta jen proto, ze tema uz v poznamkach existuje.
Vypiš jako jednoduchy cislovany seznam faktu. Kazdy fakt na jednom radku.
Uved konkretni jmena, data, dila, pojmy, definice, roky.
Nepouzivej markdown formatovani, jen holý text.
Pokud transcript opravdu neobsahuje zadna fakta ani upresneni, napiš: ZADNE NOVE INFORMACE

FAKTA:"""

    def _build_render_prompt(self, facts, transcript):
        """Phase 2 prompt: render extracted facts as structured markdown."""
        detail_instruction = {
            "fast": "Bud strucny a preferuj rychlost. Jen klicove body.",
            "detailed": "Bud podrobny, uved minimum obecnych tvrzeni a maximalne konkretni fakta.",
            "balanced": "Vyvaz rychlost a detail.",
        }.get(self.prompt_mode, "Vyvaz rychlost a detail.")
        style_constraints = (
            "- Dodrz styl a strukturu existujicich poznamek z kontextu, nevnucuj generickou sablonu.\n"
            "- Pouzivej skutecne markdown nadpisy (##, ###), ne samostatne tucne radky jako nadpisy.\n"
            "- Pro nove bloky pouzij `##` pro hlavni tema a `###` pro podsekce; nepouzivej `#` pro nove casti.\n"
            "- Zachovej stejny jazyk jako transcript (neprepinat mezi cestinou a anglictinou).\n"
            "- Pokud uz draft/fakta naznacuji nazvy sekci, zachovej je konzistentni."
        )

        return f"""PREFORMÁTUJ TYTO FAKTY DO STRUKTUROVANYCH MARKDOWN POZNAMEK.

EXTRAHOVANA FAKTA:
{facts}

TRANSCRIPT (pro kontext):
\"{self._clean_transcript(transcript)}\"

PRAVIDLA:
- Markdown format STRICTNE (## nadpisy, - bullet points)
- Min {self.min_bullet_points} bullet pointu
- Max ~{self.max_section_lines} radku na sekci
- Zadne emoji
- Zadne tabulky (jen bullet-pointy)
- Jednoradkove definice
- Pouzij cistou cestinu (bez azbuky), oprav zjevne preklepy a nesmyslne nazvy sekci
{style_constraints}
- {detail_instruction}

ODPOVED (pouze markdown text k appendnuti):"""

    def _build_repair_prompt(self, transcript, draft):
        return f"""UPRAV NIZE UVEDENY DRAFT TAK, ABY SPLNIL FORMAT.

TRANSCRIPT:
\"{self._clean_transcript(transcript)}\"

DRAFT:
{draft}

POZADAVKY:
- Zachovej jen informace podlozene transcriptem
- Vystup pouze jako markdown
- Minimalne 1 heading `##` a minimalne {self.min_bullet_points} bullet pointu `-`
- Nepouzivej tabulky ani emoji
- Pridat konkretni fakta (jmena, roky, dila, pojmy) kde jsou v transcriptu
- Zachovej puvodni nazvy sekci a styl poznamek, neprejmenovavej je na genericke sablony

VRAT JEN OPRAVENY MARKDOWN:"""

    def _build_coverage_repair_prompt(self, transcript, draft, missing_facts):
        missing_lines = "\n".join(f"- {item}" for item in missing_facts)
        return f"""DOPLN NIZE UVEDENY DRAFT TAK, ABY OBSAHOVAL CHYBEJICI FAKTA Z TRANSCRIPTU.

TRANSCRIPT:
\"{self._clean_transcript(transcript)}\"

SOUCASNY DRAFT:
{draft}

CHYBEJICI FAKTA (musis je pokryt, bez vymysleni):
{missing_lines}

PRAVIDLA:
- Zachovej markdown strukturu (##, ###, -)
- Nepouzij tabulky ani emoji
- Pouzij cistou cestinu a oprav zjevne preklepy
- Pokud je fakt nejisty nebo v transcriptu nejasny, oznac to jako nejiste misto vymysleni
- Zachovej puvodni nazvy sekci v draftu a styl okolnich poznamek

VRAT JEN UPRAVENY MARKDOWN:"""

    def _should_run_two_phase(self, transcript):
        if not self.two_phase_extraction:
            return False
        if not self.adaptive_speed_mode:
            return True

        cleaned = self._clean_transcript(transcript)
        return len(cleaned) >= self.short_transcript_chars

    def _should_attempt_coverage_repair(self, required_facts, missing_facts):
        if not required_facts or not missing_facts:
            return False
        if not self.adaptive_speed_mode:
            return True

        missing_count = len(missing_facts)
        ratio = missing_count / max(1, len(required_facts))
        return missing_count >= self.coverage_repair_min_missing and ratio >= self.coverage_repair_min_missing_ratio

    def _should_attempt_quality_repair(self, quality):
        if quality.get("ok"):
            return False
        if not self.adaptive_speed_mode:
            return True

        # Skip costly repair for minor misses if output is otherwise usable.
        headings = quality.get("heading_count", 0)
        bullets = quality.get("bullet_count", 0)
        chars = quality.get("char_count", 0)
        if headings < 1:
            return True
        return bullets < max(1, self.min_bullet_points - 1) or chars < self.quality_repair_min_chars

    def _ollama_options(self):
        return {
            "temperature": self.ollama_temperature,
            "top_p": self.ollama_top_p,
            "top_k": self.ollama_top_k,
            "repeat_penalty": self.ollama_repeat_penalty,
            "num_predict": self.ollama_num_predict,
            "num_ctx": self.ollama_num_ctx,
            "seed": self.ollama_seed,
        }

    def _facts_json_schema(self):
        return {
            "type": "object",
            "properties": {
                "language": {"type": "string"},
                "facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "kind": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["text", "kind", "confidence"],
                    },
                },
            },
            "required": ["language", "facts"],
        }

    def _extract_facts_from_json(self, response_text):
        try:
            payload = json.loads(response_text)
        except (json.JSONDecodeError, TypeError):
            return []

        facts = payload.get("facts", [])
        if not isinstance(facts, list):
            return []

        extracted = []
        for item in facts:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            extracted.append(text)
        return extracted

    def _run_ollama(self, prompt, timeout_seconds, format_schema=None):
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.ollama_keep_alive,
            "options": self._ollama_options(),
        }
        if format_schema is not None:
            payload["format"] = format_schema

        request = urllib.request.Request(
            url=f"{self.ollama_base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except TimeoutError as exc:
            raise subprocess.TimeoutExpired(cmd="ollama api generate", timeout=timeout_seconds) from exc
        except urllib.error.HTTPError as exc:
            err_payload = exc.read().decode("utf-8", errors="ignore")
            return subprocess.CompletedProcess(
                args=["ollama", "api", "generate"],
                returncode=1,
                stdout="",
                stderr=f"HTTP {exc.code}: {err_payload}".strip(),
            )
        except urllib.error.URLError as exc:
            return subprocess.CompletedProcess(
                args=["ollama", "api", "generate"],
                returncode=1,
                stdout="",
                stderr=f"Connection error: {exc}",
            )

        response_text = str(body.get("response", ""))
        thinking_text = str(body.get("thinking", "")).strip()
        merged_output = response_text
        if thinking_text:
            merged_output = f"<think>{thinking_text}</think>\n{response_text}"

        return subprocess.CompletedProcess(
            args=["ollama", "api", "generate"],
            returncode=0,
            stdout=merged_output,
            stderr="",
        )

    def _log_metrics(self):
        m = self._metrics
        total = m["total_transcripts"]
        if total == 0:
            return
        appended = m["appended"]
        avg_bullets = m["total_bullets"] / appended if appended else 0.0
        avg_chars = m["total_chars"] / appended if appended else 0.0
        print(
            f"[processor:metrics] transcripts={total} "
            f"appended={appended} duplicates_skipped={m['duplicates_skipped']} "
            f"avg_bullets={avg_bullets:.1f} avg_chars={avg_chars:.0f}"
        )
        if self.two_phase_extraction:
            print(
                f"[processor:metrics] phase1 ok={m['phase1_success']} fail={m['phase1_fail']} | "
                f"phase2 ok={m['phase2_success']} fail={m['phase2_fail']}"
            )
        else:
            print(
                f"[processor:metrics] single_phase ok={m['single_phase_success']} fail={m['single_phase_fail']}"
            )
        print(
            f"[processor:metrics] quality_ok_initial={m['quality_ok_initial']} "
            f"repair_success={m['repair_success']} repair_fail={m['repair_fail']} "
            f"quality_ok_after_repair={m['quality_ok_after_repair']}"
        )

    def _run_ollama_with_retry(self, prompt, label, format_schema=None):
        """Run ollama with retry logic. Returns (result, elapsed) or (None, None) on failure."""
        result = None
        timeout_error = None
        for attempt in range(self.ollama_retries + 1):
            try:
                timeout_seconds = self.ollama_timeout + attempt * 10
                started = time.monotonic()
                result = self._run_ollama(prompt, timeout_seconds, format_schema=format_schema)
                elapsed = time.monotonic() - started
                print(
                    f"[processor:{label}] attempt {attempt + 1}/{self.ollama_retries + 1} "
                    f"finished in {elapsed:.2f}s"
                )
                return result, elapsed
            except subprocess.TimeoutExpired as exc:
                timeout_error = exc
                print(
                    f"[processor:{label}] timeout on attempt {attempt + 1}/{self.ollama_retries + 1} "
                    f"(limit={self.ollama_timeout + attempt * 10}s)"
                )
            except Exception as exc:
                print(f"[processor:{label}] failed on attempt {attempt + 1}: {exc}")

            if attempt < self.ollama_retries and self.retry_backoff_seconds > 0:
                backoff = self.retry_backoff_seconds * (attempt + 1)
                print(f"[processor:{label}] retrying in {backoff:.1f}s")
                time.sleep(backoff)

        if timeout_error is not None:
            print(f"[processor:{label}] all attempts timed out")
        return None, None

    def _process_two_phase(self, context, transcript, similar_notes=None):
        """Two-phase pipeline: extract facts, then render as markdown.

        Returns the rendered markdown string, or None if the pipeline fails.
        """
        # Phase 1: extract raw facts
        extraction_prompt = self._build_extraction_prompt(context, transcript, similar_notes=similar_notes)
        print(f"[processor:phase1] extracting facts (prompt chars={len(extraction_prompt)})")
        format_schema = self._facts_json_schema() if self.structured_facts_extraction else None
        result1, _ = self._run_ollama_with_retry(extraction_prompt, "phase1", format_schema=format_schema)
        if result1 is None:
            self._metrics["phase1_fail"] += 1
            return None

        if result1.returncode != 0:
            err = (result1.stderr or "").strip()
            print(f"[processor:phase1] ollama error ({result1.returncode}): {self._preview(err)}")
            self._metrics["phase1_fail"] += 1
            return None

        facts_raw = (result1.stdout or "").strip()
        think_preview = self._extract_think_preview(facts_raw)
        if think_preview:
            print(f"[processor:phase1] think preview: {think_preview}")

        cleaned_facts_raw = self._strip_think_and_fences(facts_raw).strip()
        facts = cleaned_facts_raw
        if format_schema is not None:
            structured_facts = self._extract_facts_from_json(cleaned_facts_raw)
            if structured_facts:
                facts = "\n".join(f"- {line}" for line in structured_facts)
            else:
                print("[processor:phase1] structured parse failed, falling back to raw extraction")

        if not facts or len(facts) <= 10:
            print("[processor:phase1] extraction too short, skipping")
            self._metrics["phase1_fail"] += 1
            return None

        no_new_patterns = re.compile(
            r"(zadne|žádné|no)\s+(nove|nové|new)\s+(informace|info|fakta|facts)"
            r"|(none|nothing|nic)\s+(new|novel|nové|noveho)",
            re.IGNORECASE,
        )
        if no_new_patterns.search(facts):
            print(f"[processor:phase1] model reports no new information: {self._preview(facts)}")
            self._metrics["phase1_fail"] += 1
            return None

        print(f"[processor:phase1] extracted facts preview: {self._preview(facts)}")
        self._metrics["phase1_success"] += 1

        # Phase 2: render facts as markdown
        render_prompt = self._build_render_prompt(facts, transcript)
        print(f"[processor:phase2] rendering markdown (prompt chars={len(render_prompt)})")
        result2, _ = self._run_ollama_with_retry(render_prompt, "phase2")
        if result2 is None:
            self._metrics["phase2_fail"] += 1
            return None

        if result2.returncode != 0:
            err = (result2.stderr or "").strip()
            print(f"[processor:phase2] ollama error ({result2.returncode}): {self._preview(err)}")
            self._metrics["phase2_fail"] += 1
            return None

        raw_output = result2.stdout or ""
        think_preview = self._extract_think_preview(raw_output)
        if think_preview:
            print(f"[processor:phase2] think preview: {think_preview}")

        rendered = self._strip_think_and_fences(raw_output)
        if len(rendered) <= 10:
            print("[processor:phase2] output too short, skipping")
            self._metrics["phase2_fail"] += 1
            return None

        print(f"[processor:phase2] output preview: {self._preview(rendered)}")
        self._metrics["phase2_success"] += 1
        return rendered

    def _append_notes(self, content):
        self.notes_file.parent.mkdir(parents=True, exist_ok=True)
        with self.file_lock:
            with self.notes_file.open("a", encoding="utf-8") as handle:
                handle.write("\n" + content + "\n")
        if self._vector_memory is not None:
            self._vector_memory.add(content)

    def run(self):
        print(
            f"[processor] ollama worker ready (model={self.ollama_model}, timeout={self.ollama_timeout}s, "
            f"retries={self.ollama_retries}, two_phase={self.two_phase_extraction})"
        )

        while not self.stop_event.is_set():
            try:
                transcript = self.transcript_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if transcript is None:
                break

            self._metrics["total_transcripts"] += 1
            context = self._get_context()
            print(
                f"[processor] received transcript chars={len(transcript)}, context chars={len(context)}"
            )

            # Query vector memory for semantically similar existing notes;
            # these are used both for context-aware deduplication in the LLM prompt
            # and for the semantic duplicate check at the end.
            similar_notes = []
            if self._vector_memory is not None:
                similar_notes = self._vector_memory.query(
                    transcript, n_results=self._vector_context_results
                )
                if similar_notes:
                    print(f"[processor] vector memory: {len(similar_notes)} similar note(s) retrieved")

            new_content = None

            if self._should_run_two_phase(transcript):
                # Attempt two-phase pipeline (extract facts -> render markdown)
                new_content = self._process_two_phase(context, transcript, similar_notes=similar_notes)
                if new_content is None:
                    # Fall back to single-phase
                    print("[processor] two-phase failed, falling back to single-phase")

            if new_content is None:
                # Single-phase: combined extraction + formatting in one prompt
                prompt = self._build_prompt(context, transcript, similar_notes=similar_notes)
                print(f"[processor:single] sending prompt chars={len(prompt)}")
                result, _ = self._run_ollama_with_retry(prompt, "single")
                if result is None:
                    self._metrics["single_phase_fail"] += 1
                    self._log_metrics()
                    continue

                if result.returncode != 0:
                    err = (result.stderr or "").strip()
                    print(f"[processor:single] ollama error ({result.returncode}): {self._preview(err)}")
                    self._metrics["single_phase_fail"] += 1
                    self._log_metrics()
                    continue

                raw_output = result.stdout or ""
                think_preview = self._extract_think_preview(raw_output)
                if think_preview:
                    print(f"[processor:single] think preview: {think_preview}")

                new_content = self._strip_think_and_fences(raw_output)
                stderr_preview = self._preview(result.stderr or "")
                if stderr_preview:
                    print(f"[processor:single] stderr preview: {stderr_preview}")

                if len(new_content) <= 10:
                    print("[processor:single] output too short, skipping")
                    self._metrics["single_phase_fail"] += 1
                    self._log_metrics()
                    continue

                print(f"[processor:single] output preview: {self._preview(new_content)}")
                self._metrics["single_phase_success"] += 1

            new_content = self._normalize_markdown(new_content)
            new_content = self._apply_style_profile(new_content, context)

            if self.enforce_fact_coverage:
                required_facts = self._extract_required_facts(transcript)
                missing_facts = self._find_missing_required_facts(new_content, required_facts)
                if self._should_attempt_coverage_repair(required_facts, missing_facts):
                    print(
                        f"[processor] coverage low: missing {len(missing_facts)}/{len(required_facts)} required facts, repairing"
                    )
                    try:
                        coverage_repair = self._run_ollama(
                            self._build_coverage_repair_prompt(transcript, new_content, missing_facts[:8]),
                            max(20, self.ollama_timeout),
                        )
                        if coverage_repair.returncode == 0:
                            repaired = self._normalize_markdown(
                                self._strip_think_and_fences(coverage_repair.stdout or "")
                            )
                            repaired = self._apply_style_profile(repaired, context)
                            if repaired:
                                remaining_missing = self._find_missing_required_facts(repaired, required_facts)
                                if len(remaining_missing) < len(missing_facts):
                                    new_content = repaired
                                    self._metrics["coverage_repairs"] += 1
                                    print(
                                        f"[processor] coverage repair accepted (remaining missing {len(remaining_missing)})"
                                    )
                                else:
                                    self._metrics["coverage_repair_fail"] += 1
                                    print("[processor] coverage repair did not improve fact coverage")
                            else:
                                self._metrics["coverage_repair_fail"] += 1
                                print("[processor] coverage repair returned empty output")
                        else:
                            self._metrics["coverage_repair_fail"] += 1
                            print(f"[processor] coverage repair failed: {self._preview(coverage_repair.stderr or '')}")
                    except Exception as exc:
                        self._metrics["coverage_repair_fail"] += 1
                        print(f"[processor] coverage repair exception: {exc}")

            quality = self._markdown_quality(new_content)

            if quality["ok"]:
                self._metrics["quality_ok_initial"] += 1
            else:
                if not self._should_attempt_quality_repair(quality):
                    print(
                        "[processor] output quality slightly low but accepted for speed "
                        f"(h2={quality['heading_count']}, bullets={quality['bullet_count']}, chars={quality['char_count']})"
                    )
                else:
                    print(
                        "[processor] output quality low "
                        f"(h2={quality['heading_count']}, bullets={quality['bullet_count']}, chars={quality['char_count']}), repairing"
                    )
                try:
                    if self._should_attempt_quality_repair(quality):
                        repair = self._run_ollama(
                            self._build_repair_prompt(transcript, new_content),
                            max(15, self.ollama_timeout),
                        )
                        if repair.returncode == 0:
                            repaired = self._normalize_markdown(self._strip_think_and_fences(repair.stdout or ""))
                            repaired = self._apply_style_profile(repaired, context)
                            repaired_quality = self._markdown_quality(repaired)
                            if repaired and repaired_quality["ok"]:
                                new_content = repaired
                                self._metrics["repair_success"] += 1
                                self._metrics["quality_ok_after_repair"] += 1
                                print("[processor] repair pass accepted")
                            else:
                                self._metrics["repair_fail"] += 1
                                print("[processor] repair pass did not improve quality enough")
                        else:
                            self._metrics["repair_fail"] += 1
                            print(f"[processor] repair pass failed: {self._preview(repair.stderr or '')}")
                except Exception as exc:
                    self._metrics["repair_fail"] += 1
                    print(f"[processor] repair pass exception: {exc}")

            if self._is_duplicate(new_content, context):
                self._metrics["duplicates_skipped"] += 1
                print("[processor] duplicate detected, skipping")
                self._log_metrics()
                continue

            if self.enable_wiki_links:
                new_content = self._add_wiki_links(new_content)

            quality_final = self._markdown_quality(new_content)
            self._append_notes(new_content)
            self.recent_outputs.append(new_content)
            self._metrics["appended"] += 1
            self._metrics["total_bullets"] += quality_final["bullet_count"]
            self._metrics["total_chars"] += quality_final["char_count"]
            print(
                f"[processor] appended {quality_final['char_count']} chars, "
                f"h2={quality_final['heading_count']}, bullets={quality_final['bullet_count']}: "
                f"{new_content[:60]}..."
            )
            self._log_metrics()

        self._log_metrics()
        print("[processor] stopped")
