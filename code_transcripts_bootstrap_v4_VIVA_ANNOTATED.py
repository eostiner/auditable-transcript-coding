#!/usr/bin/env python3
# =============================================================================
# Transcript Coding Helper (Thesis-Safe, Fully Logged, Percentile Reporting)
# =============================================================================
# File: code_transcripts_bootstrap_v4.py
#
# WHAT THIS SCRIPT DOES
# 1) Reads transcript PDFs that contain timestamp blocks like:
#       00:00:12,260 --> 00:00:16,540
#       some text...
#    NOTE: It also handles "OCR-ish" timestamps with spaces inside digits like:
#       00:00:12,260 --> 0 0:00:16,540
#
# 2) Splits each PDF into excerpt units:
#    - One excerpt per timestamp range (start, end, text body)
#
# 3) Assigns each excerpt to the best matching code(s) using a hybrid score:
#    A) Lexicon score:
#       - tokens from code name
#       - optional user seed synonyms (JSON)
#       - learned terms from prior runs (learned_terms.json)
#       - fuzzy token matching (misspellings and near matches)
#
#    B) TF-IDF similarity:
#       - compares excerpt text to each code's (name + description)
#
# 4) Outputs a Google Docs friendly table:
#    - <out_prefix>.tsv (paste into Google Docs)
#    - <out_prefix>.csv (archive)
#
# 5) NEW: TF-IDF explanation terms per excerpt (Top_TFIDF_Terms)
#    - For the top code only, this extracts the 1-2 gram terms that contributed
#      most to the excerpt-code similarity (based on feature-wise product).
#    - This is your best "add to codebook" signal because it reflects code
#      definition language, not just literal token hits.
#
# 6) NEW: Separate reports for cross-comparison and codebook refinement
#    A) Top hits report (excerpt-level):
#       - <out_prefix>__top_hits.tsv / .csv
#       - includes only high-confidence rows using either:
#           --report_min_score 0.95
#         OR:
#           --report_percentile 95
#
#    B) Code signals report (code-level):
#       - <out_prefix>__code_signals.tsv / .csv
#       - summarizes the most frequent Top_TFIDF_Terms across top-hit excerpts
#         per code, plus example excerpt IDs.
#
# THESIS-SAFE DESIGN
# - No hidden training loop.
# - Any lexicon growth is explicit and reviewable in learned_terms.json.
# - Optional active learning (human confirmation) is explicitly saved to
#   confirmed_labels.json.
# - All key parameters and thresholds are logged in run_log.jsonl.
#
# DEPENDENCIES
# Required:
#   pip install pdfplumber scikit-learn
# Optional (recommended for faster/better fuzzy matching):
#   pip install rapidfuzz
#
# CODEBOOK FORMAT (RECOMMENDED)
#   Codebook_Thematic_only.csv with columns:
#     Name, Description
#
# EXAMPLE RUNS
# 1) Main table only:
#   python code_transcripts_bootstrap_v4.py \
#     --codebook_csv Codebook_Thematic_only.csv \
#     --pdf_dir . \
#     --out_prefix ranch_coding_run1
#
# 2) With strict top-hits report (absolute threshold):
#   python code_transcripts_bootstrap_v4.py \
#     --codebook_csv Codebook_Thematic_only.csv \
#     --pdf_dir . \
#     --out_prefix ranch_coding_run1 \
#     --report_min_score 0.95
#
# 3) With strict top-hits report (percentile threshold):
#   python code_transcripts_bootstrap_v4.py \
#     --codebook_csv Codebook_Thematic_only.csv \
#     --pdf_dir . \
#     --out_prefix ranch_coding_run1 \
#     --report_percentile 95
#
# 4) With active learning on ambiguous excerpts:
#   python code_transcripts_bootstrap_v4.py \
#     --codebook_csv Codebook_Thematic_only.csv \
#     --pdf_dir . \
#     --out_prefix ranch_coding_run2 \
#     --active_learning
# =============================================================================

import argparse
import csv
import difflib
import glob
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

# ----------------------------
# Optional dependencies
# ----------------------------
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None

# Optional speed/quality upgrade for fuzzy matching
try:
    from rapidfuzz.fuzz import ratio as rapid_ratio
except ImportError:
    rapid_ratio = None


# =============================================================================

# =============================================================================
# VIVA ANNOTATION — SECOND-ROUND INTERVIEW CODING
# =============================================================================
# PURPOSE
# Computer-assisted mapping of timestamped second-round interview excerpts onto
# an EXISTING, human-developed thematic codebook.
#
# CORE LOGIC
# - Existing codebook = framework
# - Lexicon + TF-IDF/cosine = evidence
# - Fixed weights + thresholds = decision rule
# - Logs/exports/human review = audit trail
#
# IMPORTANT VIVA LANGUAGE
# This is best described as "computer-assisted qualitative coding" or
# "semi-automated mapping to an existing codebook."
#
# It is NOT a conventional supervised ML classifier, not unsupervised theme
# discovery, and not an LLM.
#
# DEFAULTS
# Lexicon weight 0.55
# TF-IDF weight 0.45
# Minimum assignment score 0.40
# High-confidence threshold 0.75
# Top candidate codes 2
#
# NOVELTY
# Do not claim TF-IDF, cosine similarity, fuzzy matching, or lexical matching
# are individually novel. The contribution is their integration into a
# transparent, auditable workflow for mapping a second interview round back to
# an existing community-derived codebook.
#
# KEY LIMITATIONS
# Researcher-selected weights and thresholds; no independent human-coded gold
# standard in this script; lexical/context limitations of TF-IDF; translation
# effects; timestamp segmentation; and possible feedback from iterative lexicon
# refinement.
# =============================================================================

# MODULE 1: Config defaults and regex
# =============================================================================

# Timestamp format (robust to spaces inside digits)
# Example matches:
#   00:00:12,260 --> 00:00:16,540
#   00:00:12,260 --> 0 0:00:16,540
TIMESTAMP_RE = re.compile(
    r"(?P<start>(?:\d\s*){2}:(?:\d\s*){2}:(?:\d\s*){2}[,\.](?:\d\s*){3})\s*-->\s*"
    r"(?P<end>(?:\d\s*){2}:(?:\d\s*){2}:(?:\d\s*){2}[,\.](?:\d\s*){3})"
)

# Scoring defaults
DEFAULT_FUZZY_THRESHOLD = 0.92          # stricter than earlier (reduces false hits)
DEFAULT_WEIGHT_LEXICON = 0.55
DEFAULT_WEIGHT_TFIDF = 0.45

DEFAULT_TOPK_CODES = 2
DEFAULT_MIN_ASSIGN_SCORE = 0.40         # stricter than earlier
DEFAULT_HIGH_CONF_SCORE = 0.75          # used for lexicon bootstrapping + "top hits" feel

# Active learning window (only if --active_learning)
DEFAULT_AMBIGUOUS_LOW = 0.50
DEFAULT_AMBIGUOUS_HIGH = 0.75

# Bootstrapping: candidate terms per code per run
DEFAULT_LEARN_TOP_TERMS = 12

# "Explainability": how many TF-IDF terms to show per excerpt
DEFAULT_TOP_TFIDF_TERMS_N = 10

# Reporting defaults
DEFAULT_REPORT_TOP_EXCERPTS_PER_CODE = 40   # cap per code in top-hits report
DEFAULT_CODE_SIGNALS_TOP_TERMS = 30         # number of aggregated terms per code

# Basic stopwords (English + Spanish fillers)
STOPWORDS = set("""
a an and the or but if then so to of in on at for from by with without into over under
is are was were be been being this that these those it its as
i you we they he she them us our your their
not no yes ok okay like just also
muy mas menos pero porque para con sin de del la el los las un una unos unas
y o es son fue eran ser estar estoy esta estas
""".split())

# Small alias map for common ranch transcript variants.
# Keep conservative. You can add safely, but do not overdo it.
TOKEN_ALIASES = {
    "huracan": "hurricane",
    "huracanes": "hurricanes",
    "sequia": "drought",
    "sequias": "drought",
    "bichos": "insects",
    "insectos": "insects",
    "agua": "water",
    "lluvia": "rain",
    "calor": "heat",
    "temperatura": "temperature",
}


# =============================================================================
# MODULE 2: Data structures
# =============================================================================

@dataclass
class CodeDef:
    name: str
    description: str

@dataclass
class Excerpt:
    file_name: str
    identifier: str
    start_ts: str
    end_ts: str
    text: str


# =============================================================================
# MODULE 3: Utility helpers
# =============================================================================

def now_iso_utc() -> str:
    """Timezone-aware UTC timestamp for logs."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def ensure_deps_or_die():
    if pdfplumber is None:
        raise RuntimeError("Missing dependency: pdfplumber. Install with: pip install pdfplumber")
    if TfidfVectorizer is None or cosine_similarity is None:
        raise RuntimeError("Missing dependency: scikit-learn. Install with: pip install scikit-learn")

def safe_read_json(path: str, default):
    if not path or not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_write_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def normalize_text(s: str) -> str:
    """Lowercase, remove accents (lightweight), normalize punctuation/spaces."""
    s = (s or "").strip().lower()
    s = (
        s.replace("á", "a").replace("é", "e").replace("í", "i")
         .replace("ó", "o").replace("ú", "u").replace("ñ", "n")
    )
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_timestamp(ts: str) -> str:
    """
    Remove spaces inside OCR-ish timestamps and normalize decimal separator to comma.
    Example: "0 0:00:12,260" -> "00:00:12,260"
    """
    ts = (ts or "").replace(" ", "").strip()
    ts = ts.replace(".", ",")
    return ts

def tokenize(s: str) -> List[str]:
    s = normalize_text(s)
    toks = []
    for t in s.split():
        if not t or t in STOPWORDS or len(t) <= 2:
            continue
        t = TOKEN_ALIASES.get(t, t)
        toks.append(t)
    return toks

def fuzzy_ratio(a: str, b: str) -> float:
    """Return similarity ratio in [0,1]. Uses RapidFuzz if available."""
    a = a or ""
    b = b or ""
    if rapid_ratio is not None:
        return rapid_ratio(a, b) / 100.0
    return difflib.SequenceMatcher(None, a, b).ratio()

def percentile_threshold(values: List[float], percentile: int) -> float:
    """
    Compute percentile threshold from a list of floats.
    Percentile must be in [1, 99].
    """
    if not values:
        raise RuntimeError("No scores available to compute percentile threshold.")
    if percentile <= 0 or percentile >= 100:
        raise ValueError("Percentile must be between 1 and 99.")
    values = sorted(values)
    pos = (len(values) - 1) * (percentile / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


# =============================================================================
# VIVA GUIDE — CODEBOOK INPUT
# Codes are supplied externally; the algorithm maps to an existing framework.
# MODULE 4: Codebook loader (CSV only, stable and explicit)
# =============================================================================

def load_codebook_from_csv(csv_path: str) -> List[CodeDef]:
    if not os.path.exists(csv_path):
        raise RuntimeError(f"Codebook CSV not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError("Codebook CSV has no header row.")

        fields = {normalize_text(x): x for x in reader.fieldnames}

        name_field = (
            fields.get("name") or fields.get("code name") or fields.get("code") or fields.get("nombre")
        )
        desc_field = (
            fields.get("description") or fields.get("definition") or fields.get("descripcion") or fields.get("desc")
        )

        if not name_field or not desc_field:
            raise RuntimeError(
                f"Codebook CSV must contain Name + Description columns. Found: {reader.fieldnames}"
            )

        codes = []
        for row in reader:
            name = (row.get(name_field) or "").strip()
            desc = (row.get(desc_field) or "").strip()
            if name:
                codes.append(CodeDef(name=name, description=desc))

    if not codes:
        raise RuntimeError("No codes loaded from CSV. Check that Name column has values.")
    return codes


# =============================================================================
# VIVA GUIDE — UNIT OF ANALYSIS
# Each timestamp-delimited block is one interview excerpt.
# MODULE 5: PDF extraction and excerpt splitting
# =============================================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)

def split_into_excerpts(text: str, file_name: str) -> List[Excerpt]:
    """
    Split transcript text into excerpt units based on timestamp lines.
    Each excerpt contains all lines after a timestamp until the next timestamp.
    """
    lines = text.splitlines()
    excerpts: List[Excerpt] = []

    current_start = None
    current_end = None
    buffer: List[str] = []

    def flush():
        nonlocal buffer, current_start, current_end
        if current_start and current_end and buffer:
            body = "\n".join(buffer).strip()
            body = re.sub(r"\bMachine Translated by Google\b", "", body, flags=re.IGNORECASE).strip()
            if body:
                identifier = f"{file_name}::{current_start}"
                excerpts.append(Excerpt(
                    file_name=file_name,
                    identifier=identifier,
                    start_ts=current_start,
                    end_ts=current_end,
                    text=body
                ))
        buffer = []

    for line in lines:
        m = TIMESTAMP_RE.search(line)
        if m:
            flush()
            current_start = normalize_timestamp(m.group("start"))
            current_end = normalize_timestamp(m.group("end"))
            continue

        if current_start is not None and line.strip():
            buffer.append(line.strip())

    flush()
    return excerpts


# =============================================================================
# VIVA GUIDE — LEXICON
# Lexical evidence comes from code names, optional seed synonyms, and explicit learned terms.
# MODULE 6: Lexicon building (seed + learned)
# =============================================================================

def build_seed_lexicon(codes: List[CodeDef], user_seed_path: Optional[str]) -> Dict[str, List[str]]:
    """
    Seed lexicon per code:
    - Tokens from code name
    - Full normalized code name phrase
    - Optional user JSON seed file:
        { "Code Name": ["term1", "term2", ...], ... }
    """
    lex: Dict[str, List[str]] = {}
    for c in codes:
        name_terms = tokenize(c.name)
        phrase = normalize_text(c.name)
        terms = list(set(name_terms + ([phrase] if phrase else [])))
        lex[c.name] = terms

    if user_seed_path and os.path.exists(user_seed_path):
        user_seed = safe_read_json(user_seed_path, {})
        for code_name, terms in user_seed.items():
            if code_name in lex and isinstance(terms, list):
                lex[code_name].extend([normalize_text(t) for t in terms if isinstance(t, str)])
                lex[code_name] = sorted(set([t for t in lex[code_name] if t]))

    return lex

def merge_learned_terms(lex: Dict[str, List[str]], learned_path: str) -> Tuple[Dict[str, List[str]], dict]:
    """
    learned_terms.json structure:
    {
      "Code Name": {
        "terms": {
          "term": {"local_freq": 3, "global_freq": 10, "distinctiveness": 0.23, "examples": [...]}
        },
        "meta": {...}
      }
    }
    """
    learned = safe_read_json(learned_path, {})
    for code_name, payload in learned.items():
        if code_name not in lex:
            continue
        terms_obj = (payload or {}).get("terms", {})
        for term in terms_obj.keys():
            lex[code_name].append(normalize_text(term))
        lex[code_name] = sorted(set([t for t in lex[code_name] if t]))
    return lex, learned


# =============================================================================
# VIVA GUIDE — TF-IDF / COSINE
# Code names/descriptions define the feature space; excerpts are transformed into that same space.
# MODULE 7: TF-IDF model and explainability terms
# =============================================================================

def build_tfidf(codes: List[CodeDef]) -> Tuple[TfidfVectorizer, "scipy.sparse.csr_matrix"]:
    """
    Fit TF-IDF on code texts only (name + description).
    Excerpts are transformed into this same feature space.
    """
    # VIVA: The TF-IDF reference documents are the existing code names and descriptions.
    # There is no labelled interview training set being fitted here.
    code_texts = [f"{c.name}. {c.description}".strip() for c in codes]
    # VIVA: TF-IDF uses unigram and bigram features from the code-definition corpus.
    # The model is lexical and transparent, not a semantic embedding model.
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.98)
    code_matrix = vec.fit_transform(code_texts)
    return vec, code_matrix

def tfidf_similarity(vec: TfidfVectorizer, code_matrix, excerpt_text: str) -> Tuple[List[float], "scipy.sparse.csr_matrix"]:
    """
    Returns:
      - cosine similarities to each code (list)
      - excerpt TF-IDF vector (sparse row)
    """
    # VIVA: Transform the excerpt into the vocabulary learned from the codebook definitions.
    # The model is not refitted to this excerpt.
    ex_vec = vec.transform([excerpt_text])
    # VIVA: Compare this excerpt against every code definition in TF-IDF feature space.
    sims = cosine_similarity(ex_vec, code_matrix)[0].tolist()
    return sims, ex_vec

def top_tfidf_terms_for_pair(
    vec: TfidfVectorizer,
    ex_vec,
    code_vec,
    n_terms: int
) -> List[str]:
    """
    Extract the most influential 1-2 gram terms that drove similarity
    between excerpt and a specific code.

    We compute feature-wise contribution proxy: ex_tfidf[i] * code_tfidf[i]
    (for features present in either vector).
    """
    try:
        feature_names = vec.get_feature_names_out()
    except Exception:
        feature_names = None

    # Both are sparse row vectors (1 x V)
    # Find indices with nonzero in both (intersection) for meaningful product.
    ex_idx = set(ex_vec.indices)
    code_idx = set(code_vec.indices)
    common = list(ex_idx.intersection(code_idx))
    if not common:
        return []

    # Build quick lookup maps
    ex_map = dict(zip(ex_vec.indices, ex_vec.data))
    code_map = dict(zip(code_vec.indices, code_vec.data))

    scored = []
    for i in common:
        contrib = float(ex_map.get(i, 0.0) * code_map.get(i, 0.0))
        if contrib <= 0:
            continue
        term = feature_names[i] if feature_names is not None else str(i)
        scored.append((term, contrib))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in scored[:max(0, n_terms)]]


# =============================================================================
# VIVA GUIDE — HYBRID SCORE
# Lexicon and TF-IDF evidence are combined with fixed transparent weights.
# MODULE 8: Lexicon scoring and hybrid assignment
# =============================================================================

def lexicon_score(excerpt: Excerpt, code_terms: List[str], fuzzy_threshold: float) -> Tuple[float, List[str]]:
    """
    Lexicon scoring:
    - Direct token hits
    - Phrase hits (substring in normalized excerpt)
    - Fuzzy hits for misspellings and near matches (single tokens only)
    Returns:
      (score in [0,1], matched_terms list)
    """
    ex_tokens = tokenize(excerpt.text)
    ex_set = set(ex_tokens)
    ex_norm = normalize_text(excerpt.text)

    matched = set()
    direct_hits = 0
    fuzzy_hits = 0
    phrase_hits = 0

    single_terms = []
    phrase_terms = []

    for t in code_terms:
        nt = normalize_text(t)
        if not nt:
            continue
        if " " in nt:
            phrase_terms.append(nt)
        else:
            single_terms.append(nt)

    # Direct hits
    for t in single_terms:
        if t in ex_set:
            matched.add(t)
            direct_hits += 1

    # Phrase hits (exact substring in normalized excerpt)
    for p in phrase_terms:
        if p and p in ex_norm:
            matched.add(p)
            phrase_hits += 1

    # Fuzzy hits (cap excerpt tokens to keep it bounded)
    if single_terms and ex_tokens:
        ex_tokens_cap = ex_tokens[:250]
        for term in single_terms:
            if term in ex_set:
                continue
            best = 0.0
            best_tok = None
            for tok in ex_tokens_cap:
                r = fuzzy_ratio(term, tok)
                if r > best:
                    best = r
                    best_tok = tok
                if best >= 0.99:
                    break
            if best >= fuzzy_threshold and best_tok is not None:
                matched.add(f"{term}~{best_tok}")
                fuzzy_hits += 1

    denom = max(6, len(set(single_terms)) + len(set(phrase_terms)))
    raw = (direct_hits * 1.0 + fuzzy_hits * 0.45 + phrase_hits * 0.75) / denom
    score = min(1.0, raw)
    return score, sorted(matched)

def assign_codes(
    excerpt: Excerpt,
    codes: List[CodeDef],
    lex: Dict[str, List[str]],
    vec: TfidfVectorizer,
    code_matrix,
    weight_lexicon: float,
    weight_tfidf: float,
    fuzzy_threshold: float,
    topk: int,
    top_tfidf_terms_n: int
) -> Tuple[List[Tuple[str, float]], Dict[str, dict], Dict[str, List[str]], float]:
    """
    For one excerpt, compute combined score for each code.

    Returns:
      - ranked topk list: [(code_name, combined_score), ...]
      - evidence per code: {code_name: {...}}
      - top_tfidf_terms_by_code: {code_name: [terms...]} (computed for topk only)
      - top_score: float
    """
    tfidf_sims, ex_vec = tfidf_similarity(vec, code_matrix, excerpt.text)

    evidence = {}
    scored = []

    for idx, c in enumerate(codes):
        lscore, matches = lexicon_score(excerpt, lex.get(c.name, []), fuzzy_threshold)
        tscore = float(tfidf_sims[idx])
        # VIVA: Transparent fixed hybrid score.
        # With defaults: 0.55 * lexicon + 0.45 * TF-IDF.
        # This is a ranking/decision score, not a probability.
        combined = (weight_lexicon * lscore) + (weight_tfidf * tscore)

        evidence[c.name] = {
            "lexicon_score": round(lscore, 4),
            "tfidf_score": round(tscore, 4),
            "combined_score": round(combined, 4),
            "matched_terms": matches[:25],
        }
        scored.append((c.name, combined))

    scored.sort(key=lambda x: x[1], reverse=True)
    ranked = scored[:max(1, topk)]
    top_score = ranked[0][1] if ranked else 0.0

    # Explainability terms for ranked codes only
    top_tfidf_terms_by_code: Dict[str, List[str]] = {}
    for code_name, _ in ranked:
        code_idx = next((i for i, cd in enumerate(codes) if cd.name == code_name), None)
        if code_idx is None:
            top_tfidf_terms_by_code[code_name] = []
            continue
        code_vec = code_matrix[code_idx]
        terms = top_tfidf_terms_for_pair(vec, ex_vec, code_vec, top_tfidf_terms_n)
        top_tfidf_terms_by_code[code_name] = terms

    return ranked, evidence, top_tfidf_terms_by_code, float(top_score)


# =============================================================================
# VIVA GUIDE — HUMAN REVIEW
# 'Active learning' here is human confirmation/correction, not classifier retraining.
# MODULE 9: Active learning (optional human confirmations)
# =============================================================================

def active_learning_label(
    excerpt: Excerpt,
    suggestions: List[Tuple[str, float]],
    all_code_names: List[str]
) -> List[str]:
    """
    Interactive prompt for ambiguous excerpts.
    Input options:
    - ENTER: accept suggestions
    - comma-separated codes: override
    - 'none': assign nothing
    - 'skip': keep auto suggestions
    - 'list': show all codes
    """
    print("\n" + "=" * 80)
    print(f"Excerpt: {excerpt.identifier}  ({excerpt.start_ts} --> {excerpt.end_ts})")
    print("-" * 80)
    print(excerpt.text[:900] + ("..." if len(excerpt.text) > 900 else ""))
    print("-" * 80)
    print("Suggestions:")
    for c, s in suggestions:
        print(f"  - {c}   score={s:.3f}")

    norm_map = {normalize_text(n): n for n in all_code_names}

    print("\nEnter codes (comma-separated), or press Enter to accept suggestions.")
    print("Type 'none' for no code, 'skip' to keep auto, 'list' to show all codes.")
    while True:
        raw = input("Label> ").strip()
        if raw == "":
            return [c for c, _ in suggestions]
        if raw.lower() == "skip":
            return [c for c, _ in suggestions]
        if raw.lower() == "none":
            return []
        if raw.lower() == "list":
            for n in all_code_names:
                print(" - " + n)
            continue

        typed = [r.strip() for r in raw.split(",") if r.strip()]
        resolved = []
        ok = True
        for t in typed:
            nt = normalize_text(t)
            if nt in norm_map:
                resolved.append(norm_map[nt])
            else:
                print(f"Unknown code: '{t}'. Try again or type 'list'.")
                ok = False
                break
        if ok:
            return resolved


# =============================================================================
# VIVA GUIDE — LEXICON REFINEMENT
# High-scoring assignments can propose terms; these are candidate refinements, not independent validation.
# MODULE 10: Bootstrapped lexicon learning (explicit, logged)
# =============================================================================

def propose_terms(excerpts: List[Excerpt]) -> Counter:
    cnt = Counter()
    for ex in excerpts:
        for t in tokenize(ex.text):
            if t and t not in STOPWORDS and len(t) > 2:
                cnt[t] += 1
    return cnt

def bootstrap_lexicon(
    codes: List[CodeDef],
    excerpts: List[Excerpt],
    assigned: Dict[str, List[str]],
    scores: Dict[str, float],
    learned_obj: dict,
    learned_path: str,
    high_conf: float,
    top_terms: int,
    run_log_path: str
) -> dict:
    """
    For each code, collect high-confidence excerpts and propose distinctive terms.
    Distinctiveness proxy: local_count / (global_count + 1)

    NOTE:
    - This is conservative and thesis-safe.
    - It creates candidates; you can review learned_terms.json.
    """
    code_to_ex = defaultdict(list)
    for ex in excerpts:
        ex_codes = assigned.get(ex.identifier, [])
        ex_score = scores.get(ex.identifier, 0.0)
        if ex_score < high_conf:
            continue
        for c in ex_codes:
            code_to_ex[c].append(ex)

    global_cnt = propose_terms(excerpts)

    codes_updated = []
    for c in codes:
        code = c.name
        ex_list = code_to_ex.get(code, [])
        if len(ex_list) < 3:
            continue

        local_cnt = propose_terms(ex_list)

        ranked = []
        for term, lc in local_cnt.items():
            gc = global_cnt.get(term, 0)
            if term in STOPWORDS:
                continue
            if gc > 300:
                continue
            dscore = lc / (gc + 1.0)
            ranked.append((term, dscore, lc, gc))

        ranked.sort(key=lambda x: (x[1], x[2]), reverse=True)
        ranked = ranked[:top_terms]
        if not ranked:
            continue

        if code not in learned_obj:
            learned_obj[code] = {"terms": {}, "meta": {"created": now_iso_utc()}}
        learned_obj[code].setdefault("terms", {})
        learned_obj[code].setdefault("meta", {})
        learned_obj[code]["meta"]["updated"] = now_iso_utc()

        examples = [e.identifier for e in ex_list[:3]]
        for term, dscore, lc, gc in ranked:
            if term not in learned_obj[code]["terms"]:
                learned_obj[code]["terms"][term] = {
                    "local_freq": int(lc),
                    "global_freq": int(gc),
                    "distinctiveness": round(float(dscore), 4),
                    "examples": examples
                }
            else:
                learned_obj[code]["terms"][term]["local_freq"] = max(
                    learned_obj[code]["terms"][term].get("local_freq", 0), int(lc)
                )
                learned_obj[code]["terms"][term]["global_freq"] = max(
                    learned_obj[code]["terms"][term].get("global_freq", 0), int(gc)
                )
                learned_obj[code]["terms"][term]["distinctiveness"] = max(
                    learned_obj[code]["terms"][term].get("distinctiveness", 0.0), round(float(dscore), 4)
                )

        codes_updated.append(code)

    safe_write_json(learned_path, learned_obj)

    with open(run_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": now_iso_utc(),
            "event": "bootstrap_lexicon_complete",
            "high_conf": high_conf,
            "top_terms": top_terms,
            "codes_updated": sorted(codes_updated)
        }, ensure_ascii=False) + "\n")

    return learned_obj


# =============================================================================
# MODULE 11: Output writers
# =============================================================================

def write_table(out_path: str, rows: List[dict], delimiter: str):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def write_outputs(out_prefix: str, rows: List[dict]):
    if not rows:
        print("No rows to write.")
        return

    tsv_path = f"{out_prefix}.tsv"
    csv_path = f"{out_prefix}.csv"

    write_table(tsv_path, rows, delimiter="\t")
    write_table(csv_path, rows, delimiter=",")

    print(f"Wrote: {tsv_path}")
    print(f"Wrote: {csv_path}")


# =============================================================================
# MODULE 12: Reporting helpers (top-hits + code-signals)
# =============================================================================

def make_top_hits_report(
    main_rows: List[dict],
    report_threshold: float,
    top_excerpts_per_code: int
) -> List[dict]:
    """
    Filter main rows by Top_Score >= threshold.
    Then cap to N excerpts per Top_Code (prevents one code from dominating).
    """
    if report_threshold is None:
        return []

    filtered = []
    for r in main_rows:
        try:
            s = float(r.get("Top_Score", "nan"))
        except Exception:
            s = None
        if s is None:
            continue
        if s >= report_threshold:
            filtered.append(r)

    if not filtered:
        return []

    # Cap per Top_Code
    by_code = defaultdict(list)
    for r in filtered:
        by_code[r.get("Top_Code", "")].append(r)

    capped = []
    for code, rows in by_code.items():
        rows_sorted = sorted(rows, key=lambda x: float(x.get("Top_Score", 0.0)), reverse=True)
        capped.extend(rows_sorted[:top_excerpts_per_code])

    # Sort overall
    capped.sort(key=lambda x: float(x.get("Top_Score", 0.0)), reverse=True)
    return capped

def make_code_signals_report(
    top_hits_rows: List[dict],
    top_terms_n: int,
    examples_n: int = 5
) -> List[dict]:
    """
    Aggregate Top_TFIDF_Terms across top hits, per code.
    Output rows suggest "strong definition-aligned terms" per code.
    """
    if not top_hits_rows:
        return []

    code_term_counts = defaultdict(Counter)
    code_examples = defaultdict(list)

    for r in top_hits_rows:
        code = r.get("Top_Code", "")
        ex_id = r.get("Excerpt_ID", "")
        terms = (r.get("Top_TFIDF_Terms", "") or "").split(";")
        terms = [t.strip() for t in terms if t.strip()]
        if not code:
            continue

        if ex_id and len(code_examples[code]) < examples_n:
            code_examples[code].append(ex_id)

        for t in terms:
            code_term_counts[code][t] += 1

    rows = []
    for code, cnt in code_term_counts.items():
        top_terms = cnt.most_common(top_terms_n)
        top_terms_str = "; ".join([f"{t} ({n})" for t, n in top_terms])
        rows.append({
            "Code": code,
            "Top_TFIDF_Terms_Aggregated": top_terms_str,
            "Example_Excerpt_IDs": "; ".join(code_examples.get(code, [])),
            "N_TopHit_Excerpts": sum(cnt.values())
        })

    rows.sort(key=lambda r: r["N_TopHit_Excerpts"], reverse=True)
    return rows


# =============================================================================
# VIVA GUIDE — MAIN WORKFLOW
# Connects codebook, excerpts, scoring, thresholds, reports, refinement, and logging.
# MODULE 13: CLI and main program
# =============================================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Split transcript PDFs into timestamp excerpts and code them using a thesis-safe hybrid matcher."
    )

    ap.add_argument("--codebook_csv", required=True,
                    help="Path to codebook CSV with columns Name, Description.")

    ap.add_argument("--pdf_dir", default=".", help="Directory containing transcript PDFs.")
    ap.add_argument("--pdf_glob", default="*_Translated.pdf", help="Glob pattern for transcript PDFs.")
    ap.add_argument("--out_prefix", default="excerpts_coded", help="Output prefix for TSV/CSV.")

    ap.add_argument("--seed_lexicon", default=None,
                    help="Optional JSON seed synonyms per code (Code Name -> list of terms).")
    ap.add_argument("--learned_terms", default="learned_terms.json",
                    help="Learned terms JSON store (auditable lexicon growth).")
    ap.add_argument("--confirmed_labels", default="confirmed_labels.json",
                    help="Active learning confirmations store.")
    ap.add_argument("--run_log", default="run_log.jsonl", help="Run log JSONL path.")

    ap.add_argument("--fuzzy_threshold", type=float, default=DEFAULT_FUZZY_THRESHOLD)
    ap.add_argument("--weight_lexicon", type=float, default=DEFAULT_WEIGHT_LEXICON)
    ap.add_argument("--weight_tfidf", type=float, default=DEFAULT_WEIGHT_TFIDF)

    ap.add_argument("--topk", type=int, default=DEFAULT_TOPK_CODES)
    ap.add_argument("--min_assign", type=float, default=DEFAULT_MIN_ASSIGN_SCORE)
    ap.add_argument("--high_conf", type=float, default=DEFAULT_HIGH_CONF_SCORE)

    ap.add_argument("--active_learning", action="store_true",
                    help="Interactively label ambiguous excerpts.")
    ap.add_argument("--amb_low", type=float, default=DEFAULT_AMBIGUOUS_LOW)
    ap.add_argument("--amb_high", type=float, default=DEFAULT_AMBIGUOUS_HIGH)

    ap.add_argument("--learn_top_terms", type=int, default=DEFAULT_LEARN_TOP_TERMS)

    ap.add_argument("--top_tfidf_terms_n", type=int, default=DEFAULT_TOP_TFIDF_TERMS_N,
                    help="Number of TF-IDF explanation terms to show for the top code.")

    # Reporting thresholds (mutually exclusive)
    ap.add_argument("--report_min_score", type=float,
                    help="Absolute Top_Score threshold for top-hits report (example: 0.95).")
    ap.add_argument("--report_percentile", type=int,
                    help="Percentile threshold for top-hits report (example: 95 or 99).")

    ap.add_argument("--report_top_excerpts_per_code", type=int, default=DEFAULT_REPORT_TOP_EXCERPTS_PER_CODE,
                    help="Cap number of top-hit excerpts reported per code.")
    ap.add_argument("--code_signals_top_terms", type=int, default=DEFAULT_CODE_SIGNALS_TOP_TERMS,
                    help="Number of aggregated TF-IDF terms per code in code-signals report.")

    return ap.parse_args()

def main():
    args = parse_args()
    ensure_deps_or_die()

    # Log run start
    with open(args.run_log, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": now_iso_utc(),
            "event": "run_start",
            "args": vars(args)
        }, ensure_ascii=False) + "\n")

    # Load codes
    codes = load_codebook_from_csv(args.codebook_csv)
    code_names = [c.name for c in codes]
    print(f"Loaded {len(codes)} codes.")

    # Lexicon
    lex = build_seed_lexicon(codes, args.seed_lexicon)
    lex, learned_obj = merge_learned_terms(lex, args.learned_terms)
    print(f"Lexicon ready. Learned store: {args.learned_terms}")

    # TF-IDF model
    vec, code_matrix = build_tfidf(codes)

    # Collect PDFs
    pdf_paths = sorted(glob.glob(os.path.join(args.pdf_dir, args.pdf_glob)))
    if not pdf_paths:
        raise RuntimeError(f"No PDFs found: {os.path.join(args.pdf_dir, args.pdf_glob)}")

    all_excerpts: List[Excerpt] = []
    for p in pdf_paths:
        fn = os.path.basename(p)
        print(f"Reading PDF: {fn}")
        txt = extract_text_from_pdf(p)
        exs = split_into_excerpts(txt, fn)
        print(f"  Excerpts: {len(exs)}")
        all_excerpts.extend(exs)

    print(f"Total excerpts: {len(all_excerpts)}")

    # Confirmed labels (only used if active learning is enabled)
    confirmed = safe_read_json(args.confirmed_labels, {}) if args.active_learning else {}

    # Main output rows
    rows: List[dict] = []
    assigned_codes: Dict[str, List[str]] = {}
    assigned_score: Dict[str, float] = {}

    for ex in all_excerpts:
        # Use confirmed labels if present
        if args.active_learning and ex.identifier in confirmed:
            codes_final = confirmed.get(ex.identifier, [])
            top_score = 1.0 if codes_final else 0.0
            assigned_codes[ex.identifier] = codes_final
            assigned_score[ex.identifier] = top_score

            rows.append({
                "File": ex.file_name,
                "Excerpt_ID": ex.identifier,
                "Start": ex.start_ts,
                "End": ex.end_ts,
                "Assigned_Codes": "; ".join(codes_final),
                "Top_Score": f"{top_score:.3f}",
                "Top_Code": codes_final[0] if codes_final else "",
                "Top_MatchedTerms": "",
                "Top_TFIDF_Terms": "",
                "Excerpt_Text": ex.text
            })
            continue

        ranked, evidence, tfidf_terms_by_code, top_score = assign_codes(
            excerpt=ex,
            codes=codes,
            lex=lex,
            vec=vec,
            code_matrix=code_matrix,
            weight_lexicon=args.weight_lexicon,
            weight_tfidf=args.weight_tfidf,
            fuzzy_threshold=args.fuzzy_threshold,
            topk=args.topk,
            top_tfidf_terms_n=args.top_tfidf_terms_n
        )

        # Auto assignment decision
        if top_score < args.min_assign:
            codes_auto = []
        else:
            codes_auto = [c for c, s in ranked if s >= args.min_assign]

        # Active learning if ambiguous
        if args.active_learning and (args.amb_low <= top_score < args.amb_high):
            codes_final = active_learning_label(ex, ranked, code_names)
            confirmed[ex.identifier] = codes_final
            safe_write_json(args.confirmed_labels, confirmed)
        else:
            codes_final = codes_auto

        assigned_codes[ex.identifier] = codes_final
        assigned_score[ex.identifier] = float(top_score)

        # Evidence for top code
        top_code = ranked[0][0] if ranked else ""
        top_ev = evidence.get(top_code, {}) if top_code else {}
        top_matches = top_ev.get("matched_terms", [])
        top_tfidf_terms = tfidf_terms_by_code.get(top_code, [])

        rows.append({
            "File": ex.file_name,
            "Excerpt_ID": ex.identifier,
            "Start": ex.start_ts,
            "End": ex.end_ts,
            "Assigned_Codes": "; ".join(codes_final),
            "Top_Score": f"{top_score:.3f}",
            "Top_Code": top_code,
            "Top_MatchedTerms": "; ".join(top_matches),
            "Top_TFIDF_Terms": "; ".join(top_tfidf_terms),
            "Excerpt_Text": ex.text
        })

    # Write main outputs
    write_outputs(args.out_prefix, rows)

    # -------------------------------------------------------------------------
    # Determine reporting threshold (absolute or percentile-based)
    # -------------------------------------------------------------------------
    if args.report_min_score is not None and args.report_percentile is not None:
        raise RuntimeError("Use only one of --report_min_score or --report_percentile.")

    report_threshold = None
    report_method = None

    if args.report_percentile is not None:
        report_threshold = percentile_threshold(list(assigned_score.values()), args.report_percentile)
        report_method = "percentile"
        print(f"Top-hits reporting threshold: {report_threshold:.4f} (percentile {args.report_percentile})")

    elif args.report_min_score is not None:
        report_threshold = float(args.report_min_score)
        report_method = "absolute"
        print(f"Top-hits reporting threshold: {report_threshold:.4f} (absolute)")

    # Log threshold
    with open(args.run_log, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": now_iso_utc(),
            "event": "report_threshold_set",
            "method": report_method,
            "value": report_threshold,
            "percentile": args.report_percentile
        }, ensure_ascii=False) + "\n")

    # -------------------------------------------------------------------------
    # Write report files if threshold was requested
    # -------------------------------------------------------------------------
    if report_threshold is not None:
        top_hits_rows = make_top_hits_report(
            main_rows=rows,
            report_threshold=report_threshold,
            top_excerpts_per_code=args.report_top_excerpts_per_code
        )

        if not top_hits_rows:
            print("No excerpts met the top-hits reporting threshold.")
        else:
            hits_tsv = f"{args.out_prefix}__top_hits.tsv"
            hits_csv = f"{args.out_prefix}__top_hits.csv"
            write_table(hits_tsv, top_hits_rows, delimiter="\t")
            write_table(hits_csv, top_hits_rows, delimiter=",")
            print(f"Wrote: {hits_tsv}")
            print(f"Wrote: {hits_csv}")

            # Code-signals aggregation (codebook refinement signal)
            code_signals_rows = make_code_signals_report(
                top_hits_rows=top_hits_rows,
                top_terms_n=args.code_signals_top_terms,
                examples_n=5
            )
            if not code_signals_rows:
                print("No code-signals rows produced.")
            else:
                sig_tsv = f"{args.out_prefix}__code_signals.tsv"
                sig_csv = f"{args.out_prefix}__code_signals.csv"
                write_table(sig_tsv, code_signals_rows, delimiter="\t")
                write_table(sig_csv, code_signals_rows, delimiter=",")
                print(f"Wrote: {sig_tsv}")
                print(f"Wrote: {sig_csv}")

    # -------------------------------------------------------------------------
    # Bootstrap lexicon terms from high-confidence excerpts (thesis-safe)
    # -------------------------------------------------------------------------
    learned_obj = bootstrap_lexicon(
        codes=codes,
        excerpts=all_excerpts,
        assigned=assigned_codes,
        scores=assigned_score,
        learned_obj=learned_obj,
        learned_path=args.learned_terms,
        high_conf=args.high_conf,
        top_terms=args.learn_top_terms,
        run_log_path=args.run_log
    )
    print(f"Updated learned lexicon: {args.learned_terms}")

    # Log run complete
    with open(args.run_log, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": now_iso_utc(),
            "event": "run_complete",
            "n_excerpts": len(all_excerpts),
            "out_prefix": args.out_prefix,
            "active_learning": bool(args.active_learning)
        }, ensure_ascii=False) + "\n")

    print("Done.")

if __name__ == "__main__":
    main()