# Smart Resume Analyzer & Career Advisor (robust, no hard deps)
# ------------------------------------------------------------
# This Streamlit app is designed to run even if some optional
# ML libraries are missing. It uses graceful fallbacks.
#
# Recommended (but optional) installs for best results:
#   pip install streamlit PyMuPDF docx2txt torch sentence-transformers transformers
# Fallbacks used when the above are not present:
#   - PDF text: PyPDF2
#   - DOCX text: python-docx
#   - Similarity: Jaccard overlap (token-based)
#   - Bullet rewrite: Simple heuristic rewriter
# ------------------------------------------------------------

from __future__ import annotations
import io
import os
import json
import re
import tempfile
from typing import List, Tuple

import streamlit as st

# ---------------------------
# Optional imports (with fallbacks)
# ---------------------------

# PDF extractors
pymupdf = None
try:
    import fitz  # PyMuPDF
    pymupdf = fitz
except Exception:
    pymupdf = None

pypdf2 = None
if pymupdf is None:
    try:
        from PyPDF2 import PdfReader
        pypdf2 = PdfReader
    except Exception:
        pypdf2 = None

# DOCX extractors
_has_docx2txt = False
try:
    import docx2txt  # type: ignore
    _has_docx2txt = True
except Exception:
    _has_docx2txt = False

python_docx = None
if not _has_docx2txt:
    try:
        import docx  # python-docx
        python_docx = docx
    except Exception:
        python_docx = None

# Embeddings (optional)
_sbert = None
try:
    from sentence_transformers import SentenceTransformer
    _sbert = SentenceTransformer
except Exception:
    _sbert = None

# Torch (optional)
_torch = None
try:
    import torch
    _torch = torch
except Exception:
    _torch = None

# Transformers for rewriting (optional)
_hf_pipe = None
try:
    from transformers import pipeline
    _hf_pipe = pipeline
except Exception:
    _hf_pipe = None

# ---------------------------
# Configurations & Skill List
# ---------------------------

SKILLS: List[str] = [
    "Python", "Java", "C++", "SQL", "Machine Learning", "Deep Learning", "AWS", "Azure",
    "Docker", "Kubernetes", "Linux", "Excel", "PowerPoint", "Communication", "Leadership",
    "Project Management", "Data Analysis", "Statistics", "NLP", "TensorFlow", "PyTorch",
    "React", "Angular", "Time Management", "Fast Learning", "Problem Solving", "Teamwork",
    "Creativity", "Critical Thinking", "API Design", "HTML", "CSS"
]

MODEL_EMBED = 'sentence-transformers/all-MiniLM-L6-v2'
MODEL_REWRITE = 'google/flan-t5-small'

# ---------------------------
# Helpers
# ---------------------------

@st.cache_data(show_spinner=False)
def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _safe_read(file) -> bytes:
    # Streamlit uploaded file supports .read(); ensure pointer reset
    try:
        file.seek(0)
    except Exception:
        pass
    data = file.read()
    if isinstance(data, str):
        data = data.encode('utf-8', errors='ignore')
    return data


def extract_text_from_pdf(uploaded_file) -> str:
    data = _safe_read(uploaded_file)
    # Try PyMuPDF first
    if pymupdf is not None:
        try:
            doc = pymupdf.open(stream=data, filetype="pdf")
            text = []
            for page in doc:
                text.append(page.get_text())
            return _normalize_ws("\n".join(text))
        except Exception as e:
            st.info(f"PyMuPDF failed, falling back to PyPDF2. Details: {e}")
    # Fallback: PyPDF2
    if pypdf2 is not None:
        try:
            reader = pypdf2(io.BytesIO(data))
            out = []
            for page in reader.pages:
                out.append(page.extract_text() or "")
            return _normalize_ws("\n".join(out))
        except Exception as e:
            st.error(f"Could not read PDF with PyPDF2: {e}")
    st.error("No PDF reader available. Install either PyMuPDF (fitz) or PyPDF2.")
    return ""


def extract_text_from_docx(uploaded_file) -> str:
    data = _safe_read(uploaded_file)
    # Try docx2txt first (requires a file path)
    if _has_docx2txt:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            try:
                text = docx2txt.process(tmp_path)
                return _normalize_ws(text or "")
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
        except Exception as e:
            st.info(f"docx2txt failed, trying python-docx. Details: {e}")
    # Fallback: python-docx
    if python_docx is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            try:
                document = python_docx.Document(tmp_path)
                text = "\n".join(p.text for p in document.paragraphs)
                return _normalize_ws(text)
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
        except Exception as e:
            st.error(f"Could not read DOCX with python-docx: {e}")
    st.error("No DOCX reader available. Install docx2txt or python-docx.")
    return ""


def skill_match(text: str, skills: List[str]) -> List[str]:
    text_low = text.lower()
    found = [s for s in skills if s.lower() in text_low]
    return sorted(set(found), key=lambda x: x.lower())


def extract_resume_bullets(text: str) -> List[str]:
    # Split on newlines and common bullet markers
    raw_lines = [l.strip() for l in re.split(r"\r?\n", text or "") if l.strip()]
    bullets = [l for l in raw_lines if l.startswith(('-', '•', '*', '–', '—'))]
    if not bullets and raw_lines:
        # Fallback: treat long sentences as bullets
        bullets = [l for l in raw_lines if len(l) > 40]
    # Deduplicate while preserving order
    seen = set()
    out = []
    for b in bullets:
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out[:100]


# --- Similarity ---
@st.cache_resource(show_spinner=False)
def _get_embedder():
    if _sbert is None or _torch is None:
        return None
    try:
        return _sbert(MODEL_EMBED)
    except Exception:
        return None


def similarity_score(a: str, b: str) -> Tuple[float, str]:
    a = a or ""
    b = b or ""
    embedder = _get_embedder()
    if embedder is not None and _torch is not None:
        try:
            a_emb = embedder.encode([a], convert_to_tensor=True)
            b_emb = embedder.encode([b], convert_to_tensor=True)
            sim = float(_torch.nn.functional.cosine_similarity(a_emb, b_emb).item())
            pct = round(max(0.0, min(1.0, sim)) * 100, 2)
            label = 'Strong' if pct >= 80 else ('Medium' if pct >= 60 else 'Weak')
            return pct, label
        except Exception as e:
            st.info(f"Embedding similarity unavailable, using token overlap. Details: {e}")
    # Fallback: Jaccard token overlap
    def tokens(x: str) -> set:
        return set(re.findall(r"[a-zA-Z0-9#+\.\-]+", x.lower()))
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0, 'Weak'
    inter = len(ta & tb)
    union = len(ta | tb)
    pct = round((inter / union) * 100, 2)
    label = 'Strong' if pct >= 80 else ('Medium' if pct >= 60 else 'Weak')
    return pct, label


# --- Bullet Rewriter ---
@st.cache_resource(show_spinner=False)
def _get_rewriter():
    if _hf_pipe is None:
        return None
    try:
        return _hf_pipe("text2text-generation", model=MODEL_REWRITE, tokenizer=MODEL_REWRITE)
    except Exception:
        return None


def rewrite_bullet(text: str) -> str:
    pipe = _get_rewriter()
    prompt = (
        "Rewrite this resume bullet to be concise, quantified, and ATS-friendly. "
        "Keep it one line, start with a strong verb, and add a concrete metric if possible: "
        f"{text}"
    )
    if pipe is not None:
        try:
            out = pipe(prompt, max_new_tokens=64, do_sample=False)
            gen = out[0].get('generated_text', '').strip()
            return gen or _heuristic_rewrite(text)
        except Exception:
            return _heuristic_rewrite(text)
    return _heuristic_rewrite(text)


def _heuristic_rewrite(bullet: str) -> str:
    # Basic cleanup + add a quantified suffix if missing
    b = _normalize_ws(bullet)
    b = re.sub(r"^(\-|\*|•|–|—)\s*", "", b)
    # Ensure starts with a verb-like capital
    b = b[:1].upper() + b[1:]
    # Add a simple quantifier if none present
    if not re.search(r"\b(\d+%?|million|k|thousand)\b", b, re.I):
        b = f"{b} — improved results by 15% through optimization and collaboration"
    return f"• {b}"


# ---------------------------
# UI
# ---------------------------

st.set_page_config(page_title="Smart Resume Analyzer & Career Advisor", layout="centered")

st.title("Smart Resume Analyzer & Career Advisor")
st.markdown(
    "Upload your resume, paste a job description, and get actionable insights. "
    "This build has **zero hard ML dependencies** and will gracefully fall back when needed."
)

# Session State
if "profile_json" not in st.session_state:
    st.session_state["profile_json"] = None

# Sidebar controls
with st.sidebar:
    st.header("Profile Controls")
    if st.session_state["profile_json"]:
        st.success("Saved profile found.")
        if st.button("Delete Saved Profile"):
            st.session_state["profile_json"] = None
            st.rerun()
    if st.button("Save anonymized profile"):
        st.session_state["profile_json"] = st.session_state.get("last_profile", {})
        st.success("Profile saved! Only skills and scores stored.")

# 1) Resume Upload
st.subheader("Step 1: Upload Resume (PDF/DOCX)")
resume_file = st.file_uploader("Choose PDF or DOCX", type=["pdf", "docx"])
resume_text = ""

if resume_file is not None:
    try:
        name = resume_file.name.lower()
        if name.endswith(".pdf"):
            resume_text = extract_text_from_pdf(resume_file)
        else:
            resume_text = extract_text_from_docx(resume_file)
        if resume_text:
            st.success("Resume text extracted!")
            st.text_area("Resume Extracted Text", resume_text, height=200)
        else:
            st.warning("No text extracted. If it's a scanned PDF, try another extractor or OCR.")
    except Exception as e:
        st.error(f"Error extracting resume: {e}")

# 2) Job Description
st.subheader("Step 2: Paste Job Description")
jd_text = st.text_area("Paste Job Description here", "", height=200)

# 3) Similarity
match_score, match_label = None, None
if resume_text and jd_text:
    with st.spinner("Computing similarity..."):
        match_score, match_label = similarity_score(resume_text, jd_text)
        st.metric("Resume-to-JD Match Score", f"{match_score}%", match_label)

# 4) Skill Extraction
resume_skills: List[str] = []
jd_skills: List[str] = []
missing_skills: List[str] = []

if resume_text:
    resume_skills = skill_match(resume_text, SKILLS)
if jd_text:
    jd_skills = skill_match(jd_text, SKILLS)
if resume_skills and jd_skills:
    missing_skills = [s for s in jd_skills if s not in resume_skills]

if resume_skills:
    st.write("*Skills Detected in Resume:* ", ', '.join(resume_skills))
if jd_skills:
    st.write("*Skills Required by JD:* ", ', '.join(jd_skills))
if missing_skills:
    st.warning(f"Skill Gaps: {', '.join(missing_skills)}")
    st.write("*Prioritized Learning Suggestions:*")
    for skill in missing_skills:
        st.write(f"- {skill}: Explore Coursera, YouTube, free MOOCs, and Kaggle notebooks.")

# 5) Bullet Rewriter
st.subheader("Step 3: Resume Bullet Rewriter")
bullets = extract_resume_bullets(resume_text) if resume_text else []
selected_bullet = st.selectbox(
    "Pick a resume bullet to rewrite for ATS-optimization:",
    bullets if bullets else ["(No bullets detected)"]
)

if (bullets and selected_bullet) and st.button("Rewrite Selected Bullet"):
    with st.spinner("Rewriting bullet..."):
        improved = rewrite_bullet(selected_bullet)
        st.write("*Improved Bullet:*")
        st.success(improved)
        st.session_state["last_rewritten_pair"] = (selected_bullet, improved)

# 6) Download Revised Resume (.txt)
if resume_text:
    st.subheader("Step 4: Download as TXT")
    revised_resume = resume_text
    pair = st.session_state.get("last_rewritten_pair")
    if pair:
        old, new = pair
        # Replace only first occurrence to avoid over-replacing
        revised_resume = resume_text.replace(old, new, 1)
    st.download_button("Download Resume Text (.txt)", revised_resume, file_name="revised_resume.txt")

# 7) Save JSON Profile
if resume_text and match_score is not None:
    profile_dict = {
        "score": match_score,
        "label": match_label,
        "resume_skills": resume_skills,
        "jd_skills": jd_skills,
        "missing_skills": missing_skills,
    }
    st.session_state["last_profile"] = profile_dict

    if st.checkbox("Show anonymized profile JSON"):
        st.json(profile_dict)

st.caption(
    "Privacy-first: No file contents saved except anonymized skill/scores if profile button is used. "
    "All analysis is local/in-memory."
)
