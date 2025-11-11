
import streamlit as st
import fitz  # PyMuPDF
import docx2txt
import torch
import os
import json
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# =========================
# Configurations & Skill List
# =========================

SKILLS = [
    "Python", "Java", "C++", "SQL", "Machine Learning", "Deep Learning", "AWS", "Azure",
    "Docker", "Kubernetes", "Linux", "Excel", "PowerPoint", "Communication", "Leadership",
    "Project Management", "Data Analysis", "Statistics", "NLP", "TensorFlow", "PyTorch",
    "React", "Angular", "Time Management", "Fast Learning", "Problem Solving", "Teamwork",
    "Creativity", "Critical Thinking", "API Design", "HTML", "CSS"
]

MODEL_EMBED = 'sentence-transformers/all-MiniLM-L6-v2'
MODEL_REWRITE = 'google/flan-t5-small'

# =========================
# Utils
# =========================

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_docx(file):
    return docx2txt.process(file)

def skill_match(text, skills):
    found = [s for s in skills if s.lower() in text.lower()]
    return found

def extract_resume_bullets(text):
    lines = [l.strip() for l in text.split('') if l.strip()]
    bullets = [l for l in lines if l.startswith('-') or l.startswith('•') or l.startswith('*')]
    # If none detected, fallback: lines with length > 30
    if not bullets and lines:
        bullets = [l for l in lines if len(l) > 30]
    return bullets
