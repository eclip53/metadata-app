import os
import io
import json
import re
from datetime import datetime
from pdf2image import convert_from_bytes
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from PIL import Image
from google.cloud import vision
from keybert import KeyBERT
import langdetect
import wordninja
from pdf2image import convert_from_bytes
from sentence_transformers import SentenceTransformer

# Setup Cloud Vision Client

import tempfile
with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
    tmp.write(st.secrets["GOOGLE_APPLICATION_CREDENTIALS"].encode())
    tmp.flush()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name


vision_client = vision.ImageAnnotatorClient()
kw_model = KeyBERT(SentenceTransformer("all-MiniLM-L6-v2", device="cpu"))
# Clean OCR 

def normalize_headings(line):
    if line.isupper() and len(line.split()) < 6:
        return line.title()
    return line

def clean_ocr_text(text):
    cleaned_lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        tokens = []
        for word in line.split():
            if (
                len(word) > 14 and
                not re.match(r"^[\w\.-]+@[\w\.-]+$", word) and
                not re.match(r"^(https?://|www\.)", word)
            ):
                split = wordninja.split(word)
                tokens.extend(split)
            else:
                tokens.append(word)
        line = " ".join(tokens)

        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

# OCR Extraction


def extract_text_from_image(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    image_bytes = buffered.getvalue()
    image = vision.Image(content=image_bytes)
    response = vision_client.document_text_detection(image=image)

    if response.error.message:
        raise Exception(f"Vision API Error: {response.error.message}")

    return response.full_text_annotation.text

def extract_text_from_pdf(uploaded_file):
    try:
        # Try extracting text using PyPDF2
        reader = PdfReader(uploaded_file)
        raw_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text + "\n"

        if raw_text.strip():
            return raw_text
        else:
            raise ValueError("Empty text from PDF. Falling back to OCR.")

    except Exception as e:
        try:
            # Rewind file pointer for OCR
            uploaded_file.seek(0)
            images = convert_from_bytes(uploaded_file.read())
            ocr_text = ""
            for img in images:
                ocr_text += extract_text_from_image(img) + "\n"
            return ocr_text
        except Exception as ocr_error:
            raise Exception(f"PDF read failed: {str(e)} | OCR fallback failed: {str(ocr_error)}")



def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_docx(uploaded_file)
    elif uploaded_file.type.startswith("image/"):
        image = Image.open(uploaded_file)
        return extract_text_from_image(image)
    else:
        return "Unsupported file type."


# Metadata Extraction

def generate_metadata(text, filename="unknown_file.txt"):
    # Extract keywords and keyphrases
    keyword_list = [kw for kw, _ in kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=10)]
    keyphrase_list = [kp for kp, _ in kw_model.extract_keywords(text, keyphrase_ngram_range=(2, 4), stop_words='english', top_n=10)]

    # Language detection
    try:
        language = langdetect.detect(text)
    except:
        language = "unknown"

    # Attempt to extract a smart title and summary
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    title = "Untitled Document"
    summary = "No summary available"

    for i, line in enumerate(lines):
        if len(line.split()) > 3 and not line.isupper():
            title = line
            summary = lines[i+1] if i+1 < len(lines) else summary
            break

    word_count = len(re.findall(r'\w+', text))

    return {
        "filename": filename,
        "title": title,
        "summary": summary,
        "keywords": keyword_list,
        "keyphrases": keyphrase_list,
        "language": language,
        "word_count": word_count,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


# Streamlit UI

st.set_page_config(page_title="Automated Meta Data Generator", layout="wide")
st.title("📄 Automated Metadata Generator (Vision OCR)")
uploaded_file = st.file_uploader("Upload a PDF, DOCX, or Image", type=["pdf", "docx", "png", "jpg", "jpeg"])

if uploaded_file:
    with st.spinner("🔍 Extracting content..."):
        try:
            raw_text = extract_text(uploaded_file)
            cleaned_text = clean_ocr_text(raw_text)
            metadata = generate_metadata(cleaned_text, filename=uploaded_file.name)

            st.subheader("📃 Extracted Text")
            st.text_area("Full Text", cleaned_text, height=300)

            st.subheader("🧠 Metadata (JSON Format)")
            metadata_json = json.dumps(metadata, indent=4)
            st.code(metadata_json, language='json')

            st.download_button(
                label="📥 Download Metadata as JSON",
                data=metadata_json,
                file_name="metadata.json",
                mime="application/json"
            )
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


