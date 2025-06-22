# 📄 Streamlit Metadata Extraction App (YAKE Version)

import os
import io
import json
import re
from datetime import datetime
from PIL import Image
from PyPDF2 import PdfReader
from docx import Document
from pdf2image import convert_from_bytes
from google.cloud import vision
import langdetect
import wordninja
import yake
import streamlit as st

# Load credentials from Streamlit secrets
if "GOOGLE_APPLICATION_CREDENTIALS" in st.secrets:
    service_account_info = json.loads(st.secrets["GOOGLE_APPLICATION_CREDENTIALS"])
    temp_path = "/tmp/gcp_key.json"
    with open(temp_path, "w") as f:
        json.dump(service_account_info, f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_path
else:
    st.error("❌ GOOGLE_APPLICATION_CREDENTIALS not found in secrets.toml")

# ✅ Initialize Vision API client
vision_client = vision.ImageAnnotatorClient()


def clean_ocr_text(text):
    cleaned_lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Fix all-uppercase spaced letters (e.g. P L A C E M E N T)
        if re.fullmatch(r"(?:[A-Z]\s*){3,}", line):
            cleaned_lines.append(line.replace(" ", ""))
            continue

        # Fix CamelCase joins → add space between lowercase and uppercase (e.g. placementCell → placement Cell)
        line = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', line)

        # Add space between digits and letters
        line = re.sub(r'(?<=\d)(?=[A-Za-z])', ' ', line)
        line = re.sub(r'(?<=[A-Za-z])(?=\d)', ' ', line)

        # Fix overly long jammed words using wordninja (e.g. recommendedto → recommended to)
        tokens = []
        for word in line.split():
            if len(word) > 14 and not re.match(r"^[\w\.-]+@[\w\.-]+$", word):
                tokens.extend(wordninja.split(word))
            else:
                tokens.append(word)
        line = " ".join(tokens)

        # Normalize multiple spaces
        line = re.sub(r'\s{2,}', ' ', line)

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)



def extract_text_from_image(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    image_bytes = buffered.getvalue()
    image = vision.Image(content=image_bytes)
    response = vision_client.document_text_detection(image=image)
    if response.error.message:
        raise Exception(f"Vision API Error: {response.error.message}")
    return response.full_text_annotation.text

def extract_text_from_pdf(file):
    try:
        reader = PdfReader(file)
        text = "\n".join([p.extract_text() or "" for p in reader.pages])
        if text.strip():
            return text
        raise ValueError("No text found")
    except:
        file.seek(0)
        images = convert_from_bytes(file.read())
        return "\n".join([extract_text_from_image(img) for img in images])

def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join(para.text for para in doc.paragraphs)

def extract_text(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_docx(uploaded_file)
    elif uploaded_file.type.startswith("image/"):
        return extract_text_from_image(Image.open(uploaded_file))
    else:
        raise ValueError("Unsupported file type")

def generate_metadata(text, filename="unknown.txt"):
    kw_extractor = yake.KeywordExtractor(top=20, stopwords=None)
    keyword_results = kw_extractor.extract_keywords(text)

    keyword_list = [kw for kw, _ in keyword_results if len(kw.split()) == 1][:10]
    keyphrase_list = [kw for kw, _ in keyword_results if len(kw.split()) > 1][:10]

    try:
        language = langdetect.detect(text)
    except:
        language = "unknown"

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    title = "Untitled Document"
    summary = "No summary available"
    for i, line in enumerate(lines):
        if len(line.split()) > 3 and not line.isupper():
            title = line
            summary = lines[i+1] if i+1 < len(lines) else summary
            break

    word_count = len(re.findall(r"\w+", text))

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

## 🚀 6. Streamlit UI

st.set_page_config(page_title="Automated Meta Data Generator", layout="wide")
st.title("📄 Automated Metadata Generator (Vision OCR + YAKE)")

uploaded_file = st.file_uploader("Upload a PDF, DOCX, or Image", type=["pdf", "docx", "png", "jpg", "jpeg"])

if uploaded_file:
    with st.spinner("🔍 Extracting content..."):
        try:
            raw_text = extract_text(uploaded_file)
            cleaned_text = clean_ocr_text(raw_text)
            metadata = generate_metadata(cleaned_text, filename=uploaded_file.name)

            tab1, tab2 = st.tabs(["📃 Extracted Text", "🧠 Metadata"])
            with tab1:
                st.text_area("Full Text", cleaned_text, height=300)
            with tab2:
                st.json(metadata)

            st.download_button(
                label="📥 Download Metadata as JSON",
                data=json.dumps(metadata, indent=4),
                file_name="metadata.json",
                mime="application/json"
            )
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


