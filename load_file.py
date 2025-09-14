"""
load_file.py
Convert a file into text format (rax text, file handling, audio transcription...)
"""
import os
import chardet
from docx import Document
from odf.opendocument import load as load_odt
from odf.text import P
from striprtf.striprtf import rtf_to_text
from PyPDF2 import PdfReader

def load_file(file_path: str) -> str:
    """
    Load and return the contents of a text-based file in many formats.
    
    Args:
        file_path (str): The absolute path to the file.

    Returns:
        str: The text content of the file or an error message.
    """
    if not os.path.isabs(file_path):
        return("Please provide an absolute file path.")

    if not os.path.isfile(file_path):
        return(f"No such file: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".docx":
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])

    elif ext == ".odt":
        doc = load_odt(file_path)
        paragraphs = [t.data for t in doc.getElementsByType(P)]
        return "\n".join(paragraphs)

    elif ext == ".rtf":
        with open(file_path, "r", errors="ignore") as f:
            return rtf_to_text(f.read())

    elif ext == ".pdf":
        reader = PdfReader(file_path)
        text = []
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text.append(extracted)
        return "\n".join(text)

    else:
        try:
            with open(file_path, "rb") as f:
                raw = f.read()
            detected = chardet.detect(raw)
            encoding = detected["encoding"] or "utf-8"
            return raw.decode(encoding, errors="replace")
        except Exception as e :
            return(f"Error while uploading script: {e}")