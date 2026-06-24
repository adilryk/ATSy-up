import pdfplumber
import pytesseract
from PIL import Image
import io

def extract_from_pdf(file_bytes: io.BytesIO) -> str:
    """
    Extracts text from a PDF file using pdfplumber.
    """
    text = ""
    try:
        with pdfplumber.open(file_bytes) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    return text.strip()

def extract_from_image(file_bytes: io.BytesIO) -> str:
    """
    Extracts text from an image file using pytesseract.
    Note: Requires Tesseract OCR installed and in PATH.
    """
    try:
        image = Image.open(file_bytes)
        text = pytesseract.image_to_string(image)
    except Exception as e:
        raise Exception(f"Failed to extract text from image: {str(e)}")
        
    return text.strip()
