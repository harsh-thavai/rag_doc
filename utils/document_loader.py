import PyPDF2
from io import BytesIO

def load_document(uploaded_file) -> str:
    """Load and extract text from uploaded file"""
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    if file_type == 'pdf':
        return _extract_pdf_text(uploaded_file)
    elif file_type == 'txt':
        return uploaded_file.read().decode('utf-8')
    else:
        raise ValueError("Unsupported file type")

def _extract_pdf_text(uploaded_file) -> str:
    """Extract text from PDF using PyPDF2"""
    pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text