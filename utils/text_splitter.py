from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_document(text: str, chunk_size: int = 1200, overlap: int = 200) -> list:
    """
    Split document into chunks with improved parameters for academic papers
    """
    # Custom separators that work better with academic papers
    separators = [
        "\n\n\n",  # Triple line breaks (section breaks)
        "\n\n",    # Double line breaks (paragraph breaks)
        "\n",      # Single line breaks
        ". ",      # Sentence breaks
        "! ",      # Exclamation breaks
        "? ",      # Question breaks
        "; ",      # Semicolon breaks
        ", ",      # Comma breaks
        " ",       # Word breaks
        ""         # Character breaks
    ]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=separators
    )
    return splitter.split_text(text)