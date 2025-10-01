import os
import re
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_code_and_images(text, ext):
    """
    Extract code blocks and image references from text.
    Only .txt files support code block extraction; .docx currently does not extract images.
    """
    code_blocks = []
    images = []
    if ext == ".txt":
        code_blocks = re.findall(r'```[\w\+\-]*\n(.*?)```', text, re.DOTALL)
    elif ext == ".docx":
        pass
    return code_blocks, images

def extract(file_path, chunk_size=2000, chunk_overlap=20):
    """
    Extracts and splits text from a file (PDF, TXT, DOCX) into manageable chunks for summarization.
    Also extracts code blocks and image/graph references for inclusion in summary metadata.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        pages = loader.load()
    elif ext == ".txt":
        loader = TextLoader(file_path, autodetect_encoding=True)
        pages = loader.load()
    elif ext == ".docx":
        loader = UnstructuredWordDocumentLoader(file_path)
        pages = loader.load()
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported types: .pdf, .txt, .docx")

    # Split loaded pages into chunks for summarization
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(pages)
    return chunks


def save_text_to_file(text: str, input_file: str):
    base_name = os.path.basename(input_file)
    filename_without_extension = os.path.splitext(base_name)[0]    
    print(filename_without_extension)
    with open(filename_without_extension + ".txt", "w", encoding="utf-8") as f:
        f.write(text)
