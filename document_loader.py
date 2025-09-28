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


def extract(file_path, chunk_size=2000, chunk_overlap=20, save_loaded=False, save_path=None):
    """
    Extracts and splits text from a file (PDF, TXT, DOCX) into manageable chunks for summarization.
    Also extracts code blocks and image/graph references for inclusion in summary metadata.
    Optionally saves the loaded document text to a file for reuse.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        text_content = " ".join([p.page_content for p in pages])
    elif ext == ".txt":
        loader = TextLoader(file_path, autodetect_encoding=True)
        pages = loader.load()
        text_content = " ".join([p.page_content for p in pages])
    elif ext == ".docx":
        loader = UnstructuredWordDocumentLoader(file_path)
        pages = loader.load()
        text_content = " ".join([p.page_content for p in pages])
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported types: .pdf, .txt, .docx")

    if save_loaded:
        # Save the loaded document text to a file for reuse
        if not save_path:
            base_name = os.path.basename(file_path)
            file_name_without_ext = os.path.splitext(base_name)[0]
            save_path = f"{file_name_without_ext}_loaded.txt"
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text_content)

    code_blocks, images = extract_code_and_images(text_content, ext)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(pages)
    for doc in texts:
        if code_blocks:
            doc.metadata["code_blocks"] = code_blocks
        if images:
            doc.metadata["images"] = images
    return texts
