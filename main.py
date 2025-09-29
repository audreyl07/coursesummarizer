import os
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from document_loader import extract, extract_content_by_type
from pdf_manager import append_summary_to_pdf, get_title
from summarizer import summarize_document_with_kmeans_clustering, cross_check_summary, get_llm
from model_collaborator import ModelCollaborator
from document_categorizer import categorize_chunks, categorize_chunk_model
from langchain_community.embeddings import HuggingFaceEmbeddings

# === FastAPI App Setup ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Helper Function for API Key ===
def set_api_key(env_var: str, api_key: str):
    """
    Sets a single API key as an environment variable.
    """
    if env_var and api_key:
        os.environ[env_var] = api_key

def set_api_keys(api_keys: dict):
    """
    Sets multiple API keys from a dictionary of environment variable names and values.
    """
    if api_keys:
        for env_var, api_key in api_keys.items():
            if env_var and api_key:
                os.environ[env_var] = api_key

# === Document Loader Endpoint ===
class DocumentLoaderRequest(BaseModel):
    file_path: str
    chunk_size: int = 2000
    chunk_overlap: int = 20
    save_loaded: bool = False
    save_path: str = None
    api_key: str = None  # API key for document loader
    api_key_env: str = "DOCUMENT_LOADER_API_KEY"  # Environment variable name

@app.post("/document_loader/")
def document_loader_endpoint(req: DocumentLoaderRequest):
    set_api_key(req.api_key_env, req.api_key)
    texts = extract(
        req.file_path,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
        save_loaded=req.save_loaded,
        save_path=req.save_path
    )
    return {"chunks": [doc.page_content for doc in texts], "metadata": [doc.metadata for doc in texts]}

# === PDF Manager Endpoint ===
class PDFManagerRequest(BaseModel):
    pdf_path: str
    new_summary: str
    output_pdf: str = "summary_output.pdf"
    api_key: str = None  # API key for PDF manager
    api_key_env: str = "PDF_MANAGER_API_KEY"  # Environment variable name

@app.post("/pdf_manager/")
def pdf_manager_endpoint(req: PDFManagerRequest):
    set_api_key(req.api_key_env, req.api_key)
    append_summary_to_pdf(req.pdf_path, req.new_summary, req.output_pdf)
    return {"output_pdf": req.output_pdf}

# === Summarizer Endpoint ===
class SummarizeRequest(BaseModel):
    file_path: str
    chunk_size: int = 2000
    chunk_overlap: int = 20
    num_clusters: int = 20
    model: str = "gpt"
    check_accuracy: bool = False
    api_keys: dict = None  # Dictionary of API keys

class SummarizeResponse(BaseModel):
    message: str
    loaded_document: str = None
    pdf_output: str = None
    accuracy_report: str = None

@app.post("/summarizer/")
def summarizer_endpoint(req: SummarizeRequest):
    set_api_key("SUMMARIZER_API_KEY", req.api_keys.get("SUMMARIZER_API_KEY") if req.api_keys else None)
    llm = get_llm(req.model)
    embeddings = HuggingFaceEmbeddings()
    texts = extract(req.file_path, chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap)
    summary = summarize_document_with_kmeans_clustering(texts, embeddings, num_clusters=req.num_clusters)
    accuracy_report = None
    if req.check_accuracy:
        with open(req.file_path, "r", encoding="utf-8") as f:
            source_text = f.read()
        accuracy_report = cross_check_summary(summary, source_text)
    return {"summary": summary, "accuracy_report": accuracy_report}

# === API Endpoint ===
@app.post("/summarize/", response_model=SummarizeResponse)
def summarize_endpoint(req: SummarizeRequest):
    """
    API endpoint to summarize a document and save the summary to PDF.
    Optionally checks summary accuracy and returns a report.
    Example Postman usage:
    - Method: POST
    - URL: http://localhost:8000/summarize/
    - Body (raw, JSON):
      {
        "file_path": "COMP2401_Ch1_SystemsProgramming.pdf",
        "chunk_size": 2000,
        "chunk_overlap": 20,
        "num_clusters": 20,
        "model": "gpt",
        "check_accuracy": true,
        "api_keys": {"OPENAI_API_KEY": "sk-..."}
      }
    """
    try:
        set_api_keys(req.api_keys)
        llm = get_llm(req.model)
        embeddings = HuggingFaceEmbeddings()

        # Extract and chunk document
        texts = extract(
            req.file_path,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
            save_loaded=True
        )
        # Summarize using multi-model pipeline
        summary = summarize_document_with_kmeans_clustering(
            texts, embeddings, num_clusters=req.num_clusters
        )
        # Optionally cross-check summary accuracy
        accuracy_report = None
        if req.check_accuracy:
            with open(req.file_path, "r", encoding="utf-8") as f:
                source_text = f.read()
            accuracy_report = cross_check_summary(summary, source_text)
        # Save summary to PDF
        append_summary_to_pdf("summary_output.pdf", summary)
        base_name = os.path.basename(req.file_path)
        file_name_without_ext = os.path.splitext(base_name)[0]
        loaded_path = f"{file_name_without_ext}_loaded.txt"
        pdf_output = f"{file_name_without_ext} summary.pdf"
        return SummarizeResponse(
            message="Summary generated and saved to PDF.",
            loaded_document=loaded_path,
            pdf_output=pdf_output,
            accuracy_report=accuracy_report
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === CLI Main Function ===
def main():
    """
    CLI entry point for document summarization.
    Usage: python main.py <document_path> [model] [check_accuracy] [api_key_env_var=api_key_value ...]
    """
    if len(os.sys.argv) < 2:
        print("Usage: python main.py <document_path> [model] [check_accuracy] [api_key_env_var=api_key_value ...]")
        os.sys.exit(1)
    file_path = os.sys.argv[1]
    chunk_size = 2000
    chunk_overlap = 20
    num_clusters = 20
    model = os.sys.argv[2] if len(os.sys.argv) > 2 else "gpt"
    check_accuracy = bool(int(os.sys.argv[3])) if len(os.sys.argv) > 3 else False
    api_keys = {}
    for arg in os.sys.argv[4:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            api_keys[k] = v
    set_api_keys(api_keys)
    llm = get_llm(model)

    embeddings = HuggingFaceEmbeddings()
    print(f"Extracting and chunking document: {file_path}")
    texts = extract(file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap, save_loaded=True)
    print(f"Summarizing document with clustering...")
    summary = summarize_document_with_kmeans_clustering(texts, embeddings, num_clusters=num_clusters)
    print("Summary generated. Saving to PDF...")
    append_summary_to_pdf("summary_output.pdf", summary)
    print("Done.")
    if check_accuracy:
        with open(file_path, "r", encoding="utf-8") as f:
            source_text = f.read()
        report = cross_check_summary(summary, source_text)
        print("Accuracy Report:", report)

# === Entry Point ===
if __name__ == "__main__":
    if len(os.sys.argv) > 1:
        main()
    else:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True
        )