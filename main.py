import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from document_loader import extract
from summarizer import summarize_document_with_kmeans_clustering, cross_check_summary, get_llm
from pdf_manager import append_summary_to_pdf
from langchain_community.embeddings import HuggingFaceEmbeddings

# === FastAPI App Setup ===
app = FastAPI()

# Enable CORS for Postman and other clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SummarizeRequest(BaseModel):
    """
    Request model for the /summarize/ API endpoint.
    Allows user to specify file path, chunking parameters, model, accuracy check, and API keys.
    """
    file_path: str
    chunk_size: int = 2000
    chunk_overlap: int = 20
    num_clusters: int = 20
    model: str = "gpt"  # Default model for categorization
    check_accuracy: bool = False
    api_keys: dict = None  # Optional dict for API keys

class SummarizeResponse(BaseModel):
    """
    Response model for the /summarize/ API endpoint.
    Returns summary status, file paths, and optional accuracy report.
    """
    message: str
    loaded_document: str
    pdf_output: str
    accuracy_report: dict = None

# === Helper Functions ===
def set_api_keys(api_keys: dict):
    """
    Sets API keys for LLMs as environment variables.
    Args:
        api_keys: Dictionary of environment variable names and values.
    """
    if not api_keys:
        return
    for key, value in api_keys.items():
        os.environ[key] = value

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