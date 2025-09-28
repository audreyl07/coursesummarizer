
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from document_loader import extract
from summarizer import summarize_document_with_kmeans_clustering
from pdf_manager import append_summary_to_pdf

app = FastAPI()

class SummarizeRequest(BaseModel):
    file_path: str
    chunk_size: int = 2000
    chunk_overlap: int = 20
    num_clusters: int = 20

@app.post("/summarize/")
def summarize_endpoint(req: SummarizeRequest):
    try:
        from langchain_community.llms import OpenAI
        from langchain_community.embeddings import HuggingFaceEmbeddings
        llm = OpenAI()
        embeddings = HuggingFaceEmbeddings()

        texts = extract(
            req.file_path,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
            save_loaded=True
        )
        summary = summarize_document_with_kmeans_clustering(
            texts, llm, embeddings, num_clusters=req.num_clusters
        )
        append_summary_to_pdf("summary_output.pdf", summary)
        base_name = os.path.basename(req.file_path)
        file_name_without_ext = os.path.splitext(base_name)[0]
        loaded_path = f"{file_name_without_ext}_loaded.txt"
        pdf_output = f"{file_name_without_ext} summary.pdf"
        return {
            "message": "Summary generated and saved to PDF.",
            "loaded_document": loaded_path,
            "pdf_output": pdf_output
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    if len(os.sys.argv) < 2:
        print("Usage: python main.py <document_path>")
        os.sys.exit(1)
    file_path = os.sys.argv[1]
    chunk_size = 2000
    chunk_overlap = 20
    num_clusters = 20
    from langchain_community.llms import OpenAI
    from langchain_community.embeddings import HuggingFaceEmbeddings
    llm = OpenAI()
    embeddings = HuggingFaceEmbeddings()
    print(f"Extracting and chunking document: {file_path}")
    texts = extract(file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap, save_loaded=True)
    print(f"Summarizing document with clustering...")
    summary = summarize_document_with_kmeans_clustering(texts, llm, embeddings, num_clusters=num_clusters)
    print("Summary generated. Saving to PDF...")
    append_summary_to_pdf("summary_output.pdf", summary)
    print("Done.")

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