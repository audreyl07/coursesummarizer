import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from document_loader import extract, save_text_to_file
from summarizer import summarize_document_with_kmeans_clustering
from pdf_manager import append_summary_to_pdf
from typing import Optional

app = FastAPI()

# Unified request model for all summarization endpoints
class SummarizeRequest(BaseModel):
    file_path: str
    model: Optional[str] = None
    num_clusters: int = 20
    output_file: Optional[str] = None
    openai_api_key: Optional[str] = None

# Unified response model for all summarization endpoints
class SummarizeResponse(BaseModel):
    summary: str
    output_file: str
    
def get_full_file_path(file_path: str, sub_directory: str = "") -> str:

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sub_dir = os.path.join(parent_dir, sub_directory)
    return os.path.join(sub_dir, file_path)
            
@app.post("/extract/")
def extract_content(req: SummarizeRequest):
    try:
        print(req)
        # Construct full file path under 'documents' directory
        full_file_path = get_full_file_path(req.file_path, "documents")
        print(f"[DEBUG] Requested input file: {full_file_path}")
        texts = extract(full_file_path)
        print(f"[DEBUG] Extracted text chunks: {len(texts)}")
        
        base_name = os.path.basename(full_file_path)
        file_name_without_ext = os.path.splitext(base_name)[0]
        output_full_file_path = get_full_file_path(f"{file_name_without_ext}_deepseek_summary.txt", "output")
        
        print(f"OUTPUT PATH: {output_full_file_path}") 
        data = req.dict()
        data["full_file_path"] = full_file_path
        return data
    except Exception as e:
        import traceback
        print(f"[ERROR] Exception in extract_content: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

def summarize_and_save_core(
    req: SummarizeRequest,
    llm,
    embeddings,
    summary_suffix: str = "summary"
) -> SummarizeResponse:
    full_file_path = get_full_file_path(req.file_path, "documents")
    print(f"[DEBUG] Requested input file: {full_file_path}")
    texts = extract(full_file_path)
    print(f"[DEBUG] Extracted text chunks: {len(texts)}")

    summary = summarize_document_with_kmeans_clustering(
        texts, llm, embeddings, num_clusters=req.num_clusters
    )
    print(f"[DEBUG] Summary length: {len(summary)}")

    base_name = os.path.basename(full_file_path)
    file_name_without_ext = os.path.splitext(base_name)[0]
    output_full_file_path = get_full_file_path(
        f"{file_name_without_ext}_{summary_suffix}.txt", "output"
    )
    with open(output_full_file_path, "w", encoding="utf-8") as f:
        f.write(summary)

    return SummarizeResponse(summary=summary, output_file=output_full_file_path)

@app.post("/summarize_and_save/", response_model=SummarizeResponse)
def summarize_and_save_endpoint(req: SummarizeRequest):
    try:
        from langchain_ollama import OllamaLLM
        from langchain_huggingface import HuggingFaceEmbeddings
        llm = OllamaLLM(model=req.model, temperature=0)
        embeddings = HuggingFaceEmbeddings()
        return summarize_and_save_core(req, llm, embeddings, summary_suffix="summary")
    except Exception as e:
        import traceback
        print(f"[ERROR] Exception in summarize_and_save_endpoint: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize_deepseek/", response_model=SummarizeResponse)
def summarize_deepseek_endpoint(req: SummarizeRequest):
    try:
        # Use Ollama-style initialization for DeepSeek r-1
        from langchain_ollama import OllamaLLM
        from langchain_huggingface import HuggingFaceEmbeddings
        llm = OllamaLLM(model="deepseek-v2", temperature=0)
        embeddings = HuggingFaceEmbeddings()
        return summarize_and_save_core(req, llm, embeddings, summary_suffix="summary")
    except Exception as e:
        import traceback
        print(f"[ERROR] Exception in summarize_deepseek_endpoint: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize_llm/", response_model=SummarizeResponse)
def summarize_llm_endpoint(req: SummarizeRequest):
    try:
        # Dynamically import and initialize the LLM based on the model name
        llm = None
        if req.model.lower().startswith("deepseek"):
            from langchain_ollama import OllamaLLM
            llm = OllamaLLM(model=req.model, temperature=0)
        elif req.model.lower().startswith("llama"):
            from langchain_ollama import OllamaLLM
            llm = OllamaLLM(model=req.model, temperature=0)
        elif req.model.lower().startswith("gpt"):
            from langchain_community.llms import OpenAI
            llm = OpenAI(model=req.model)
        else:
            # Try to import a generic LLM class by name
            try:
                module_name, class_name = req.model.split(":", 1) if ":" in req.model else ("langchain_ollama", "OllamaLLM")
                module = __import__(module_name, fromlist=[class_name])
                llm_class = getattr(module, class_name)
                llm = llm_class(model=req.model, temperature=0)
            except Exception as import_err:
                raise HTTPException(status_code=400, detail=f"Could not import or initialize LLM for model '{req.model}': {import_err}")

        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings()
        return summarize_and_save_core(req, llm, embeddings, summary_suffix="summary")
    except Exception as e:
        import traceback
        print(f"[ERROR] Exception in summarize_llm_endpoint: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize_openai/", response_model=SummarizeResponse)
def summarize_openai_endpoint(req: SummarizeRequest):
    try:
        import os
        import openai
        os.environ["OPENAI_API_KEY"] = req.openai_api_key
        openai.api_key = req.openai_api_key
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.chat_models import ChatOpenAI
        llm = ChatOpenAI(model=req.model, openai_api_key=req.openai_api_key)
        embeddings = HuggingFaceEmbeddings()
        return summarize_and_save_core(req, llm, embeddings, summary_suffix="summary")
    except Exception as e:
        import traceback
        print(f"[ERROR] Exception in summarize_openai_endpoint: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

def main():
    if len(os.sys.argv) < 2:
        print("Usage: python main.py <document_path>")
        os.sys.exit(1)
    file_path = os.sys.argv[1]
    chunk_size = 2000
    chunk_overlap = 20
    num_clusters = 20
    from langchain_ollama import OllamaLLM
    from langchain_community.embeddings import HuggingFaceEmbeddings
    llm = OllamaLLM(model="llama3.1:8b", temperature=0)
    embeddings = HuggingFaceEmbeddings()
    print(f"Extracting and chunking document: {file_path}")
    texts = extract(file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap, save_loaded=True)
    print(f"Summarizing document with clustering...")
    summary = summarize_document_with_kmeans_clustering(texts, llm, embeddings, num_clusters=num_clusters)
    print("Summary generated. Saving to PDF...")
    append_summary_to_pdf("summary_output.pdf", summary)
    print("Done.")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",       # "module_name:app_instance"
        host="0.0.0.0",
        port=8000,
        reload=True       # auto-reloads on code changes
    )