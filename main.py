import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from document_loader import extract, save_text_to_file
from summarizer import summarize_document_with_kmeans_clustering
from pdf_manager import append_summary_to_pdf

app = FastAPI()

class Info(BaseModel):
    filename: str

class Item(BaseModel):
    inputfile: str
    outputfile: str | None = None


@app.post("/items/")
def create_item(item: Item):
    #texts = extract(item.filename)
    texts = extract("D:\\Github\\coursesummarizer\\documents\\DiscreteStructures.pdf")
    print(texts)
    # if item.price < 0:
    #     raise HTTPException(status_code=400, detail="Price must be non-negative")
  
    # data = item.dict()
    # if item.tax is not None:
    #     data["total_price"] = item.price + item.tax
    return texts

@app.post("/extract/")
def extract_content(item: Item):
    try:
        print(f"[DEBUG] Requested input file: {item.inputfile}")
        texts = extract(item.inputfile)
        if not texts or not isinstance(texts, str):
            raise ValueError("Extracted text is empty or not a string.")
        print(f"[DEBUG] Extracted text length: {len(texts)}")
        save_text_to_file(texts, item.outputfile)
        data = item.dict()
        return data
    except Exception as e:
        import traceback
        print(f"[ERROR] Exception in extract_content: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))



# Combined endpoint for summarizing a saved file and saving the summary to its own file
class SummarizeAndSaveRequest(BaseModel):
    txt_path: str
    model: str = "llama3.1:8b"
    num_clusters: int = 20
    output_file: str | None = None

class SummarizeAndSaveResponse(BaseModel):
    summary: str
    output_file: str

@app.post("/summarize_and_save/", response_model=SummarizeAndSaveResponse)
def summarize_and_save_endpoint(req: SummarizeAndSaveRequest):
    try:
        from langchain_ollama import OllamaLLM
        from langchain_huggingface import HuggingFaceEmbeddings
        llm = OllamaLLM(model=req.model, temperature=0)
        embeddings = HuggingFaceEmbeddings()

        print(f"[DEBUG] Requested txt_path: {req.txt_path}")
        texts = extract(req.txt_path)
        print(f"[DEBUG] Extracted text chunks: {len(texts)}")

        summary = summarize_document_with_kmeans_clustering(texts, llm, embeddings, num_clusters=req.num_clusters)
        print(f"[DEBUG] Summary length: {len(summary)}")

        # Save summary to its own file
        base_name = os.path.basename(req.txt_path)
        file_name_without_ext = os.path.splitext(base_name)[0]
        output_file = req.output_file or f"{file_name_without_ext}_summary.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(summary)

        return SummarizeAndSaveResponse(summary=summary, output_file=output_file)
    except Exception as e:
        import traceback
        print(f"[ERROR] Exception in summarize_and_save_endpoint: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for summarizing documents using the DeepSeek LLM model
class SummarizeDeepSeekRequest(BaseModel):
    file_path: str
    model: str = "deepseek-chat"
    num_clusters: int = 20
    output_file: str | None = None
    
class SummarizeDeepSeekResponse(BaseModel):
    summary: str
    output_file: str

@app.post("/summarize_deepseek/", response_model=SummarizeDeepSeekResponse)
def summarize_deepseek_endpoint(req: SummarizeDeepSeekRequest):
    try:
        # # Load .env file if DEEPSEEK_API_KEY is not set
        # if not os.getenv("DEEPSEEK_API_KEY"):
        #     from dotenv import load_dotenv
        #     load_dotenv()
        # api_key = os.getenv("DEEPSEEK_API_KEY")
        # if not api_key:
        #     raise HTTPException(status_code=500, detail="DEEPSEEK_API_KEY environment variable is not set. Please set it in your shell or .env file.")
        # os.environ["DEEPSEEK_API_KEY"] = api_key

    
        # Use Ollama-style initialization for DeepSeek r-1
        from langchain_ollama import OllamaLLM
        from langchain_huggingface import HuggingFaceEmbeddings
        llm = OllamaLLM(model="deepseek-r1", temperature=0)
        embeddings = HuggingFaceEmbeddings()

        print(f"[DEBUG] Requested file path: {req.file_path}")
        texts = extract(req.file_path)
        print(f"[DEBUG] Extracted text chunks: {len(texts)}")

        summary = summarize_document_with_kmeans_clustering(texts, llm, embeddings, num_clusters=req.num_clusters)
        print(f"[DEBUG] Summary length: {len(summary)}")

        # Save summary to its own file
        base_name = os.path.basename(req.file_path)
        file_name_without_ext = os.path.splitext(base_name)[0]
        output_file = req.output_file or f"{file_name_without_ext}_deepseek_summary.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(summary)

        return SummarizeDeepSeekResponse(summary=summary, output_file=output_file)
    except Exception as e:
        import traceback
        print(f"[ERROR] Exception in summarize_deepseek_endpoint: {e}")
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