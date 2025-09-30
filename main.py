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
        # Use Ollama-style initialization for DeepSeek r-1
        from langchain_ollama import OllamaLLM
        from langchain_huggingface import HuggingFaceEmbeddings
        llm = OllamaLLM(model="deepseek-v2", temperature=0)
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

# Generic endpoint for summarizing documents with any LLM model
class SummarizeLLMRequest(BaseModel):
    file_path: str
    model: str
    num_clusters: int = 20
    output_file: str | None = None

class SummarizeLLMResponse(BaseModel):
    summary: str
    output_file: str

@app.post("/summarize_llm/", response_model=SummarizeLLMResponse)
def summarize_llm_endpoint(req: SummarizeLLMRequest):
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

        print(f"[DEBUG] Requested file path: {req.file_path}")
        texts = extract(req.file_path)
        print(f"[DEBUG] Extracted text chunks: {len(texts)}")

        summary = summarize_document_with_kmeans_clustering(texts, llm, embeddings, num_clusters=req.num_clusters)
        print(f"[DEBUG] Summary length: {len(summary)}")

        base_name = os.path.basename(req.file_path)
        file_name_without_ext = os.path.splitext(base_name)[0]
        output_file = req.output_file or f"{file_name_without_ext}_{req.model}_summary.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(summary)

        return SummarizeLLMResponse(summary=summary, output_file=output_file)
    except Exception as e:
        import traceback
        print(f"[ERROR] Exception in summarize_llm_endpoint: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for summarizing documents using OpenAI's LLM
class SummarizeOpenAIRequest(BaseModel):
    file_path: str
    openai_api_key: str
    model: str = "gpt-4.1"	
    num_clusters: int = 20
    output_file: str | None = None

class SummarizeOpenAIResponse(BaseModel):
    summary: str
    output_file: str

@app.post("/summarize_openai/", response_model=SummarizeOpenAIResponse)
def summarize_openai_endpoint(req: SummarizeOpenAIRequest):
    try:
        import os
        import openai
        os.environ["OPENAI_API_KEY"] = req.openai_api_key
        openai.api_key = req.openai_api_key
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings()

        print(f"[DEBUG] Requested file path: {req.file_path}")
        texts = extract(req.file_path)
        print(f"[DEBUG] Extracted text chunks: {len(texts)}")

        # Compose the prompt for summarization
        prompt = "Summarize the following document. Only use English and ensure all sentences are grammatically correct and well-structured.\n\n" + "\n\n".join([doc.page_content for doc in texts])
        messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes documents."},
            {"role": "user", "content": prompt}
        ]
        response = openai.ChatCompletion.create(
            model=req.model,
            messages=messages
        )
        summary = response.choices[0].message.content
        print(f"[DEBUG] Summary length: {len(summary)}")

        base_name = os.path.basename(req.file_path)
        file_name_without_ext = os.path.splitext(base_name)[0]
        output_file = req.output_file or f"{file_name_without_ext}_openai_summary.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(summary)

        return SummarizeOpenAIResponse(summary=summary, output_file=output_file)
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