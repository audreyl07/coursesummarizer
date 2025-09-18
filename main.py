from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import document_loader, summarizer, pdf_manager

app = FastAPI()

class Info(BaseModel):
    filename: str

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

@app.post("/items/")
def create_item(item: Item):
    if item.price < 0:
        raise HTTPException(status_code=400, detail="Price must be non-negative")
    data = item.dict()
    if item.tax is not None:
        data["total_price"] = item.price + item.tax
    return data

@app.post("/extract-pdf/")
def extract_pdf(info: Info):
    try:
        text = document_loader(info.filename)
        # Step 1: Get "COMP2401_Ch1_SystemsProgramming.pdf"
        base_name = os.path.basename(info.filename)

        # Step 2: Split into ("COMP2401_Ch1_SystemsProgramming", ".pdf") and take [0]
        file_name_without_ext = os.path.splitext(base_name)[0]

        print(text)
        summarizer(text, file_name_without_ext + ".txt")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    data = info.dict()
    return data

if __name__ == "__main__":
    uvicorn.run(
        "main:app",       # "module_name:app_instance"
        host="0.0.0.0",
        port=8000,
        reload=True       # auto-reloads on code changes
    )