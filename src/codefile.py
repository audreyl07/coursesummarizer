import pdfplumber
import subprocess

def pdf_to_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Extract text from PDF
pdf_path = "D:\\work\\coursepdfproject\\COMP2401_Ch1_SystemsProgramming.pdf"  # Replace with your PDF file path
text = pdf_to_text(pdf_path)

# Save text to file
with open("pdf_text.txt", "w", encoding="utf-8") as f:
    f.write(text)
print("PDF text saved to pdf_text.txt")

# Send text to Ollama LLM and get summary
prompt = "Summarize the following text:\n" + text
# print("prompt:" + prompt)

result = subprocess.run(
    ["ollama", "run", "llama3.1:8b"],
    input=prompt.encode("utf-8"),
    capture_output=True
)
print("Summary from LLM:")
print(result.stdout.decode("utf-8"))