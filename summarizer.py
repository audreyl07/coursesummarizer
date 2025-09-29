from langchain_community.document_transformers import EmbeddingsClusteringFilter
from langchain_core.documents import Document
from document_categorizer import categorize_chunks
from model_collaborator import ModelCollaborator


def get_llm(model_name="ollama"):
    """
    Dynamically load the requested LLM model.
    Supported: 'ollama', 'gpt', 'claude', 'gemini'
    """
    if model_name.lower() == "ollama":
        from langchain_community.llms import Ollama
        return Ollama()
    elif model_name.lower() == "gpt":
        from langchain_community.llms import OpenAI
        return OpenAI()
    elif model_name.lower() == "claude":
        from langchain_community.llms import Anthropic
        return Anthropic()
    elif model_name.lower() == "gemini":
        from langchain_community.llms import GoogleVertexAI
        return GoogleVertexAI(model_name="gemini-pro")
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def summarize_document_with_kmeans_clustering(texts, embeddings, num_clusters=20):
    """
    Cluster document chunks using embeddings, then summarize each cluster using multiple specialized LLMs.
    Includes code and image/graph examples in the summary output.
    """
    # Cluster the documents
    filter = EmbeddingsClusteringFilter(embeddings=embeddings, num_clusters=num_clusters)
    filtered_docs = filter.transform_documents(documents=texts)

    # Categorize chunks into dictionary
    categorized_chunks = categorize_chunks(filtered_docs)
    
    # Initialize model collaboration
    collaborator = ModelCollaborator(get_llm)
    
    # Get collaborative summary
    summary = collaborator.collaborate(**categorized_chunks)

    # Gather code blocks and images from metadata for appendix
    all_code_blocks = []
    all_images = []
    for doc in filtered_docs:
        if "code_blocks" in doc.metadata:
            all_code_blocks.extend(doc.metadata["code_blocks"])
        if "images" in doc.metadata:
            all_images.extend(doc.metadata["images"])

    # Add appendices
    appendix = ""
    if all_code_blocks:
        appendix += "\n\n=== Extracted Code Blocks ===\n"
        for idx, code in enumerate(all_code_blocks, 1):
            indented_code = '\n'.join('    ' + line for line in code.strip().splitlines())
            appendix += f"\nCode Block {idx}:\n" + indented_code + "\n"
    if all_images:
        appendix += "\n\n=== Image/Graph References ===\n"
        for idx, img in enumerate(all_images, 1):
            appendix += f"Image/Graph {idx}: {img}\n"
    
    return summary + appendix


def cross_check_summary(summary: str, source_text: str) -> dict:
    """
    Cross-checks the summarized content against the original document text for accuracy.
    Returns a report with confidence score and discrepancies.
    """
    import difflib
    import re

    report = {
        "missing": [],
        "altered": [],
        "hallucinated": [],
        "confidence": 1.0,
    }

    # Normalize text for comparison
    def normalize(text):
        return re.sub(r"\s+", " ", text.strip().lower())

    norm_source = normalize(source_text)
    norm_summary = normalize(summary)

    # Split summary into sentences/lines for checking
    summary_lines = [line.strip() for line in re.split(r'[\n\.!?]', summary) if line.strip()]

    # Check each line in summary
    for line in summary_lines:
        norm_line = normalize(line)
        if norm_line and norm_line not in norm_source:
            # Try fuzzy match
            matches = difflib.get_close_matches(norm_line, [norm_source], n=1, cutoff=0.8)
            if not matches:
                report["hallucinated"].append(line)
            else:
                report["altered"].append(line)

    # Check for missing key facts/code/math/graphs
    # Example: look for code blocks, equations, graph/image references
    code_blocks = re.findall(r'```[\w+\-]*\n([\s\S]*?)```', source_text)
    for code in code_blocks:
        if normalize(code) not in norm_summary:
            report["missing"].append(f"Code block: {code[:50]}...")

    equations = re.findall(r'\$\$(.*?)\$\$', source_text)
    for eq in equations:
        if normalize(eq) not in norm_summary:
            report["missing"].append(f"Equation: {eq}")

    images = re.findall(r'\!\[([^\]]*)\]\(([^\)]*)\)', source_text)
    for desc, ref in images:
        if normalize(desc) not in norm_summary:
            report["missing"].append(f"Image/Graph: {desc}")

    # Confidence score: penalize for hallucinated/missing/altered
    total_lines = len(summary_lines)
    penalty = (len(report["hallucinated"]) + len(report["missing"]) + 0.5 * len(report["altered"]))
    if total_lines > 0:
        report["confidence"] = max(0.0, 1.0 - penalty / total_lines)
    else:
        report["confidence"] = 0.0

    return report
