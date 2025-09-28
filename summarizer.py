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
