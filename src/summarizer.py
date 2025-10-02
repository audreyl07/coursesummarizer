from langchain_community.document_transformers import EmbeddingsClusteringFilter
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

def summarize_document_with_kmeans_clustering(texts, llm, embeddings, num_clusters=20):
    """
    Cluster document chunks using embeddings, then summarize each cluster using an LLM.
    Includes code and image/graph examples in the summary output.
    """
    filter = EmbeddingsClusteringFilter(embeddings=embeddings, num_clusters=num_clusters)
    filtered_docs = filter.transform_documents(documents=texts)

    # Gather code blocks and images from metadata for appendix
    all_code_blocks = []
    all_images = []
    for doc in filtered_docs:
        if "code_blocks" in doc.metadata:
            all_code_blocks.extend(doc.metadata["code_blocks"])
        if "images" in doc.metadata:
            all_images.extend(doc.metadata["images"])

    # Prompt for LLM summarization
    custom_prompt = PromptTemplate.from_template(
        """
Summarize the following document. For each section or heading:
- **Code:** If there are code examples, format them as code blocks with clear syntax, proper indentation, and add a brief explanation of what the code does and why it is relevant.
- **Math:** If there are mathematical formulas, equations, or calculations, explain their meaning, context, and how they relate to the topic. Provide step-by-step breakdowns if possible.
- **Graphs/Images/Plots:** If there are images, graphs, charts, or plots, describe their content, purpose, and relevance to the specific topic. Explain what the visualization shows and how it supports the document's points.
- **Text:** Clearly explain the main concepts, arguments, and supporting details. Make connections between ideas and provide context for understanding.
- **General Text:** Synthesize any other information, ensuring clarity and completeness. Integrate code, math, and visual elements into the overall summary for a cohesive understanding.
- Use clear, concise language and avoid unnecessary repetition.
- **IMPORTANT:** Only use English in your output and ensure all sentences are grammatically correct and well-structured.

\n\n{text}
"""
    )
    summarization_chain = load_summarize_chain(
        llm, chain_type="stuff", prompt=custom_prompt
    )
    result = summarization_chain.invoke({"input_documents": filtered_docs})

    # Compose summary with code and image/graph appendix
    summary_text = result.get("output_text", str(result)) if isinstance(result, dict) else str(result)
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
    
    return summary_text + appendix
