from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document

class ModelCollaborator:
    """
    Handles collaboration between different LLM models for document summarization.
    Coordinates the interaction between models specialized in different content types.
    """
    
    def __init__(self, get_llm_func):
        """
        Initialize the collaborator with a function to get LLM instances.
        
        Args:
            get_llm_func: Function that returns an LLM instance given a model name
        """
        self.get_llm = get_llm_func
        self.specialized_prompts = self._initialize_prompts()
        self.models = self._initialize_models()
        
    def _initialize_models(self):
        """Initialize the specialized LLM models"""
        return {
            'code': self.get_llm('ollama'),     # Specialized for code analysis
            'math': self.get_llm('claude'),     # Excellent at mathematical reasoning
            'graphs': self.get_llm('gemini'),   # Strong visual and data analysis
            'general': self.get_llm('gpt'),     # Strong general comprehension and synthesis
        }
        
    def _initialize_prompts(self):
        """Initialize specialized prompts for different content types"""
        return {
            'code': PromptTemplate.from_template(
                """
Analyze and explain the following code-heavy content. Focus on:
- Clear explanation of code functionality and purpose
- Proper code formatting and syntax highlighting
- Implementation details and best practices
- How this code relates to the broader concepts
- Examples of usage and potential modifications

{text}
"""
            ),
            'math': PromptTemplate.from_template(
                """
Analyze and explain the following mathematical content. Focus on:
- Clear explanation of mathematical concepts and formulas
- Step-by-step breakdowns of calculations
- Visual representations or diagrams where applicable
- How these mathematical concepts connect to the broader topic
- Practical applications and examples

{text}
"""
            ),
            'graphs': PromptTemplate.from_template(
                """
Analyze and explain the graphs, charts, and visualizations in the content. Focus on:
- Clear interpretation of the graphs' purpose and meaning
- Key trends, patterns, and relationships shown
- Important data points and their significance
- How the visualizations support the main concepts
- Connections to mathematical concepts and practical applications
- Limitations or considerations in the data representation

{text}
"""
            ),
            'general': PromptTemplate.from_template(
                """
Synthesize and explain the following content. Focus on:
- Main concepts and their relationships
- Supporting examples and illustrations
- Integration with code, mathematical concepts, and visualizations
- Clear and cohesive narrative structure
- Real-world applications and implications

{text}
"""
            ),
            'synthesis': PromptTemplate.from_template(
                """
Create a cohesive final summary that integrates the following specialized summaries.
Ensure smooth transitions between topics and maintain clear connections between concepts.

Specialized Summaries:
{summaries}

Guidelines:
1. Start with an overview that sets the context
2. Integrate code examples with their theoretical foundations
3. Connect mathematical concepts to their practical applications
4. Explain how graphs and visualizations support key concepts
5. Maintain a clear narrative flow throughout
6. Highlight relationships between different topics
7. Provide practical takeaways and applications
8. Address potential areas for further exploration

Create a final, unified summary that connects all concepts:
"""
            )
        }
    
    def process_chunks(self, categorized_chunks):
        """
        Process different types of chunks with their specialized models.
        
        Args:
            categorized_chunks: Dictionary with keys 'code', 'math', 'general'
                              containing respective document chunks
        Returns:
            List of tuples (section_title, summary_content)
        """
        summaries = []
        
        for content_type, chunks in categorized_chunks.items():
            if not chunks:
                continue
                
            model = self.models.get(content_type)
            prompt = self.specialized_prompts.get(content_type)
            
            if model and prompt and chunks:
                chain = load_summarize_chain(model, chain_type="stuff", prompt=prompt)
                summary = chain.invoke({"input_documents": chunks})
                title = self._get_section_title(content_type)
                summaries.append((title, summary.get("output_text", "")))
                
        return summaries
    
    def create_final_synthesis(self, specialized_summaries):
        """
        Create a final synthesis combining all specialized summaries.
        
        Args:
            specialized_summaries: List of tuples (section_title, content)
        Returns:
            String containing the final synthesized summary
        """
        combined_summaries = "\n\n".join([f"{title}:\n{content}" 
                                        for title, content in specialized_summaries])
        
        synthesis_chain = load_summarize_chain(self.models['general'], chain_type="stuff")
        final_summary = synthesis_chain.invoke({
            "input_documents": [
                Document(
                    page_content=self.specialized_prompts['synthesis'].format(
                        summaries=combined_summaries
                    )
                )
            ]
        })
        
        return final_summary.get("output_text", "")
    
    def _get_section_title(self, content_type):
        """Get formatted section title for content type"""
        titles = {
            'code': "Code Implementation and Examples",
            'math': "Mathematical Concepts and Formulas",
            'graphs': "Graph and Visualization Analysis",
            'general': "Core Concepts and Context"
        }
        return titles.get(content_type, content_type.title())
    
    def collaborate(self, code_chunks, math_chunks, general_chunks):
        """
        Main collaboration method that orchestrates the entire summarization process.
        
        Args:
            code_chunks: List of code-related document chunks
            math_chunks: List of math-related document chunks
            general_chunks: List of general content chunks
        Returns:
            String containing the final synthesized summary
        """
        categorized_chunks = {
            'code': code_chunks,
            'math': math_chunks,
            'general': general_chunks
        }
        
        # Get specialized summaries from each model
        specialized_summaries = self.process_chunks(categorized_chunks)
        
        # Create final synthesis
        final_summary = self.create_final_synthesis(specialized_summaries)
        
        return final_summary