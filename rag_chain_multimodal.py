"""
Multimodal RAG Chain Module
Implements Retrieval-Augmented Generation with Gemini, Qdrant, and multimodal support.
"""

from typing import List, Dict, AsyncGenerator, Optional, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from vector_store_multimodal import MultimodalVectorStore
from hybrid_retriever import HybridRetriever, MultimodalContextBuilder


MULTIMODAL_RAG_PROMPT = """You are an intelligent and helpful AI assistant that answers questions based on the provided context, which includes both text and images.

Instructions:
1. Use ONLY the information from the provided context (text and images) to answer.
2. If the information is not present in the context, state it clearly.
3. Be concise yet thorough in your responses.
4. Cite sources whenever relevant (mention which document the information comes from).
5. If there are images in the context, analyze them and describe what they display.
6. If the user requests to see an image, describe the image present in the context.
7. NEVER state that there are no images if the context indicates that images are available.

Text context:
{context}

{image_context}

IMPORTANT: Images are included in this message. If image_context indicates that images are available, ANALYZE them and ANSWER based on them.

Conversation history:
{chat_history}
"""


class MultimodalRAGChain:
    """
    Multimodal RAG Chain with Gemini 2.5 Flash.
    Supports text and image context for multimodal queries.
    """
    
    def __init__(
        self,
        vector_store: MultimodalVectorStore,
        hybrid_retriever: HybridRetriever,
        google_api_key: str = None,
        model_name: str = "gemini-2.5-flash"
    ):
        self.vector_store = vector_store
        self.hybrid_retriever = hybrid_retriever
        self.context_builder = MultimodalContextBuilder(max_images=3)
        
        # Initialize Gemini for text (streaming)
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=google_api_key,
            temperature=0.7,
            streaming=True
        )
        
        # Initialize native Gemini client for multimodal (using new google.genai)
        from google import genai
        self.genai_client = genai.Client(api_key=google_api_key)
        self.model_name = model_name
        
        self.chat_history: List[Dict] = []
    
    def _format_history(self) -> str:
        """Format chat history for context."""
        if not self.chat_history:
            return "No previous conversation history."
        
        formatted = []
        for msg in self.chat_history[-6:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        return "\n".join(formatted)
    
    def add_to_history(self, role: str, content: str):
        """Add message to chat history."""
        self.chat_history.append({"role": role, "content": content})
    
    def clear_history(self):
        """Clear chat history."""
        self.chat_history = []
    
    def retrieve_context(
        self,
        query: str,
        k: int = 5,
        text_k: int = 10,
        image_k: int = 5
    ) -> Tuple[List[Document], List[Document]]:
        """
        Retrieve relevant text and images for a query.
        
        Returns:
            Tuple of (text_docs, image_docs)
        """
        return self.hybrid_retriever.retrieve(
            query=query,
            k=k,
            text_k=text_k,
            image_k=image_k
        )
    
    def retrieve_context_with_scores(
        self,
        query: str,
        k: int = 5,
        text_k: int = 10,
        image_k: int = 10,
        image_score_threshold: float = 0.3
    ) -> Tuple[List[Document], List[Document], List[Tuple[Document, float]]]:
        """
        Retrieve relevant text and images with image scores.
        
        Args:
            query: Search query
            k: Number of final fused results
            text_k: Number of text results to fetch
            image_k: Number of image results to fetch
            image_score_threshold: Minimum score for images to be included
            
        Returns:
            Tuple of (text_docs, image_docs, image_results_with_scores)
        """
        return self.hybrid_retriever.retrieve_with_image_scores(
            query=query,
            k=k,
            text_k=text_k,
            image_k=image_k,
            image_score_threshold=image_score_threshold
        )
    
    async def astream_multimodal(
        self,
        query: str,
        text_docs: List[Document] = None,
        image_docs: List[Document] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream response with multimodal context (text + images).
        
        Uses native Gemini API for multimodal support.
        """
        self.add_to_history("user", query)
        
        # Retrieve if not provided
        if text_docs is None or image_docs is None:
            text_docs, image_docs = self.retrieve_context(query)
        
        # Build context
        text_context, image_parts = self.context_builder.build_context(text_docs, image_docs)
        chat_history = self._format_history()
        
        # Debug: log image parts and metadata
        print(f"[DEBUG] build_context returned {len(image_parts)} image parts")
        print(f"[DEBUG] image_docs count: {len(image_docs)}")
        if image_docs:
            print(f"[DEBUG] First image doc metadata keys: {list(image_docs[0].metadata.keys())}")
            print(f"[DEBUG] First image has base64_image: {'base64_image' in image_docs[0].metadata}")
        if image_parts:
            print(f"[DEBUG] First image has {len(image_parts[0].get('data', ''))} chars of base64 data")
        
        # Build image context description
        image_context = ""
        if image_parts:
            image_context = f"AVAILABLE IMAGES: {len(image_parts)} relevant image(s) have been provided related to your question. These images are shown below and you can analyze them directly."
        else:
            image_context = "AVAILABLE IMAGES: No relevant images were found for this query."
        
        
        # Build system prompt
        system_prompt = MULTIMODAL_RAG_PROMPT.format(
            context=text_context,
            image_context=image_context,
            chat_history=chat_history
        )
        
        try:
            # Stream from Gemini using new google.genai API
            from google.genai import types
            import base64
            
            # Build parts list - text first, then images
            parts = []
            
            # Add text prompt
            text_content = f"{system_prompt}\n\nUser Question: {query}"
            parts.append(types.Part(text=text_content))
            
            # Add images as inline data in the SAME message
            for i, img_part in enumerate(image_parts):
                img_bytes = base64.b64decode(img_part["data"])
                parts.append(types.Part(
                    inline_data=types.Blob(
                        data=img_bytes,
                        mime_type="image/jpeg"
                    )
                ))
            
            # Create single content with all parts
            contents = [types.Content(
                role="user",
                parts=parts
            )]
            
            # Stream response
            response = self.genai_client.models.generate_content_stream(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=2048
                )
            )
            
            full_response = ""
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    yield chunk.text
            
            if full_response:
                self.add_to_history("assistant", full_response)
                
        except Exception as e:
            yield f"\n[Error: {str(e)}]"
            return
    
    async def astream(
        self,
        query: str,
        text_docs: List[Document] = None,
        image_docs: List[Document] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream response - delegates to multimodal method.
        """
        async for chunk in self.astream_multimodal(query, text_docs, image_docs):
            yield chunk
    
    async def query(self, query: str) -> str:
        """Get complete response for a query (non-streaming)."""
        full_response = ""
        async for chunk in self.astream(query):
            full_response += chunk
        return full_response
    
    def get_relevant_documents(
        self,
        query: str,
        k: int = 5,
        include_images: bool = True
    ) -> Tuple[List[Document], List[Document]]:
        """
        Get relevant documents without generating response.
        
        Returns:
            Tuple of (text_documents, image_documents)
        """
        return self.retrieve_context(query, k=k)
    
    def get_context_for_display(
        self,
        text_docs: List[Document],
        image_docs: List[Document],
        min_image_score: float = -1.0
    ) -> Dict:
        """
        Get formatted context for display in frontend.
        
        Args:
            text_docs: List of text documents
            image_docs: List of image documents with relevance_score in metadata
            min_image_score: Minimum relevance score for images (-1.0 = all)
        """
        return {
            "text_sources": [
                {
                    "source": doc.metadata.get("source", "Unknown"),
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "chunk_index": doc.metadata.get("chunk_index")
                }
                for doc in text_docs[:3]
            ],
            "image_sources": self.context_builder.get_image_sources(image_docs, min_score=min_image_score)
        }
