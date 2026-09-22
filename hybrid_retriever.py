"""
Hybrid Retriever Module
Implements Reciprocal Rank Fusion (RRF) for combining text and image retrieval results.
"""

from typing import List, Dict, Tuple, Optional
from langchain_core.documents import Document
from vector_store_multimodal import MultimodalVectorStore
from image_embeddings import ImageEmbedder


class HybridRetriever:
    """
    Hybrid retriever that combines text and image search results
    using Reciprocal Rank Fusion (RRF).
    """
    
    def __init__(
        self,
        vector_store: MultimodalVectorStore,
        image_embedder: ImageEmbedder = None,
        rrf_k: int = 60,  # RRF constant
        text_weight: float = 1.0,
        image_weight: float = 1.0
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: MultimodalVectorStore instance
            image_embedder: ImageEmbedder instance for query image encoding
            rrf_k: RRF constant (default 60 as per original paper)
            text_weight: Weight for text results
            image_weight: Weight for image results
        """
        self.vector_store = vector_store
        self.image_embedder = image_embedder or ImageEmbedder()
        self.rrf_k = rrf_k
        self.text_weight = text_weight
        self.image_weight = image_weight
    
    def _reciprocal_rank_fusion(
        self,
        text_results: List[Tuple[Document, float]],
        image_results: List[Tuple[Document, float]],
        k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        RRF score = sum(1 / (k + rank)) for each result list
        
        Args:
            text_results: List of (Document, score) from text search
            image_results: List of (Document, score) from image search
            k: Number of final results to return
            
        Returns:
            Fused list of (Document, rrf_score) tuples
        """
        # Build rank maps
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        
        # Process text results
        for rank, (doc, _) in enumerate(text_results, 1):
            doc_id = self._get_doc_id(doc)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (self.text_weight / (self.rrf_k + rank))
            doc_map[doc_id] = doc
        
        # Process image results
        for rank, (doc, _) in enumerate(image_results, 1):
            doc_id = self._get_doc_id(doc)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (self.image_weight / (self.rrf_k + rank))
            doc_map[doc_id] = doc
        
        # Sort by RRF score
        sorted_docs = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top k
        return [(doc_map[doc_id], score) for doc_id, score in sorted_docs[:k]]
    
    def _get_doc_id(self, doc: Document) -> str:
        """Generate a unique ID for a document."""
        # Use hash + source + chunk_index if available
        file_hash = doc.metadata.get("file_hash", "")
        source = doc.metadata.get("source", "")
        chunk_index = doc.metadata.get("chunk_index", "")
        image_hash = doc.metadata.get("image_hash", "")
        
        # Create unique ID
        if image_hash:
            return f"img_{image_hash}"
        elif file_hash and chunk_index != "":
            return f"txt_{file_hash}_{chunk_index}"
        else:
            # Fallback to content hash
            import hashlib
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:12]
            return f"doc_{content_hash}"
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        text_k: int = 10,
        image_k: int = 10,
        include_images: bool = True
    ) -> Tuple[List[Document], List[Document]]:
        """
        Retrieve relevant text and images for a query.
        
        Args:
            query: Search query
            k: Number of final fused results
            text_k: Number of text results to fetch
            image_k: Number of image results to fetch
            include_images: Whether to include image search
            
        Returns:
            Tuple of (text_documents, image_documents)
        """
        text_docs, image_docs, _ = self.retrieve_with_image_scores(
            query=query, k=k, text_k=text_k, image_k=image_k, include_images=include_images
        )
        return text_docs, image_docs
    
    def retrieve_with_image_scores(
        self,
        query: str,
        k: int = 5,
        text_k: int = 10,
        image_k: int = 10,
        include_images: bool = True,
        image_score_threshold: float = -1.0  # Include all by default
    ) -> Tuple[List[Document], List[Document], List[Tuple[Document, float]]]:
        """
        Retrieve relevant text and images with image scores for filtering.
        Uses HYBRID image search: text caption search (BGE-M3) + visual search (SigLIP).
        
        Args:
            query: Search query
            k: Number of final fused results
            text_k: Number of text results to fetch
            image_k: Number of image results to fetch
            include_images: Whether to include image search
            image_score_threshold: Minimum score for images to be included (-1.0 = all)
            
        Returns:
            Tuple of (text_documents, image_documents, image_results_with_scores)
        """
        # Text search
        text_results = self.vector_store.search_text(query, k=text_k)
        
        image_results = []
        if include_images:
            # HYBRID IMAGE SEARCH: Combine visual and text-based search
            
            # 1. Visual search using SigLIP text encoder
            query_image_embedding = self.image_embedder.encode_text(query)
            visual_results = self.vector_store.search_images(query_image_embedding, k=image_k)
            
            # Debug: show visual results
            print("\n[Visual Search - SigLIP]:")
            for doc, score in visual_results[:5]:
                print(f"  {doc.metadata.get('source', 'Unknown')}: {score:.4f}")
            
            # 2. Text-based search using BGE-M3 on image captions
            # Images are stored with captions in page_content, so we search them as text
            # Get all image documents first
            all_images = self.vector_store.get_all_images()
            
            # Search images by their captions using text embedding
            caption_results = self.vector_store.search_text_in_images(query, k=image_k)
            
            # Debug: show caption results
            print("\n[Caption Search - BGE-M3]:")
            for doc, score in caption_results[:5]:
                print(f"  {doc.metadata.get('source', 'Unknown')}: {score:.4f}")
            
            # 3. Combine results with RRF fusion
            image_results = self._fuse_image_results(visual_results, caption_results, k=image_k)
            
            # Debug: show fused results
            print("\n[Fused Results]:")
            for doc, score in image_results[:5]:
                print(f"  {doc.metadata.get('source', 'Unknown')}: {score:.4f}")
        
        
        # Separate text and image documents
        text_docs = [doc for doc, score in text_results]
        
        # Filter images by score threshold
        filtered_image_docs = []
        for doc, score in image_results:
            if score >= image_score_threshold:
                # Add score to metadata for later use
                doc.metadata["relevance_score"] = score
                filtered_image_docs.append(doc)
        
        
        return text_docs, filtered_image_docs, image_results
    
    def _fuse_image_results(
        self,
        visual_results: List[Tuple[Document, float]],
        caption_results: List[Tuple[Document, float]],
        k: int = 10,
        rrf_k: int = 60
    ) -> List[Tuple[Document, float]]:
        """
        Fuse visual and caption-based image results using Reciprocal Rank Fusion.
        
        Args:
            visual_results: Results from SigLIP visual search
            caption_results: Results from BGE-M3 text search on captions
            k: Number of final results
            rrf_k: RRF constant
            
        Returns:
            Fused and sorted list of (Document, score) tuples
        """
        # Build doc_id to scores mapping
        doc_scores = {}
        
        # Process visual results
        for rank, (doc, score) in enumerate(visual_results):
            doc_id = doc.metadata.get("source", str(id(doc)))
            rrf_score = 1.0 / (rrf_k + rank + 1)
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"doc": doc, "visual": 0, "caption": 0, "visual_score": 0, "caption_score": 0}
            doc_scores[doc_id]["visual"] = rrf_score
            doc_scores[doc_id]["visual_score"] = score
        
        # Process caption results
        for rank, (doc, score) in enumerate(caption_results):
            doc_id = doc.metadata.get("source", str(id(doc)))
            rrf_score = 1.0 / (rrf_k + rank + 1)
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"doc": doc, "visual": 0, "caption": 0, "visual_score": 0, "caption_score": 0}
            doc_scores[doc_id]["caption"] = rrf_score
            doc_scores[doc_id]["caption_score"] = score
        
        # Combine scores (weighted: caption 0.8, visual 0.2 for optimal text & visual balance)
        fused = []
        for doc_id, data in doc_scores.items():
            combined_score = 0.8 * data["caption"] + 0.2 * data["visual"]
            fused.append((data["doc"], combined_score))
        
        # Sort by combined score
        fused.sort(key=lambda x: x[1], reverse=True)
        
        return fused[:k]
    
    def retrieve_with_scores(
        self,
        query: str,
        k: int = 5,
        text_k: int = 10,
        image_k: int = 10,
        include_images: bool = True
    ) -> Tuple[List[Tuple[Document, float]], List[Tuple[Document, float]]]:
        """
        Retrieve with scores for debugging/transparency.
        
        Returns:
            Tuple of (text_results_with_scores, image_results_with_scores)
        """
        text_results = self.vector_store.search_text(query, k=text_k)
        
        image_results = []
        if include_images:
            # Encode query as image embedding using SigLIP's encode_text method
            query_image_embedding = self.image_embedder.encode_text(query)
            image_results = self.vector_store.search_images(query_image_embedding, k=image_k)
        
        return text_results, image_results
    
    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        text_k: int = 10,
        image_k: int = 10
    ) -> List[Tuple[Document, float]]:
        """
        Perform hybrid search with RRF fusion.
        
        Returns fused results from both text and image collections.
        """
        text_results, image_results = self.retrieve_with_scores(
            query, k=k, text_k=text_k, image_k=image_k
        )
        
        return self._reciprocal_rank_fusion(text_results, image_results, k=k)


class MultimodalContextBuilder:
    """
    Builds context for LLM from retrieved text and images.
    Formats content for Gemini multimodal input.
    """
    
    def __init__(self, max_images: int = 3):
        self.max_images = max_images
    
    def build_context(
        self,
        text_docs: List[Document],
        image_docs: List[Document]
    ) -> Tuple[str, List[Dict]]:
        """
        Build context string and image parts for Gemini.
        
        Returns:
            Tuple of (text_context, image_parts)
        """
        # Build text context
        text_context = self._format_text_docs(text_docs)
        
        # Build image parts (limited)
        image_parts = self._format_image_docs(image_docs[:self.max_images])
        
        return text_context, image_parts
    
    def _format_text_docs(self, docs: List[Document]) -> str:
        """Format text documents into context string."""
        if not docs:
            return "No relevant text documents found."
        
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            content = doc.page_content
            formatted.append(f"[Document {i} - {source}]\n{content}\n")
        
        return "\n".join(formatted)
    
    def _format_image_docs(self, docs: List[Document]) -> List[Dict]:
        """
        Format image documents for Gemini multimodal input.
        
        Returns list of dicts with inline_data for Gemini API.
        """
        image_parts = []
        
        for doc in docs:
            base64_image = doc.metadata.get("base64_image")
            if base64_image:
                image_parts.append({
                    "mime_type": "image/jpeg",
                    "data": base64_image
                })
        
        return image_parts
    
    def get_image_sources(self, image_docs: List[Document], min_score: float = -1.0) -> List[Dict]:
        """
        Get image metadata for display in frontend.
        
        Args:
            image_docs: List of image documents with relevance_score in metadata
            min_score: Minimum relevance score to include (-1.0 = all)
            
        Returns:
            List of image sources sorted by score (highest first)
        """
        sources = []
        
        for doc in image_docs:
            score = doc.metadata.get("relevance_score", 0)
            # Only include images with score above threshold
            if score >= min_score:
                sources.append({
                    "source": doc.metadata.get("source", "Unknown"),
                    "base64": doc.metadata.get("base64_image", ""),
                    "page": doc.metadata.get("page_number"),
                    "parent_pdf": doc.metadata.get("parent_pdf"),
                    "relevance_score": score
                })
        
        
        # Sort by score (highest first)
        sources.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        return sources
