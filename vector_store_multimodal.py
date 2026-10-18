"""
Multimodal Vector Store Module
Manages two Qdrant collections:
- text_chunks: BGE-M3 embeddings for text
- image_embeddings: SigLIP embeddings for images
"""

import os
from typing import List, Dict, Optional, Set, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client.http import models
from langchain_core.documents import Document
import numpy as np

# BGE-M3 for text embeddings (multilingual, 1024 dims)
from FlagEmbedding import BGEM3FlagModel


TEXT_COLLECTION = "text_chunks"
IMAGE_COLLECTION = "image_embeddings"

# BGE-M3 embedding dimension
BGE_M3_DIM = 1024
# SigLIP 2 embedding dimension
SIGLIP_DIM = 1152


class TextEmbedder:
    """BGE-M3 text embedder for multilingual support."""
    
    def __init__(self, device: str = None):
        import torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        print(f"Loading BGE-M3 model on {device}...")
        
        self.model = BGEM3FlagModel(
            'BAAI/bge-m3',
            use_fp16=True if device == "cuda" else False,
            device=device
        )
        
        print(f"BGE-M3 model loaded (embedding dim: {BGE_M3_DIM})")
    
    def embed_texts(self, texts: List[str], batch_size: int = 12) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            output = self.model.encode(
                batch,
                batch_size=len(batch),
                max_length=8192,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False
            )
            batch_embeddings = output['dense_vecs'].tolist()
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        output = self.model.encode(
            [query],
            batch_size=1,
            max_length=8192,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )
        return output['dense_vecs'][0].tolist()


class MultimodalVectorStore:
    """
    Manages two Qdrant collections for multimodal RAG.
    """
    
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        text_embedder: TextEmbedder = None
    ):
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.text_embedder = text_embedder or TextEmbedder()
        
        # Ensure collections exist
        self._ensure_collections()
    
    def _ensure_collections(self):
        """Create collections if they don't exist."""
        collections = self.client.get_collections().collections
        collection_names = {c.name for c in collections}
        
        # Text collection
        if TEXT_COLLECTION not in collection_names:
            self.client.create_collection(
                collection_name=TEXT_COLLECTION,
                vectors_config=VectorParams(
                    size=BGE_M3_DIM,
                    distance=Distance.COSINE
                )
            )
            print(f"Created collection: {TEXT_COLLECTION}")
        
        # Image collection
        if IMAGE_COLLECTION not in collection_names:
            self.client.create_collection(
                collection_name=IMAGE_COLLECTION,
                vectors_config=VectorParams(
                    size=SIGLIP_DIM,
                    distance=Distance.COSINE
                )
            )
            print(f"Created collection: {IMAGE_COLLECTION}")
    
    def get_existing_hashes(self, collection: str = TEXT_COLLECTION) -> Set[str]:
        """Get all file hashes from a collection for deduplication."""
        if not self._collection_exists(collection):
            return set()
        
        existing_hashes = set()
        offset = None
        
        while True:
            points, offset = self.client.scroll(
                collection_name=collection,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            for point in points:
                if point.payload and "file_hash" in point.payload:
                    existing_hashes.add(point.payload["file_hash"])
            
            if offset is None:
                break
        
        print(f"Found {len(existing_hashes)} existing hashes in {collection}")
        return existing_hashes
    
    def _collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        collections = self.client.get_collections().collections
        return any(c.name == collection_name for c in collections)
    
    def add_text_documents(self, documents: List[Document]) -> int:
        """
        Add text documents to the text collection.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            Number of documents added
        """
        if not documents:
            return 0
        
        # Generate embeddings
        texts = [doc.page_content for doc in documents]
        embeddings = self.text_embedder.embed_texts(texts)
        
        # Get current point count for ID generation
        collection_info = self.client.get_collection(TEXT_COLLECTION)
        start_id = collection_info.points_count
        
        # Create points
        points = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            point = PointStruct(
                id=start_id + i,
                vector=embedding,
                payload={
                    "page_content": doc.page_content,
                    **doc.metadata
                }
            )
            points.append(point)
        
        # Upload to Qdrant
        self.client.upsert(
            collection_name=TEXT_COLLECTION,
            points=points
        )
        
        print(f"Added {len(points)} text documents to {TEXT_COLLECTION}")
        return len(points)
    
    def add_image_embeddings(self, image_data: List[Dict]) -> int:
        """
        Add image embeddings to the image collection.
        
        Args:
            image_data: List of dicts with 'document', 'embedding', 'base64_image', 'hash'
            
        Returns:
            Number of images added
        """
        if not image_data:
            return 0
        
        # Get current point count for ID generation
        collection_info = self.client.get_collection(IMAGE_COLLECTION)
        start_id = collection_info.points_count
        
        # Create points
        points = []
        for i, img_data in enumerate(image_data):
            doc = img_data["document"]
            
            point = PointStruct(
                id=start_id + i,
                vector=img_data["embedding"],
                payload={
                    "page_content": doc.page_content,
                    "base64_image": img_data["base64_image"],
                    "image_hash": img_data["hash"],
                    **doc.metadata
                }
            )
            points.append(point)
        
        # Upload to Qdrant
        self.client.upsert(
            collection_name=IMAGE_COLLECTION,
            points=points
        )
        
        print(f"Added {len(points)} images to {IMAGE_COLLECTION}")
        return len(points)
    
    def search_text(
        self,
        query: str,
        k: int = 5,
        filter_conditions: List[FieldCondition] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search text collection.
        
        Returns:
            List of (Document, score) tuples
        """
        query_embedding = self.text_embedder.embed_query(query)
        
        query_filter = None
        if filter_conditions:
            query_filter = Filter(must=filter_conditions)
        
        results = self.client.query_points(
            collection_name=TEXT_COLLECTION,
            query=query_embedding,
            limit=k,
            query_filter=query_filter
        )
        
        documents = []
        for result in results.points:
            doc = Document(
                page_content=result.payload.get("page_content", ""),
                metadata={k: v for k, v in result.payload.items() if k != "page_content"}
            )
            documents.append((doc, result.score))
        
        return documents
    
    def search_images(
        self,
        query_embedding: List[float],
        k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        Search image collection with a pre-computed embedding.
        
        Returns:
            List of (Document, score) tuples with base64_image in metadata
        """
        results = self.client.query_points(
            collection_name=IMAGE_COLLECTION,
            query=query_embedding,
            limit=k
        )
        
        documents = []
        for result in results.points:
            doc = Document(
                page_content=result.payload.get("page_content", ""),
                metadata={k: v for k, v in result.payload.items() if k != "page_content"}
            )
            documents.append((doc, result.score))
        
        return documents
    
    def get_all_images(self, limit: int = 100) -> List[Document]:
        """
        Get all image documents from the image collection.
        
        Returns:
            List of Document objects with base64_image in metadata
        """
        results = self.client.scroll(
            collection_name=IMAGE_COLLECTION,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        
        documents = []
        for point in results[0]:
            doc = Document(
                page_content=point.payload.get("page_content", ""),
                metadata={k: v for k, v in point.payload.items() if k != "page_content"}
            )
            documents.append(doc)
        
        return documents
    
    def search_text_in_images(
        self,
        query: str,
        k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        Search images by their text captions using BGE-M3 text embedding.
        Uses cached caption embeddings for faster retrieval.
        
        Args:
            query: Text query to search in image captions
            k: Number of results to return
            
        Returns:
            List of (Document, score) tuples
        """
        import numpy as np
        
        # Get all images with their captions
        all_images = self.get_all_images(limit=100)
        
        if not all_images:
            return []
        
        # Get text embedding for the query (single embedding)
        query_embedding = np.array(self.text_embedder.embed_query(query))
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Check if images have cached text embeddings
        # If not, compute and cache them
        caption_embeddings = []
        for doc in all_images:
            if "text_embedding" in doc.metadata and doc.metadata["text_embedding"]:
                # Use cached embedding
                caption_embeddings.append(np.array(doc.metadata["text_embedding"]))
            else:
                # Need to compute embeddings for all (first time)
                # This is the slow path - only happens once
                captions = [d.page_content for d in all_images]
                caption_embeddings = np.array(self.text_embedder.embed_texts(captions))
                caption_embeddings = caption_embeddings / np.linalg.norm(caption_embeddings, axis=1, keepdims=True)
                break
        
        if isinstance(caption_embeddings, list):
            caption_embeddings = np.array(caption_embeddings)
        
        # Ensure normalized
        if caption_embeddings.ndim == 1:
            caption_embeddings = caption_embeddings.reshape(1, -1)
        norms = np.linalg.norm(caption_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        caption_embeddings = caption_embeddings / norms
        
        # Calculate similarities
        similarities = np.dot(caption_embeddings, query_embedding)
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        # Build results
        documents = []
        for idx in top_indices:
            doc = all_images[idx]
            score = float(similarities[idx])
            documents.append((doc, score))
        
        return documents
    
    def get_collection_info(self, collection: str = None) -> Dict:
        """Get information about collections."""
        info = {}
        
        for col_name in [TEXT_COLLECTION, IMAGE_COLLECTION]:
            if self._collection_exists(col_name):
                col_info = self.client.get_collection(col_name)
                info[col_name] = {
                    "exists": True,
                    "points_count": col_info.points_count,
                    "status": col_info.status.value
                }
            else:
                info[col_name] = {"exists": False}
        
        if collection:
            return info.get(collection, {"exists": False})
        return info
    
    def clear_collection(self, collection: str = None):
        """Clear one or both collections."""
        collections_to_clear = [collection] if collection else [TEXT_COLLECTION, IMAGE_COLLECTION]
        
        for col in collections_to_clear:
            if self._collection_exists(col):
                self.client.delete_collection(col)
                print(f"Deleted collection: {col}")
        
        # Recreate collections
        self._ensure_collections()
    
    def get_all_hashes(self) -> Dict[str, Set[str]]:
        """Get hashes from both collections."""
        return {
            "text": self.get_existing_hashes(TEXT_COLLECTION),
            "image": self.get_existing_hashes(IMAGE_COLLECTION)
        }
