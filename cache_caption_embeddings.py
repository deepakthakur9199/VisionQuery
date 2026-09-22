"""
Script to pre-compute and cache text embeddings for image captions.
This significantly speeds up the hybrid image search.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

from image_embeddings import ImageEmbedder
from vector_store_multimodal import MultimodalVectorStore, TextEmbedder

def main():
    print("=" * 60)
    print("Caching Caption Embeddings for Faster Image Search")
    print("=" * 60)
    
    # Initialize text embedder (BGE-M3)
    print("\n[1/3] Loading BGE-M3 text embedder...")
    text_embedder = TextEmbedder()
    
    # Initialize vector store
    print("[2/3] Connecting to Qdrant...")
    vector_store = MultimodalVectorStore(
        text_embedder=text_embedder,
        qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
        qdrant_port=int(os.getenv("QDRANT_PORT", 6333))
    )
    
    # Get all images
    print("[3/3] Processing image captions...")
    all_images = vector_store.get_all_images(limit=100)
    
    if not all_images:
        print("No images found in database!")
        return
    
    print(f"Found {len(all_images)} images")
    
    # Compute embeddings for all captions
    captions = [doc.page_content for doc in all_images]
    print(f"Computing embeddings for {len(captions)} captions...")
    
    caption_embeddings = text_embedder.embed_texts(captions)
    
    # Update each image with its text embedding
    print("Updating image documents with cached embeddings...")
    
    from qdrant_client import QdrantClient
    
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333))
    )
    
    from vector_store_multimodal import IMAGE_COLLECTION
    
    # Scroll through all points and update with text_embedding
    results = client.scroll(
        collection_name=IMAGE_COLLECTION,
        limit=100,
        with_payload=True,
        with_vectors=False
    )
    
    points = results[0]
    print(f"Found {len(points)} points to update")
    
    from qdrant_client.models import PointIdsList
    
    for i, (point, embedding) in enumerate(zip(points, caption_embeddings)):
        # Update the point with text_embedding in payload
        client.set_payload(
            collection_name=IMAGE_COLLECTION,
            payload={
                "text_embedding": embedding
            },
            points=PointIdsList(
                points=[point.id]
            )
        )
        source = point.payload.get("source", "Unknown")
        print(f"  [{i+1}/{len(points)}] Updated {source}")
    
    print("\n" + "=" * 60)
    print("Caption embeddings cached successfully!")
    print("Image search will now be much faster.")
    print("=" * 60)

if __name__ == "__main__":
    main()
