"""
Script to rebuild the image collection with new SigLIP 2 embeddings.
Run this after updating to SigLIP 2 (1152 dims instead of 768).
"""

import sys
import asyncio
from pathlib import Path

# Set Windows event loop policy
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from image_embeddings import ImageEmbedder, ImageDocumentProcessor
from vector_store_multimodal import IMAGE_COLLECTION, SIGLIP_DIM
from dotenv import load_dotenv

load_dotenv()

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
FILES_DIRECTORY = Path(__file__).parent / "filesRAG"


def rebuild_image_collection():
    """Delete and recreate the image collection with new embedding dimension."""
    
    print("\n" + "="*60)
    print("Rebuilding Image Collection with SigLIP 2")
    print("="*60)
    
    # Connect to Qdrant
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    # Delete existing image collection
    print(f"\n[1/3] Deleting existing '{IMAGE_COLLECTION}' collection...")
    try:
        client.delete_collection(IMAGE_COLLECTION)
        print(f"  ✓ Collection '{IMAGE_COLLECTION}' deleted")
    except Exception as e:
        print(f"  ! Collection doesn't exist or already deleted: {e}")
    
    # Recreate with new dimension
    print(f"\n[2/3] Creating new '{IMAGE_COLLECTION}' collection with {SIGLIP_DIM} dimensions...")
    client.create_collection(
        collection_name=IMAGE_COLLECTION,
        vectors_config=VectorParams(
            size=SIGLIP_DIM,
            distance=Distance.COSINE
        )
    )
    print(f"  ✓ Collection created with {SIGLIP_DIM} dimensions")
    
    # Re-process all images
    print(f"\n[3/3] Re-processing all images with SigLIP 2...")
    
    # Initialize embedder
    embedder = ImageEmbedder()
    processor = ImageDocumentProcessor(embedder)
    
    # Process all images
    image_data, skipped = processor.process_images_in_directory(
        FILES_DIRECTORY,
        existing_hashes=set(),  # Process all images
        recursive=True
    )
    
    if image_data:
        # Add to Qdrant
        from vector_store_multimodal import MultimodalVectorStore, TextEmbedder
        
        text_embedder = TextEmbedder()
        vector_store = MultimodalVectorStore(
            qdrant_host=QDRANT_HOST,
            qdrant_port=QDRANT_PORT,
            text_embedder=text_embedder
        )
        
        count = vector_store.add_image_embeddings(image_data)
        print(f"\n  ✓ Added {count} images to collection")
    
    print(f"\n  Skipped: {len(skipped)} files")
    
    # Show final status
    info = client.get_collection(IMAGE_COLLECTION)
    print("\n" + "="*60)
    print("Image Collection Rebuilt Successfully!")
    print(f"  Total images: {info.points_count}")
    print(f"  Embedding dimension: {SIGLIP_DIM}")
    print("="*60 + "\n")


if __name__ == "__main__":
    rebuild_image_collection()
