"""
Multimodal RAG Chat Application with FastAPI, WebSocket, Gemini, and Qdrant.
Supports PDFs, Images, and Text files with hybrid retrieval.
"""

import os
import sys
import asyncio
import json
import base64
from pathlib import Path
from typing import Optional, Set
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import multimodal modules
from multimodal_processor import MultimodalProcessor
from vector_store_multimodal import MultimodalVectorStore, TextEmbedder
from image_embeddings import ImageEmbedder
from hybrid_retriever import HybridRetriever
from rag_chain_multimodal import MultimodalRAGChain

# Set Windows event loop policy
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# Configuration
FILES_DIRECTORY = Path(__file__).parent / "filesRAG"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# Initialize FastAPI app
app = FastAPI(
    title="Multimodal RAG Chat Assistant",
    description="AI Chat with PDF, Image, and Text retrieval using Gemini, BGE-M3, SigLIP, and Qdrant",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global instances
vector_store: Optional[MultimodalVectorStore] = None
image_embedder: Optional[ImageEmbedder] = None
hybrid_retriever: Optional[HybridRetriever] = None
rag_chain: Optional[MultimodalRAGChain] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the multimodal RAG system on startup."""
    global vector_store, image_embedder, hybrid_retriever, rag_chain
    
    print("\n" + "="*60)
    print("Initializing Multimodal RAG System...")
    print("="*60)
    
    # Initialize image embedder (SigLIP)
    print("\n[1/5] Loading SigLIP image embedder...")
    image_embedder = ImageEmbedder()
    
    # Initialize text embedder and vector store
    print("\n[2/5] Loading BGE-M3 text embedder and connecting to Qdrant...")
    text_embedder = TextEmbedder()
    vector_store = MultimodalVectorStore(
        qdrant_host=QDRANT_HOST,
        qdrant_port=QDRANT_PORT,
        text_embedder=text_embedder
    )
    
    # Check Qdrant connection
    try:
        info = vector_store.get_collection_info()
        print(f"Qdrant connection: OK")
        for col_name, col_info in info.items():
            if col_info["exists"]:
                print(f"  Collection '{col_name}': {col_info['points_count']} documents")
            else:
                print(f"  Collection '{col_name}': Not found (will be created)")
    except Exception as e:
        print(f"ERROR: Cannot connect to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        print(f"Make sure Qdrant is running: docker run -d -p 6333:6333 qdrant/qdrant")
        raise e
    
    # Initialize hybrid retriever
    print("\n[3/5] Initializing Hybrid Retriever (RRF)...")
    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        image_embedder=image_embedder
    )
    
    # Initialize RAG chain
    print("\n[4/5] Initializing Multimodal RAG Chain with Gemini...")
    rag_chain = MultimodalRAGChain(
        vector_store=vector_store,
        hybrid_retriever=hybrid_retriever,
        google_api_key=GOOGLE_API_KEY
    )
    
    # Auto-process files on startup
    print("\n[5/5] Processing files in filesRAG directory...")
    print("-"*60)
    
    try:
        # Get existing hashes from both collections
        existing_hashes = vector_store.get_all_hashes()
        all_existing_hashes = existing_hashes["text"].union(existing_hashes["image"])
        
        # Initialize multimodal processor
        processor = MultimodalProcessor(
            files_directory=str(FILES_DIRECTORY),
            image_embedder=image_embedder
        )
        
        # Process all files
        text_docs, image_data, file_hashes = processor.process_all_files(all_existing_hashes)
        
        # Add to vector store
        if text_docs:
            vector_store.add_text_documents(text_docs)
        
        if image_data:
            vector_store.add_image_embeddings(image_data)
        
        print(f"\nFiles processed: {len(file_hashes)}")
        print(f"Text chunks added: {len(text_docs)}")
        print(f"Images added: {len(image_data)}")
        
    except Exception as e:
        print(f"Warning: Could not process files: {e}")
        import traceback
        traceback.print_exc()
    
    # Show final status
    info = vector_store.get_collection_info()
    print("\n" + "="*60)
    print("Multimodal RAG System Ready!")
    print(f"  Text documents: {info.get('text_chunks', {}).get('points_count', 0)}")
    print(f"  Image embeddings: {info.get('image_embeddings', {}).get('points_count', 0)}")
    print("="*60 + "\n")


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Multimodal RAG Chat Assistant API",
        "version": "2.0.0",
        "features": ["PDF", "Images", "Text", "Hybrid Retrieval (RRF)"],
        "models": {
            "text_embeddings": "BGE-M3 (1024 dims)",
            "image_embeddings": "SigLIP (768 dims)",
            "llm": "Gemini 2.0 Flash"
        },
        "endpoints": {
            "websocket": "/ws/chat",
            "process_files": "/api/process-files",
            "status": "/api/status"
        }
    }


@app.get("/api/status")
async def get_status():
    """Get system status."""
    try:
        info = vector_store.get_collection_info()
        
        # Count files by type
        pdf_count = 0
        image_count = 0
        text_count = 0
        
        for folder in FILES_DIRECTORY.iterdir():
            if folder.is_dir():
                for f in folder.rglob("*"):
                    if f.is_file():
                        suffix = f.suffix.lower()
                        if suffix == ".pdf":
                            pdf_count += 1
                        elif suffix in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}:
                            image_count += 1
                        elif suffix in {'.txt', '.md', '.csv', '.json'}:
                            text_count += 1
        
        return {
            "status": "operational",
            "qdrant": {
                "connected": True,
                "text_collection": info.get("text_chunks", {}),
                "image_collection": info.get("image_embeddings", {})
            },
            "files_directory": str(FILES_DIRECTORY),
            "file_counts": {
                "pdfs": pdf_count,
                "images": image_count,
                "text_files": text_count
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@app.post("/api/process-files")
async def process_files():
    """Process all files in the filesRAG directory."""
    try:
        # Get existing hashes
        existing_hashes = vector_store.get_all_hashes()
        all_existing_hashes = existing_hashes["text"].union(existing_hashes["image"])
        
        # Initialize processor
        processor = MultimodalProcessor(
            files_directory=str(FILES_DIRECTORY),
            image_embedder=image_embedder
        )
        
        # Process all files
        text_docs, image_data, file_hashes = processor.process_all_files(all_existing_hashes)
        
        # Add to vector store
        text_count = 0
        image_count = 0
        
        if text_docs:
            text_count = vector_store.add_text_documents(text_docs)
        
        if image_data:
            image_count = vector_store.add_image_embeddings(image_data)
        
        return {
            "status": "success",
            "message": f"Processed {len(file_hashes)} files",
            "text_chunks_added": text_count,
            "images_added": image_count,
            "files_processed": list(file_hashes.keys())
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/clear-database")
async def clear_database():
    """Clear the vector database (use with caution)."""
    try:
        vector_store.clear_collection()
        return {"status": "success", "message": "Database cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for multimodal RAG chat."""
    await websocket.accept()
    print("WebSocket client connected")
    
    def user_requests_image(query: str) -> bool:
        """Detect if user is explicitly requesting to see an image."""
        image_keywords = [
            "image", "images", "photo", "photos", "photography",
            "picture", "pictures", "snapshot", "illustration",
            "show me", "show", "display", "view", "see",
            "give me the image", "give me the photo", "give me the picture",
            "want to see", "need to see", "can i see"
        ]
        query_lower = query.lower()
        return any(kw in query_lower for kw in image_keywords)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                message_data = json.loads(data)
                query = message_data.get("message", "")
                
                if not query:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Empty message"
                    })
                    continue
                
                # Send user message confirmation
                await websocket.send_json({
                    "type": "user_message",
                    "message": query
                })
                
                # Check if user explicitly requests images
                should_show_images = user_requests_image(query)
                
                # Retrieve context with scores for better image filtering
                # Use very low threshold to get all relevant images
                text_docs, image_docs, image_scores = rag_chain.retrieve_context_with_scores(
                    query, k=5, image_score_threshold=-1.0  # Include all retrieved images
                )
                
                # Debug: log image scores
                print(f"\n=== HYBRID Image retrieval for query: {query} ===")
                for doc, score in image_scores:
                    print(f"  Image: {doc.metadata.get('source', 'Unknown')} - Score: {score:.4f}")
                print(f"Total images retrieved: {len(image_docs)}")
                
                # Send sources (text only, indicate if images available)
                if text_docs or image_docs:
                    # Get all images for display (no score filtering)
                    context_info = rag_chain.get_context_for_display(
                        text_docs, image_docs, min_image_score=-1.0  # Show all images
                    )
                    
                    await websocket.send_json({
                        "type": "sources",
                        "text_sources": context_info["text_sources"],
                        "image_sources": [
                            {
                                "source": img["source"],
                                "has_image": bool(img.get("base64")),
                                "relevance_score": img.get("relevance_score", 0)
                            }
                            for img in context_info["image_sources"][:3]
                        ],
                        "images_available": len(image_docs) > 0,
                        "show_images_hint": "Mention 'image', 'photo', or 'picture' in your question if you want to view related images."
                    })
                    
                    # Stream response
                    await websocket.send_json({
                        "type": "stream_start",
                        "message": "Generating response..."
                    })
                    
                    # Stream the response
                    async for chunk in rag_chain.astream(query, text_docs, image_docs):
                        await websocket.send_json({
                            "type": "stream_chunk",
                            "chunk": chunk
                        })
                    
                    # Send image AFTER streaming if user requested it
                    # This ensures the image is associated with the response
                    if should_show_images and context_info["image_sources"]:
                        best_image = context_info["image_sources"][0]
                        print(f"[DEBUG] should_show_images=True, best_image source: {best_image.get('source')}")
                        print(f"[DEBUG] best_image has base64: {bool(best_image.get('base64'))}")
                        if best_image.get("base64"):
                            print(f"[DEBUG] Sending image message to frontend...")
                            await websocket.send_json({
                                "type": "image",
                                "source": best_image["source"],
                                "base64": best_image["base64"],
                                "relevance_score": best_image.get("relevance_score", 0)
                            })
                            print(f"[DEBUG] Image message sent!")
                    
                    # Send completion message AFTER image
                    await websocket.send_json({
                        "type": "stream_complete",
                        "message": "Response complete"
                    })
                
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format"
                })
            except Exception as e:
                print(f"Error processing message: {e}")
                import traceback
                traceback.print_exc()
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error: {str(e)}"
                })
    
    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")


if __name__ == "__main__":
    import uvicorn
    print("\nStarting Multimodal RAG Chat Server...")
    print(f"Files Directory: {FILES_DIRECTORY}")
    print(f"Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
    print(f"WebSocket: ws://localhost:8000/ws/chat")
    print(f"API Docs: http://localhost:8000/docs\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
