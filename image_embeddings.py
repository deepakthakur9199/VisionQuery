"""
Image Embeddings Module using SigLIP + BLIP
- SigLIP for image embeddings (768 dims)
- BLIP for image captioning (text descriptions)
"""

import base64
import hashlib
import re
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, BlipProcessor, BlipForConditionalGeneration
from langchain_core.documents import Document


class ImageEmbedder:
    """
    SigLIP 2 image embedder + BLIP captioner for multimodal RAG.
    Uses SigLIP 2 for embeddings (better multilingual support) and BLIP for captions.
    """
    
    SIGLIP_MODEL = "google/siglip2-so400m-patch14-384"  # SigLIP 2 - better multilingual
    BLIP_MODEL = "Salesforce/blip-image-captioning-base"
    EMBEDDING_DIM = 1152  # SigLIP 2 embedding dimension
    
    def __init__(self, device: str = None):
        """
        Initialize SigLIP and BLIP models.
        
        Args:
            device: Device to run models on. Auto-detects if None.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        print(f"Loading SigLIP model on {device}...")
        
        # Load SigLIP model and processor
        self.processor = AutoProcessor.from_pretrained(self.SIGLIP_MODEL)
        self.model = AutoModel.from_pretrained(self.SIGLIP_MODEL).to(device)
        self.model.eval()
        
        # Use half precision on GPU for faster inference
        if device == "cuda":
            self.model = self.model.half()
        
        print(f"SigLIP model loaded successfully (embedding dim: {self.EMBEDDING_DIM})")
        
        # Load BLIP for captioning
        print(f"Loading BLIP captioning model on {device}...")
        self.blip_processor = BlipProcessor.from_pretrained(self.BLIP_MODEL)
        self.blip_model = BlipForConditionalGeneration.from_pretrained(self.BLIP_MODEL).to(device)
        self.blip_model.eval()
        
        # Use half precision on GPU for faster inference
        if device == "cuda":
            self.blip_model = self.blip_model.half()
        
        print(f"BLIP model loaded successfully")
    
    def _load_image(self, image_source: Union[str, Path, Image.Image, bytes]) -> Image.Image:
        """Load image from various sources."""
        if isinstance(image_source, Image.Image):
            return image_source.convert("RGB")
        elif isinstance(image_source, bytes):
            return Image.open(BytesIO(image_source)).convert("RGB")
        elif isinstance(image_source, (str, Path)):
            path = Path(image_source)
            if path.exists():
                return Image.open(path).convert("RGB")
            else:
                # Assume base64 encoded
                image_data = base64.b64decode(image_source)
                return Image.open(BytesIO(image_data)).convert("RGB")
        else:
            raise ValueError(f"Unsupported image source type: {type(image_source)}")
    
    def encode_image(self, image_source: Union[str, Path, Image.Image, bytes]) -> List[float]:
        """
        Generate embedding for a single image.
        
        Args:
            image_source: Image path, PIL Image, bytes, or base64 string
            
        Returns:
            List of floats representing the image embedding
        """
        image = self._load_image(image_source)
        
        with torch.no_grad():
            # Process image only (no text)
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            # Use model forward pass to get vision model output
            vision_outputs = self.model.vision_model(pixel_values=inputs['pixel_values'])
            # Get the pooled output and project it
            pooled_output = vision_outputs.pooler_output
            # Apply projection layer if available
            if hasattr(self.model, 'visual_projection'):
                embedding = self.model.visual_projection(pooled_output)
            else:
                embedding = pooled_output
            # Normalize embedding
            embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
            return embedding.cpu().numpy().flatten().tolist()
    
    def encode_images_batch(
        self, 
        image_sources: List[Union[str, Path, Image.Image, bytes]],
        batch_size: int = 8
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple images in batches.
        
        Args:
            image_sources: List of image sources
            batch_size: Batch size for processing
            
        Returns:
            List of embeddings
        """
        all_embeddings = []
        
        for i in range(0, len(image_sources), batch_size):
            batch = image_sources[i:i + batch_size]
            images = [self._load_image(src) for src in batch]
            
            with torch.no_grad():
                # Process images only (no text)
                inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
                # Use model forward pass to get vision model output
                vision_outputs = self.model.vision_model(pixel_values=inputs['pixel_values'])
                # Get the pooled output and project it
                pooled_output = vision_outputs.pooler_output
                # Apply projection layer if available
                if hasattr(self.model, 'visual_projection'):
                    embeddings = self.model.visual_projection(pooled_output)
                else:
                    embeddings = pooled_output
                # Normalize embeddings
                embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
                all_embeddings.extend(embeddings.cpu().numpy().tolist())
        
        return all_embeddings
    
    def encode_text(self, text: str) -> List[float]:
        """
        Generate embedding for a text query (for image search).
        
        Args:
            text: Text query
            
        Returns:
            List of floats representing the text embedding
        """
        with torch.no_grad():
            # Process text only
            inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
            # Use model forward pass to get text model output
            text_outputs = self.model.text_model(input_ids=inputs['input_ids'])
            # Get the pooled output and project it
            pooled_output = text_outputs.pooler_output
            # Apply projection layer if available
            if hasattr(self.model, 'text_projection'):
                embedding = self.model.text_projection(pooled_output)
            else:
                embedding = pooled_output
            # Normalize embedding
            embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
            return embedding.cpu().numpy().flatten().tolist()
    
    def image_to_base64(self, image_source: Union[str, Path, Image.Image, bytes]) -> str:
        """Convert image to base64 string for storage."""
        image = self._load_image(image_source)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def get_image_hash(self, image_source: Union[str, Path, Image.Image, bytes]) -> str:
        """Generate MD5 hash for image deduplication."""
        if isinstance(image_source, (str, Path)):
            path = Path(image_source)
            if path.exists():
                hasher = hashlib.md5()
                with open(path, 'rb') as f:
                    for chunk in iter(lambda: f.read(8192), b''):
                        hasher.update(chunk)
                return hasher.hexdigest()
        
        # For other sources, convert to bytes and hash
        image = self._load_image(image_source)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        return hashlib.md5(buffer.getvalue()).hexdigest()
    
    def generate_caption(self, image_source: Union[str, Path, Image.Image, bytes]) -> str:
        """
        Generate a caption for an image using BLIP.
        
        Args:
            image_source: Image path, PIL Image, bytes, or base64 string
            
        Returns:
            Caption string describing the image
        """
        image = self._load_image(image_source)
        
        with torch.no_grad():
            inputs = self.blip_processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.blip_model.generate(**inputs, max_length=50)
            caption = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
        
        return caption
    
    def generate_caption_batch(self, images: List[Image.Image]) -> List[str]:
        """Generate captions for multiple images."""
        captions = []
        for image in images:
            caption = self.generate_caption(image)
            captions.append(caption)
        return captions


def extract_context_from_path(file_path: Path) -> Dict[str, str]:
    """
    Extract company/product context from file path.
    
    Example paths:
    - filesRAG/brandstore/strawberry.jpg -> company: brandstore, product: strawberry
    - filesRAG/fresh-green-brand/hass_avocado.jpg -> company: fresh-green-brand, product: hass avocado
    
    Returns:
        Dict with 'company', 'product', 'category' keys
    """
    parts = file_path.parts
    
    # Find the index of 'filesRAG' in the path
    try:
        files_idx = parts.index('filesRAG')
        # The folder after filesRAG is the company/folder name
        if len(parts) > files_idx + 1:
            company_folder = parts[files_idx + 1]
        else:
            company_folder = "unknown"
    except ValueError:
        company_folder = "unknown"
    
    # Extract product name from filename (without extension)
    product_name = file_path.stem
    # Clean up product name (replace underscores, dashes with spaces)
    product_name = product_name.replace('_', ' ').replace('-', ' ')
    
    # Try to detect category based on common patterns
    category = "general"
    product_lower = product_name.lower()
    
    # Product categories
    category_keywords = {
        'fruit': ['strawberry', 'mango', 'pineapple', 'lime', 'lemon', 'lulo', 'avocado', 'fruit'],
        'product': ['wallet', 'bottle', 'cup', 'pencil', 'tumbler', 'refill', 'scissors', 'paper', 'books', 'cardholder'],
        'catalog': ['catalog', 'overview', 'earnings', 'financial', 'profits', 'summary', 'report']
    }
    
    for cat, keywords in category_keywords.items():
        if any(kw in product_lower for kw in keywords):
            category = cat
            break
    
    return {
        "company": company_folder,
        "product": product_name,
        "category": category
    }


class ImageDocumentProcessor:
    """
    Process images and create Document objects with embeddings.
    """
    
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
    
    def __init__(self, embedder: ImageEmbedder = None):
        self.embedder = embedder or ImageEmbedder()
    
    def is_image_file(self, file_path: Path) -> bool:
        """Check if file is a supported image format."""
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS
    
    def process_image(
        self, 
        image_path: Path, 
        metadata: Dict = None
    ) -> Tuple[Document, List[float], str]:
        """
        Process a single image file with BLIP captioning and context extraction.
        
        Returns:
            Tuple of (Document, embedding, base64_image)
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Generate embedding
        embedding = self.embedder.encode_image(image_path)
        
        # Convert to base64
        base64_image = self.embedder.image_to_base64(image_path)
        
        # Generate hash
        image_hash = self.embedder.get_image_hash(image_path)
        
        # Generate caption using BLIP
        caption = self.embedder.generate_caption(image_path)
        
        # Extract context from path (company, product, category)
        context = extract_context_from_path(image_path)
        
        # Create enriched page content with caption and context
        # Format: "company - product. Image: caption"
        company = context["company"]
        product = context["product"]
        category = context["category"]
        
        # Build descriptive content
        if company != "unknown":
            page_content = f"{company} - {product}. Image: {caption}"
        else:
            page_content = f"{product}. Image: {caption}"
        
        # Create document with enriched metadata
        doc_metadata = {
            "source": str(image_path.name),
            "full_path": str(image_path),
            "file_type": "image",
            "file_hash": image_hash,
            "extension": image_path.suffix.lower(),
            "caption": caption,
            "company": company,
            "product": product,
            "category": category,
            **(metadata or {})
        }
        
        # Create document with enriched content
        doc = Document(
            page_content=page_content,
            metadata=doc_metadata
        )
        
        return doc, embedding, base64_image
    
    def process_images_in_directory(
        self,
        directory: Path,
        existing_hashes: set = None,
        recursive: bool = True
    ) -> Tuple[List[Dict], List[str]]:
        """
        Process all images in a directory.
        
        Args:
            directory: Directory to search for images
            existing_hashes: Set of already processed image hashes
            recursive: Whether to search recursively
            
        Returns:
            Tuple of (list of processed image data, list of skipped files)
        """
        if existing_hashes is None:
            existing_hashes = set()
        
        processed = []
        skipped = []
        
        # Find all image files
        if recursive:
            image_files = []
            for ext in self.SUPPORTED_EXTENSIONS:
                image_files.extend(directory.rglob(f"*{ext}"))
        else:
            image_files = []
            for ext in self.SUPPORTED_EXTENSIONS:
                image_files.extend(directory.glob(f"*{ext}"))
        
        for image_path in image_files:
            image_hash = self.embedder.get_image_hash(image_path)
            
            if image_hash in existing_hashes:
                skipped.append(str(image_path.name))
                print(f"Skipping {image_path.name} (already processed)")
                continue
            
            try:
                doc, embedding, base64_image = self.process_image(image_path)
                processed.append({
                    "document": doc,
                    "embedding": embedding,
                    "base64_image": base64_image,
                    "hash": image_hash,
                    "path": str(image_path)
                })
                print(f"Processed image: {image_path.name}")
            except Exception as e:
                print(f"Error processing {image_path.name}: {e}")
        
        return processed, skipped
