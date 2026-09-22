"""
Multimodal Processor Module
Processes PDFs, images, and text files for multimodal RAG.

Supports:
- PDFs (text extraction + image extraction from pages)
- Images (direct embedding with BLIP captions)
- Text files (.txt, .md, etc.)
- Tables/Charts (extracted as images from PDFs)
"""

import hashlib
import base64
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import fitz  # PyMuPDF
from PIL import Image
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from image_embeddings import ImageEmbedder, ImageDocumentProcessor, extract_context_from_path


@dataclass
class ProcessedContent:
    """Container for processed multimodal content."""
    text_documents: List[Document]
    image_data: List[Dict]  # List of {document, embedding, base64, hash}
    source_file: str
    file_hash: str
    file_type: str


class MultimodalProcessor:
    """
    Multimodal document processor supporting PDFs, images, and text files.
    Extracts text and images from documents for multimodal RAG.
    """
    
    SUPPORTED_PDF_EXTENSIONS = {'.pdf'}
    SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
    SUPPORTED_TEXT_EXTENSIONS = {'.txt', '.md', '.csv', '.json'}
    
    def __init__(
        self,
        files_directory: str,
        image_embedder: ImageEmbedder = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.files_directory = Path(files_directory)
        self.image_embedder = image_embedder or ImageEmbedder()
        self.image_processor = ImageDocumentProcessor(self.image_embedder)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def get_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file for deduplication."""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _get_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Get configured text splitter."""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def _extract_text_from_pdf(self, pdf_path: Path) -> Tuple[str, List[Dict]]:
        """
        Extract text and images from PDF using PyMuPDF.
        
        Returns:
            Tuple of (full_text, list of image_info dicts)
        """
        text = ""
        images_data = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text
                page_text = page.get_text()
                text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                
                # Extract images from page
                image_list = page.get_images(full=True)
                
                for img_index, img_info in enumerate(image_list):
                    xref = img_info[0]
                    try:
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        # Skip very small images (likely icons/bullets)
                        if len(image_bytes) < 5000:
                            continue
                        
                        # Convert to PIL Image
                        pil_image = Image.open(BytesIO(image_bytes))
                        
                        # Skip tiny dimensions
                        if pil_image.width < 100 or pil_image.height < 100:
                            continue
                        
                        # Generate embedding
                        embedding = self.image_embedder.encode_image(pil_image)
                        base64_image = self.image_embedder.image_to_base64(pil_image)
                        image_hash = hashlib.md5(image_bytes).hexdigest()
                        
                        # Generate caption using BLIP
                        caption = self.image_embedder.generate_caption(pil_image)
                        
                        # Extract context from PDF path
                        context = extract_context_from_path(pdf_path)
                        company = context["company"]
                        
                        # Create enriched page content
                        page_content = f"{company} - Image from {pdf_path.stem}, page {page_num + 1}. Caption: {caption}"
                        
                        # Create metadata
                        img_metadata = {
                            "source": f"{pdf_path.name} - Page {page_num + 1} - Image {img_index + 1}",
                            "parent_pdf": pdf_path.name,
                            "page_number": page_num + 1,
                            "image_index": img_index + 1,
                            "file_type": "pdf_image",
                            "file_hash": image_hash,
                            "width": pil_image.width,
                            "height": pil_image.height,
                            "caption": caption,
                            "company": company
                        }
                        
                        # Create document with enriched content
                        img_doc = Document(
                            page_content=page_content,
                            metadata=img_metadata
                        )
                        
                        images_data.append({
                            "document": img_doc,
                            "embedding": embedding,
                            "base64_image": base64_image,
                            "hash": image_hash,
                            "path": str(pdf_path)
                        })
                        
                    except Exception as e:
                        print(f"  Warning: Could not extract image {img_index} from page {page_num + 1}: {e}")
            
            doc.close()
            
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
            return "", []
        
        return text.strip(), images_data
    
    def _process_pdf(self, pdf_path: Path, existing_hashes: Set[str]) -> Optional[ProcessedContent]:
        """Process a PDF file extracting text and images."""
        file_hash = self.get_file_hash(pdf_path)
        
        # Check if entire PDF was already processed
        if file_hash in existing_hashes:
            print(f"Skipping {pdf_path.name} (already processed)")
            return None
        
        print(f"Processing PDF: {pdf_path.name}")
        
        # Extract text and images
        text, images_data = self._extract_text_from_pdf(pdf_path)
        
        # Filter out already processed images
        new_images = []
        for img_data in images_data:
            if img_data["hash"] not in existing_hashes:
                new_images.append(img_data)
            else:
                print(f"  Skipping duplicate image in {pdf_path.name}")
        
        # Split text into chunks
        text_documents = []
        if text:
            doc_name = pdf_path.stem.replace("_", " ").replace("-", " ")
            enriched_text = f"Document: {doc_name} | Content:\n{text}"
            
            text_splitter = self._get_text_splitter()
            chunks = text_splitter.split_text(enriched_text)
            
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": pdf_path.name,
                        "full_path": str(pdf_path),
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "file_type": "pdf_text",
                        "file_hash": file_hash
                    }
                )
                text_documents.append(doc)
        
        print(f"  Extracted {len(text_documents)} text chunks")
        print(f"  Extracted {len(new_images)} images")
        
        return ProcessedContent(
            text_documents=text_documents,
            image_data=new_images,
            source_file=pdf_path.name,
            file_hash=file_hash,
            file_type="pdf"
        )
    
    def _process_image(self, image_path: Path, existing_hashes: Set[str]) -> Optional[ProcessedContent]:
        """Process a standalone image file."""
        image_hash = self.image_embedder.get_image_hash(image_path)
        
        if image_hash in existing_hashes:
            print(f"Skipping {image_path.name} (already processed)")
            return None
        
        print(f"Processing image: {image_path.name}")
        
        try:
            doc, embedding, base64_image = self.image_processor.process_image(image_path)
            
            return ProcessedContent(
                text_documents=[],
                image_data=[{
                    "document": doc,
                    "embedding": embedding,
                    "base64_image": base64_image,
                    "hash": image_hash,
                    "path": str(image_path)
                }],
                source_file=image_path.name,
                file_hash=image_hash,
                file_type="image"
            )
        except Exception as e:
            print(f"Error processing image {image_path.name}: {e}")
            return None
    
    def _process_text_file(self, text_path: Path, existing_hashes: Set[str]) -> Optional[ProcessedContent]:
        """Process a text file."""
        file_hash = self.get_file_hash(text_path)
        
        if file_hash in existing_hashes:
            print(f"Skipping {text_path.name} (already processed)")
            return None
        
        print(f"Processing text file: {text_path.name}")
        
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            doc_name = text_path.stem.replace("_", " ").replace("-", " ")
            enriched_text = f"Document: {doc_name} | Content:\n{text}"
            
            text_splitter = self._get_text_splitter()
            chunks = text_splitter.split_text(enriched_text)
            
            text_documents = []
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": text_path.name,
                        "full_path": str(text_path),
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "file_type": "text",
                        "file_hash": file_hash
                    }
                )
                text_documents.append(doc)
            
            print(f"  Extracted {len(text_documents)} text chunks")
            
            return ProcessedContent(
                text_documents=text_documents,
                image_data=[],
                source_file=text_path.name,
                file_hash=file_hash,
                file_type="text"
            )
        except Exception as e:
            print(f"Error processing text file {text_path.name}: {e}")
            return None
    
    def process_all_files(
        self,
        existing_hashes: Set[str] = None,
        recursive: bool = True
    ) -> Tuple[List[Document], List[Dict], Dict[str, str]]:
        """
        Process all supported files in the directory.
        
        Args:
            existing_hashes: Set of already processed file/image hashes
            recursive: Whether to search subdirectories
            
        Returns:
            Tuple of (text_documents, image_data_list, file_hashes_dict)
        """
        if existing_hashes is None:
            existing_hashes = set()
        
        all_text_docs = []
        all_image_data = []
        file_hashes = {}
        
        # Get all files
        if recursive:
            all_files = list(self.files_directory.rglob("*"))
        else:
            all_files = list(self.files_directory.glob("*"))
        
        # Filter to supported file types
        for file_path in all_files:
            if not file_path.is_file():
                continue
            
            suffix = file_path.suffix.lower()
            
            if suffix in self.SUPPORTED_PDF_EXTENSIONS:
                result = self._process_pdf(file_path, existing_hashes)
                if result:
                    all_text_docs.extend(result.text_documents)
                    all_image_data.extend(result.image_data)
                    file_hashes[file_path.name] = result.file_hash
            
            elif suffix in self.SUPPORTED_IMAGE_EXTENSIONS:
                result = self._process_image(file_path, existing_hashes)
                if result:
                    all_image_data.extend(result.image_data)
                    file_hashes[file_path.name] = result.file_hash
            
            elif suffix in self.SUPPORTED_TEXT_EXTENSIONS:
                result = self._process_text_file(file_path, existing_hashes)
                if result:
                    all_text_docs.extend(result.text_documents)
                    file_hashes[file_path.name] = result.file_hash
        
        print(f"\n{'='*50}")
        print(f"Processing Summary:")
        print(f"  Text documents: {len(all_text_docs)}")
        print(f"  Images processed: {len(all_image_data)}")
        print(f"  Files processed: {len(file_hashes)}")
        print(f"{'='*50}")
        
        return all_text_docs, all_image_data, file_hashes
