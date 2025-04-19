import os
import logging
from typing import List, Union
from tempfile import NamedTemporaryFile
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentManager:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = None
        self.retriever = None

    def load_document(self, source: Union[str, bytes, Document]) -> List[Document]:
        if isinstance(source, str):
            if os.path.exists(source):
                return self._load_file(source)
            else:
                return self._load_text(source)
        elif isinstance(source, bytes):
            return self._load_binary(source)
        elif isinstance(source, Document):
            return [source]
        else:
            raise ValueError("Unsupported source type")

    def _load_file(self, file_path: str) -> List[Document]:
        try:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
            elif file_path.endswith('.txt'):
                loader = TextLoader(file_path)
            else:
                raise ValueError("Unsupported file format")
            return loader.load()
        except Exception as e:
            logger.error(f"Error loading file: {e}")
            return []

    def _load_text(self, text: str) -> List[Document]:
        return [Document(page_content=text)]

    def _load_binary(self, file_content: bytes) -> List[Document]:
        with NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        
        try:
            return self._load_file(tmp_path)
        finally:
            os.unlink(tmp_path)

    def process_documents(self, documents: List[Document]):
        if documents:
            split_docs = self.text_splitter.split_documents(documents)
            self.vector_store = FAISS.from_documents(split_docs, self.embeddings)
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
            logger.info("Documents processed successfully")