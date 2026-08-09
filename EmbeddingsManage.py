from typing import List

from langchain_core import documents
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.core.base.embeddings.base import similarity
from llama_index.readers.file import PyMuPDFReader
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import SimpleDirectoryReader,VectorStoreIndex
from pathlib import Path
import chromadb
from workflows.runtime.types import results

FOLDER_PATH = Path("pdfs")

embedding_model = OllamaEmbedding(
    model_name="bge-m3",
    base_url="http://localhost:11434/",
    api_key="ollama",
)

#abrindo ou criando o vector store
chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="my_pdfs"
)

#conectando o ChromaDB ao LlamaIndex
vector_store = ChromaVectorStore(
    chroma_collection=collection,
)

def get_data() -> List[Document]:
    documents = SimpleDirectoryReader(
        input_dir=str(FOLDER_PATH),
        required_exts=[".pdf"],
        recursive=True,
        file_extractor={
            ".pdf": PyMuPDFReader(),
        }
    ).load_data()

    return documents

def index_pdf():
    #faz o LlamaIndex entender o pdf
    documents = get_data();

    for document in documents:
        path = document.metadata.get("file_path", "")
        if path:
            document.metadata["file"] = Path(path).name

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(
                chunk_size=600,
                chunk_overlap=100
            ),
            embedding_model
        ],
        vector_store=vector_store,
    )

    pipeline.run(
        documents=documents,
        show_progress=True
    )

def create_retriever():
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embedding_model
    )

    return index.as_retriever(
        similarity_top_k=5
    )

def search_best_chunk_for_context(question: str) -> List[str]:
    retriever = create_retriever()

    results = retriever.retrieve(question)

    contents: List[str] = []
    for result in results:
        contents.append(
            result.node.get_content()
        )

    return contents



