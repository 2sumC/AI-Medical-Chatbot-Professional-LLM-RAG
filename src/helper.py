from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document

# 1 extract text from pdf files
def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob='**/*.pdf',
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    return documents

# 2 filter unnecessary data
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    '''
    filter the necessary data:
    keep source and page in metadata and the original page_content.
    '''
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get('source')
        page = doc.metadata.get('page')
        minimal_docs.append(
            Document(
                page_content = doc.page_content,
                metadata = {'source':src, 'page': page}
            )
        )
    return minimal_docs

# 3 seperate the documents into smaller chunks
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 20
    )
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk

# 4 download embedding from HuggingFace
def download_embeddings():
    '''
    Download and return the HuggingFace embeddings model.
    '''
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    embeddings = HuggingFaceEmbeddings(
        model_name = model_name
    )
    return embeddings
