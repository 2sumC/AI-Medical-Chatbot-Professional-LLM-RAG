from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

print("Loading PDF files from ./data ...")
extracted_data = load_pdf_files(data='data/')
print(f"Loaded {len(extracted_data)} pages/documents.")

print("Filtering metadata ...")
filter_data = filter_to_minimal_docs(extracted_data)

print("Splitting into chunks ...")
text_chunks = text_split(filter_data)
print(f"Generated {len(text_chunks)} text chunks.")

print("Loading embeddings model ...")
embeddings = download_embeddings()
print("Embeddings model ready.")

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)

index_name = 'medical-chatbot'

print("Checking Pinecone index ...")
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name = index_name,
        dimension = 768,
        metric = 'cosine',
        spec = ServerlessSpec(cloud='aws', region='us-east-1')
    )
    print(f"Created index: {index_name}")
else:
    print(f"Using existing index: {index_name}")
index = pc.Index(index_name)


print("Upserting embeddings to Pinecone (this can take a while) ...")
docsearch = PineconeVectorStore.from_documents(
    documents = text_chunks,
    embedding = embeddings,
    index_name = index_name,
    batch_size = 10,
    embeddings_chunk_size = 50
)
print("Indexing complete.")

# How to add more content to existing Pinecone Index
# newDoc = Document(
#     page_content = 'This is a new content',
#     metadata = {'source':'Youtube'}
# )
# docsearch.add_documents(documents=[newDoc])
