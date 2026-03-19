from flask import Flask, render_template, jsonify, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

embeddings = download_embeddings()

index_name = 'medical-chatbot'

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

RETRIEVAL_K = 3
SCORE_THRESHOLD = 0.2

chatModel = ChatOpenAI(model='gpt-5-nano')
prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system_prompt),
        ('human', '{input}')
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)



@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get', methods=['GET', 'POST'])
def chat():
    msg = request.form['msg']
    input = msg
    print(input)

    results = docsearch.similarity_search_with_score(msg, k=RETRIEVAL_K)
    filtered = [(doc, score) for doc, score in results if score >= SCORE_THRESHOLD]

    if not filtered:
        return "I don't know."

    docs = [doc for doc, _score in filtered]
    response = question_answer_chain.invoke({'input': msg, 'context': docs})
    raw_answer = response.get('answer') if isinstance(response, dict) else response

    lines = [line.lstrip("- ").strip() for line in raw_answer.splitlines() if line.strip()]
    clear_answer = '\n'.join(lines)

    sources = []
    source_tags = []
    for doc in docs:
        if not doc.metadata:
            continue
        src = doc.metadata.get('source')
        page = doc.metadata.get('page')
        if src:
            base = os.path.basename(src)
            name = os.path.splitext(base)[0]
            normalized = "".join(ch if ch.isalnum() else " " for ch in name).strip()
            short = normalized.split()[0] if normalized else "Source"
            source_tags.append(f"[{short}]")
        if page is not None:
            page_num = int(page) + 1
            sources.append(f"[p.{page_num}]")
    if sources:
        unique_sources = sorted(set(sources), key=lambda s: int(s.strip("[]").split(".")[1]))
        unique_tags = sorted(set(source_tags))
        citation = "".join(unique_tags) + " " + "".join(unique_sources) if unique_tags else "".join(unique_sources)
        clear_answer = f"{clear_answer} " + f"<span class=\"citations\">{citation}</span>"

    print('Response:', clear_answer)
    return clear_answer

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
