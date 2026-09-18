import os

import cohere
from langchain_cohere import ChatCohere
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")
loader = DirectoryLoader(
    DOCUMENTS_DIR,
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)

documents = loader.load()

print(f"Loaded {len(documents)} SBA documents.")

source_info = {
    "SBA_Plan_Your_Business.txt": {
        "title": "Plan Your Business",
        "url": "https://www.sba.gov/counseling/plan-your-business/"
    },
    "SBA_Launch_Your_Business.txt": {
        "title": "Launch Your Business",
        "url": "https://www.sba.gov/counseling/launch-your-business/"
    },
    "SBA_Manage_Your_Business.txt": {
        "title": "Manage Your Business",
        "url": "https://www.sba.gov/counseling/manage-your-business/"
    },
    "SBA_Grow_Your_Business.txt": {
        "title": "Grow Your Business",
        "url": "https://www.sba.gov/counseling/grow-your-business/"
    }
}

for document in documents:
    filename = os.path.basename(document.metadata["source"])

    if filename in source_info:
        document.metadata["title"] = source_info[filename]["title"]
        document.metadata["url"] = source_info[filename]["url"]
        document.metadata["organization"] = "U.S. Small Business Administration"

print("Source metadata added.")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

texts = text_splitter.split_documents(documents)

print(f"Documents split into {len(texts)} chunks.")

class CohereV2Embeddings(Embeddings):
    def __init__(self):
        self.client = cohere.ClientV2(
            api_key=os.getenv("COHERE_API_KEY")
        )

    def embed_query(self, text):
        response = self.client.embed(
            model="embed-english-v3.0",
            texts=[text],
            input_type="search_query",
            embedding_types=["float"]
        )
        return response.embeddings.float_[0]

    def embed_documents(self, texts):
        response = self.client.embed(
            model="embed-english-v3.0",
            texts=texts,
            input_type="search_document",
            embedding_types=["float"]
        )
        return response.embeddings.float_


embeddings = CohereV2Embeddings()

CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

docsearch = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

print("Small Business Chroma database loaded.")

llm = ChatCohere(
    model="command-a-03-2025",
    cohere_api_key=os.getenv("COHERE_API_KEY")
)

print("Cohere chat model ready.")

prompt_template = """
Answer the question using the knowledge base as your primary source.

Base most of your answer on the provided context.
You may add a small amount of general knowledge to make the answer more complete,
but clearly prioritize information from the knowledge base.

Do not invent facts that conflict with the context.

If the knowledge base contains little or no relevant information, say so,
then provide a brief general answer if helpful.

Context:
{context}

Question:
{question}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=docsearch.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

print("RetrievalQA ready.")

chatbot_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are ThinkBot, a helpful, persuasive, and knowledgeable assistant.

Answer the user's question clearly and conversationally.
Provide useful detail without making the response unnecessarily complicated.

Question:
{question}

Answer:
"""
)
chatbot_chain = chatbot_prompt | llm

print("Chatbot ready.")

def search_knowledge_base(query, k=3):
    results = docsearch.similarity_search(query, k=k)

    unique_results = []
    seen_content = set()

    for result in results:
        content = result.page_content.strip()

        if content not in seen_content:
            seen_content.add(content)
            unique_results.append(result)

    return unique_results

def run_thinkbot(mode, question):
    if mode == "Answer as Chatbot":
        response = chatbot_chain.invoke({
            "question": question
        })
        return response.content

    elif mode == "Answer from Knowledge Base":
        response = qa.invoke({
            "query": question
        })
        return response["result"]

    elif mode == "Search Knowledge Base":
        results = search_knowledge_base(question)

        return "\n\n".join(
            result.page_content for result in results
        )

    else:
        return "Please select a valid ThinkBot mode."

