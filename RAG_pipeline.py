
import os, json
from dotenv import load_dotenv
load_dotenv()


from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.manager import CallbackManager
from langchain_core.output_parsers import StrOutputParser


# --- Neo4j graph & Cypher QA ---
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph


# from langchain.chains import GraphCypherQAChain




embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=os.environ["GOOGLE_API_KEY"])




def build_vector_index(chunks_jsonl="chunks.jsonl"):
    docs = []
    with open(chunks_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            docs.append(Document(page_content=d["text"], metadata=d["metadata"]))
    Neo4jVector.from_documents(
        docs,
        embedding=embeddings,
        url=os.environ["NEO4J_URI"],
        username=os.environ["NEO4J_USER"],
        password=os.environ["NEO4J_PASSWORD"],
        index_name="rag_chunks"
    )
    print("Vector index built in Neo4j.")



graph = Neo4jGraph(
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USER"],
    password=os.environ["NEO4J_PASSWORD"]
)
graph.refresh_schema()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0, google_api_key=os.environ["GOOGLE_API_KEY"])

cypher_qa = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=False,
    validate_cypher=True,  # safer in POC
    top_k=30,
    allow_dangerous_requests=True            # pull ample context
)

# --- Simple router: prefer graph when user asks relationships, counts, patterns ---
import re
def is_graphy(q: str) -> bool:
    patterns = [
        r"\b(pattern|trend|anomal|correlat|co-occur|after|before|between)\b",
        r"\b(count|how many|most common|top|frequent)\b",
        r"\b(prescrib|diagnos|test result|lab result|history)\b"
    ]
    return any(re.search(p, q.lower()) for p in patterns)

# --- Semantic retriever over chunks ---


# --- Answer synthesis with Gemini ---
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a careful medical QA assistant for insurance review. "
     "Cite evidence snippets as bullet points. If uncertain, say so."),
    ("human",
     "Question: {question}\n\n"
     "Graph facts (optional):\n{graph_context}\n\n"
     "Text snippets:\n{snippets}\n\n"
     "Give a concise, factual answer. Include a short reasoning and numbered citations to quotes.")
])

def run_query(question: str):
    # cb = CallbackManager([LangChainTracer(project_name=os.environ.get("LANGCHAIN_PROJECT","med-rag-poc"))])

    graph_context = ""
    if is_graphy(question):
        try:
            graph_context = cypher_qa.run(question)
        except Exception as e:
            graph_context = f"(Graph path failed: {e})"

    docs = retriever.get_relevant_documents(question)
    snippets = "\n---\n".join([f"[{i+1}] {d.page_content[:800]}" for i,d in enumerate(docs)])

    chain = ANSWER_PROMPT | llm | StrOutputParser()
    return chain.invoke({"question": question, "graph_context": graph_context, "snippets": snippets})

if __name__ == "__main__":
    # First time only:
    build_vector_index("chunks.jsonl")

    vstore = Neo4jVector.from_existing_index(
    embedding=embeddings,
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USER"],
    password=os.environ["NEO4J_PASSWORD"],
    index_name="rag_chunks", # create once; or call .from_documents() below to build it
    )

    retriever = vstore.as_retriever(search_kwargs={"k": 5})


    print(run_query("Give me the summary of patients who are treated by Dr. RUPA BANERJEE"))
