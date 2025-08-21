import os, json, re
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_neo4j import GraphCypherQAChain, Neo4jGraph


from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.manager import CallbackManager


LANGCHAIN_PROJECT = os.getenv("LANGSMITH_PROJECT", "Windflow")
LANGCHAIN_API_KEY = os.getenv("LANGSMITH_API_KEY")


if LANGCHAIN_API_KEY:
    tracer = LangChainTracer(project_name=LANGCHAIN_PROJECT)
    callback_manager = CallbackManager([tracer])
else:
    tracer = None
    callback_manager = CallbackManager([])


embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)


vstore = Neo4jVector.from_existing_index(
    embedding=embeddings,
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USER"],
    password=os.environ["NEO4J_PASSWORD"],
    index_name="rag_chunks",
)


graph = Neo4jGraph(
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USER"],
    password=os.environ["NEO4J_PASSWORD"],
    timeout=30  # Add timeout
)




llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  
    temperature=0,
    google_api_key=os.environ["GOOGLE_API_KEY"]
)


CYPHER_GENERATION_TEMPLATE = """You are an expert Neo4j database administrator generating Cypher queries for a healthcare system.

Current schema:
{schema}

CRITICAL RULES:
1. ONLY use node labels and relationships that exist in the schema above
2. ALWAYS use LIMIT to prevent excessive results (default: LIMIT 20)
3. Handle case-insensitive searches with toLower() or LOWER()
4. Return meaningful data, not just counts unless specifically requested
5. Use proper error handling and null checks
6. For doctor names, search in node properties or relationship properties

Common patterns:
- Patient searches: MATCH (p:Patient) WHERE toLower(p.name) CONTAINS toLower('name')
- Doctor searches: MATCH (d:Doctor) or check relationship properties
- Medicine queries: MATCH (m:Medicine) or ()-[r:PRESCRIBED]->(m)
- Test results: MATCH (t:Test)-[:HAS_RESULT]->(r:Result)

Question: {question}
Generate a syntactically correct Cypher query:"""


def create_cypher_chain():
    """Create Cypher QA chain with proper error handling"""
   
    
    try:
        cypher_prompt = ChatPromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)
        
        chain = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=graph,
            cypher_prompt=cypher_prompt,
            verbose=True,
            validate_cypher=True,
            top_k=30,
            allow_dangerous_requests=True,
            return_intermediate_steps=True  # This helps with debugging
        )
        print("Cypher QA chain created successfully")
        return chain
        
    except Exception as e:
        print(f"Failed to create Cypher chain: {e}")
        return None

cypher_qa = create_cypher_chain()

# --- Simplified graph query detection ---
def is_graphy(q: str) -> bool:
   
        
    question_lower = q.lower()
    
    # Key patterns that indicate need for structured queries
    graph_indicators = [
        # Specific entity mentions
        r'\b(patient|doctor|medicine|test|diagnosis|encounter)\b',
        # Relationship queries
        r'\b(treated by|prescribed|diagnosed|tested)\b',
        # Quantitative queries
        r'\b(how many|count|most|least|top|bottom|frequent)\b',
        # Comparison queries
        r'\b(compare|versus|vs|different|similar)\b',
        # Aggregation queries
        r'\b(all|every|total|sum|average)\b',
        # Specific names (doctors, patients)
        r'\b(dr\.|doctor|rupa|banerjee)\b'
    ]
    
    return any(re.search(pattern, question_lower) for pattern in graph_indicators)


def test_cypher_query(cypher: str):
    """Test a Cypher query directly"""
    try:
        result = graph.query(cypher)
        return result, None
    except Exception as e:
        return None, str(e)

def generate_and_test_cypher(question: str):
    """Generate and test Cypher query manually"""
    
    
    try:
        # Create prompt
        cypher_prompt = ChatPromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)
        
        
        generated = cypher_prompt.format(schema=graph.schema, question=question)
        response = llm.invoke(generated)
        
        
        cypher = response.content.strip()
        cypher = re.sub(r'^```(?:cypher)?', '', cypher, flags=re.IGNORECASE).strip()
        cypher = re.sub(r'```$', '', cypher).strip()
        
        print(f"Generated Cypher: {cypher}")
        
        
        result, error = test_cypher_query(cypher)
        
        if error:
            print(f"Cypher execution error: {error}")
            return cypher, error
        else:
            print(f"Cypher result: {result}")
            return cypher, result
            
    except Exception as e:
        print(f"Error generating/testing Cypher: {e}")
        return None, str(e)


retriever = vstore.as_retriever(search_kwargs={"k": 5})


ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a healthcare data analyst providing insights from medical records.

Instructions:
- Use both structured graph data and text snippets to provide comprehensive answers
- If graph data is available, prioritize it as it represents verified relationships
- Clearly distinguish between different data sources
- If information is missing or incomplete, state this explicitly
- Format structured data (like patient lists, prescriptions) clearly
- Always maintain patient privacy and use only the information provided"""),
    
    ("human", """Question: {question}

Structured Database Results:
{graph_context}

Additional Text Context:
{snippets}

Please provide a clear, well-organized answer combining both sources.""")
])


def run_query(question: str, debug: bool = False):
    """Process question with improved error handling"""
    
    
    print(f"Processing: {question}")
   
    
    graph_context = "No structured data available."
    use_graph = is_graphy(question)
    
    print(f"Graph query needed: {use_graph}")
    
    print(f"Cypher chain available: {cypher_qa is not None}")
    
    
    if use_graph and cypher_qa:
        print("Attempting graph query...")
        try:
            
            result = cypher_qa.invoke({"query": question})
            if isinstance(result, dict):
                graph_context = result.get("result", str(result))
            else:
                graph_context = str(result)
                
            print(f"Graph query successful")
            if debug:
                print(f"Graph result: {graph_context}")
                
        except Exception as e:
            print(f"Graph query failed: {e}")
            
            # Method 2: Try manual Cypher generation as fallback
            print("Trying manual Cypher generation...")
            try:
                cypher, result = generate_and_test_cypher(question)
                if result and not isinstance(result, str):  # Not an error
                    graph_context = f"Query results: {result}"
                    print("Manual Cypher successful")
                else:
                    graph_context = f"Graph query failed: {result}"
                    print("Manual Cypher also failed")
            except Exception as e2:
                print(f" Manual Cypher failed: {e2}")
                graph_context = f"Graph queries failed: {e}"
    
    
    print("Performing vector search...")
    try:
        docs = retriever.invoke(question)
        snippets = "\n---\n".join([
            f"[Source {i+1}] {d.page_content[:600]}" 
            for i, d in enumerate(docs)
        ])
        # print(f"Vector search found {len(docs)} documents")
        
    except Exception as e:
        print(f" Vector search failed: {e}")
        snippets = f"Vector search failed: {e}"

    # Generating final answer
    print("Generating answer...")
    try:
        chain = ANSWER_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({
            "question": question,
            "graph_context": graph_context,
            "snippets": snippets
        })
        
        return answer
        
    except Exception as e:
        print(f"Answer generation failed: {e}")
        return f"Error: {e}\n\nGraph Context: {graph_context}\n\nVector Context: {snippets}"




if __name__ == "__main__":
    
    ##"How many patients are in the database?",
        # "What medicines were prescribed most frequently?",
        # "Show me patients treated by Dr. RUPA BANERJEE",
        # "Find abnormal test results"

    answer = run_query("What medicines were prescribed most frequently?", debug=True)
    print(answer)
    
    
    
   
    
   
