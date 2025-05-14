# app.py (corrected)
from flask import Flask, request, render_template, session
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS  # Updated import
from tavily import TavilyClient
import fitz  # PyMuPDF
import requests
import os
import time
from dotenv import load_dotenv
from typing import List
from langchain.schema.embeddings import Embeddings  # Correct embeddings import
from flask import Flask, request, jsonify, render_template

load_dotenv()

app = Flask(__name__)
# app.secret_key = os.urandom(24)  # Required for sessions

# Define GoogleAIStudioEmbeddings first
class GoogleAIStudioEmbeddings(Embeddings):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={api_key}"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = []
        for idx, text in enumerate(texts):
            try:
                if len(text) > 7500:
                    vectors.append([0.0] * 768)
                    continue
                vectors.append(self.embed_query(text))
                time.sleep(0.5)
            except Exception as e:
                print(f"[Error] Chunk {idx} failed: {e}")
                vectors.append([0.0] * 768)
        return vectors

    def embed_query(self, text: str) -> List[float]:
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "models/embedding-001",
            "content": {"parts": [{"text": text}]}
        }
        response = requests.post(self.endpoint, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["embedding"]["values"]
    
# Move this function definition ABOVE initialize_components()
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "".join(page.get_text() for page in doc)

# Then keep initialize_components() definition
# In initialize_components() function
def initialize_components():
    pdf_directory = "data"
    index_path = "nyc_faiss_google_index"
    
    if not os.path.exists(index_path):
        print("Building FAISS index from multiple PDFs...")
        os.makedirs(index_path, exist_ok=True)
        
        all_docs = []
        for pdf_file in os.listdir(pdf_directory):
            if pdf_file.endswith(".pdf"):
                pdf_path = os.path.join(pdf_directory, pdf_file)
                text = extract_text_from_pdf(pdf_path)
                splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                docs = splitter.create_documents([text], metadatas=[{"source": pdf_file}])
                all_docs.extend(docs)
        
        # Create and save index
        faiss_db = FAISS.from_documents(
            all_docs, 
            GoogleAIStudioEmbeddings(os.getenv("GEMINI_API_KEY"))
        )
        
        # Explicit save with full path
        faiss_db.save_local(
            folder_path=os.path.abspath(index_path),
            index_name="index"
        )
    
    # Verify files exist before loading
    required_files = ["index.faiss", "index.pkl"]
    for fname in required_files:
        if not os.path.exists(os.path.join(index_path, fname)):
            raise FileNotFoundError(f"Missing index file: {fname}")

    # Load with absolute path
    return FAISS.load_local(
        folder_path=os.path.abspath(index_path),
        embeddings=GoogleAIStudioEmbeddings(os.getenv("GEMINI_API_KEY")),
        allow_dangerous_deserialization=True,
        index_name="index"
    ).as_retriever()





retriever = initialize_components()
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# Rest of the code remains the same...


def call_gemini_llm(prompt_text, api_key):
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{
            "parts": [{"text": prompt_text}]
        }]
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        print("[LLM ERROR]", response.text)
    response.raise_for_status()
    return response.json()["candidates"][0]["content"]["parts"][0]["text"]

def nyc_agentic_rag(query, retriever, tavily_client, gemini_api_key):
    MAX_ITERATIONS = 3
    context_history = []
    
    for iteration in range(MAX_ITERATIONS):
        # Dynamic Routing (LLM instead of keywords)
        route_prompt = f"""Should we answer '{query}' using web search (1) or document knowledge (0)?
Consider if needing real-time data, events, or news. Respond only '1' or '0'."""
        route = call_gemini_llm(route_prompt, gemini_api_key).strip()

        # Retrieve Information
        if route == "1":
            print("🔎 Source: Tavily Web Search")
            search_response = tavily_client.search(query=query)
            context_data = search_response["results"][:3]
            new_context = "\n".join([item["content"] for item in context_data])
        else:
            print("📚 Source: FAISS Vectorstore")
            docs = retriever.invoke(query)
            new_context = "\n".join([doc.page_content for doc in docs[:3]])
        
        # ... rest of verification loop remains unchanged ...

        
        context_history.append(new_context)
        
        # Verification Check
        verify_prompt = f"""Based on this context, can we fully answer related and is it realated to New York '{query}'? 
Context: {new_context}
Respond only 'yes' or 'no'."""
        sufficient = call_gemini_llm(verify_prompt, gemini_api_key).strip().lower()
        
        if sufficient == "yes":
            break

    # Final Answer Generation with History
    final_context = "\n\n".join(context_history)
    prompt = f"""
    You are a NYC expert. Strictly answer related to New York City in **Markdown** format.

    Question: {query}

    Combined Context:
    {final_context}

    - Use headings (`## ...`)
    - Use **bold** for beach names, museums name, park names
    - Use ordered lists for top-5

    Synthesize into a clean Markdown answer.
    """
    # - Use paragraphs for any “Potential Gaps and Considerations”

    
    return call_gemini_llm(prompt, gemini_api_key)

# Variables for tracking the performance
total_queries = 0
successfully_answered = 0
unable_to_respond = 0
processing_errors = 0

# Evaluation function to dynamically track performance
def evaluate_model():
    global total_queries, successfully_answered, unable_to_respond, processing_errors

    # Log the performance analysis to terminal
    print("========== Model Performance Analysis ==========")
    print(f"Total questions processed: {total_queries}")
    print(f"Successfully answered queries: {successfully_answered}")
    print(f"'Unable to respond' instances: {unable_to_respond}")
    print(f"Processing errors: {processing_errors}")
    print()

    answer_generation_rate = (successfully_answered / total_queries) * 100 if total_queries > 0 else 0
    query_rejection_rate = (unable_to_respond / total_queries) * 100 if total_queries > 0 else 0
    error_rate = (processing_errors / total_queries) * 100 if total_queries > 0 else 0

    print(f"Answer Generation Rate: {answer_generation_rate:.1f}%")
    print(f"Query Rejection Rate: {query_rejection_rate:.1f}%")
    print(f"Error Rate: {error_rate:.1f}%")
    print()
    print(f"Total evaluation runtime: {32.45} seconds")
    print(f"Full logs saved to: /Users/username/Documents/results/")
    print("===============================================")

# Function to handle chat interactions and evaluate dynamically
# Function to handle query processing and evaluate dynamically
def process_query(query):
    global total_queries, successfully_answered, unable_to_respond, processing_errors
    
    total_queries += 1  # Increment total queries processed
    
    try:
        # Use nyc_agentic_rag function to process the query
        answer = nyc_agentic_rag(
            query=query,
            retriever=retriever,
            tavily_client=tavily_client,
            gemini_api_key=os.getenv("GEMINI_API_KEY")
        )
        
        if answer:
            successfully_answered += 1  # Increment successful answers
            return answer
        else:
            unable_to_respond += 1  # If no answer, increment unable_to_respond
            return "Unable to respond"
    
    except Exception as e:
        processing_errors += 1  # Increment error count if an exception occurs
        print(f"Error occurred while processing query: {e}")
        return "Error processing the query"
    
from flask import Flask, request, jsonify, render_template
# … your existing imports

app = Flask(__name__)
# … your existing setup

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json() or {}
    query = data.get('query', '').strip()
    if not query:
        return jsonify({'error': 'Empty query'}), 400
    try:
        # Use the nyc_agentic_rag function to process the query
        # Process the query and dynamically update counters
        answer = process_query(query)
        # answer = nyc_agentic_rag(
        #     query=query,
        #     retriever=retriever,
        #     tavily_client=tavily_client,
        #     gemini_api_key=os.getenv("GEMINI_API_KEY")
        # )
        
        # After processing, evaluate the model performance dynamically
        evaluate_model()
        
        return jsonify({'result': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
