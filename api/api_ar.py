import json
import os
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS to handle frontend requests
import torch
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers import GPT2Tokenizer

# Initialize the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS to allow cross-origin requests

# Download required NLTK resources
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt")
nltk.download('punkt_tab')

# Load NLP tools
lemmatizer = WordNetLemmatizer()

# Medical synonyms dictionary
medical_synonyms = {
    "myocardial infarction": ["heart attack"],
    "hypertension": ["high blood pressure"],
    "diabetes mellitus": ["diabetes"],
    "carcinoma": ["cancer", "malignant tumor"],
    "cerebrovascular accident": ["stroke"],
    "renal failure": ["kidney failure"],
    "chronic obstructive pulmonary disease": ["COPD", "chronic bronchitis", "emphysema"],
    "gastroesophageal reflux disease": ["GERD", "acid reflux"],
    "hepatocellular carcinoma": ["liver cancer"],
    "osteoarthritis": ["degenerative joint disease"],
}

# Load hierarchical structured textbook data
with open("C:/Users/LENOVO/OneDrive/Desktop/Arogo/api/hierarchical_structure.json", "r") as f:
    hierarchical_data = json.load(f)

# Flatten textbook content
documents = []
document_ids = []

def extract_text(node):
    """Extracts text from hierarchical nodes recursively."""
    if node.get("content"):
        documents.append(" ".join(node["content"]))
        document_ids.append(node["node_id"])

    for child in node.get("children", []):
        extract_text(child)

extract_text(hierarchical_data)

# Initialize BM25 Index
tokenized_corpus = [doc.split() for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)

# Load Sentence-BERT model
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
document_embeddings = semantic_model.encode(documents, convert_to_tensor=True)

# Load Cross-Encoder Model for Re-ranking
cross_encoder_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
cross_encoder_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Load GPT Model for Answer Generation
llm = pipeline("text-generation", model="gpt2", device=-1)

def expand_query(query):
    """Expands query with synonyms and lemmatization."""
    tokens = nltk.word_tokenize(query.lower())
    expanded_tokens = []

    for token in tokens:
        lemma = lemmatizer.lemmatize(token)
        if lemma in medical_synonyms:
            expanded_tokens.extend(medical_synonyms[lemma])
        else:
            expanded_tokens.append(lemma)

    return " ".join(expanded_tokens)

def bm25_search(query, top_n=5):
    """Performs BM25 lexical search."""
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    return [{"node_id": document_ids[i], "content": documents[i]} for i in top_indices]

def semantic_search(query, top_n=5):
    """Performs Semantic Search with Sentence-BERT."""
    query_embedding = semantic_model.encode(query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]
    
    top_n = min(top_n, len(documents))
    top_results = torch.topk(similarities, top_n)
    
    return [{"node_id": document_ids[idx], "content": documents[idx]} for idx in top_results.indices]

def rerank_results(query, results):
    """Re-ranks results using Cross-Encoder."""
    input_texts = [f"{query} [SEP] {res['content']}" for res in results]
    inputs = cross_encoder_tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        scores = cross_encoder_model(**inputs).logits.squeeze(-1)

    sorted_results = sorted(zip(results, scores.tolist()), key=lambda x: x[1], reverse=True)
    return [res[0] for res in sorted_results]

def generate_answer(query):
    # Retrieve relevant documents based on the query
    retrieved_documents = semantic_search(query, top_n=3)

    # Debugging: Check if retrieval failed
    if not retrieved_documents:
        print("Warning: No documents retrieved!")
        retrieved_documents = ["No relevant context available."]

    # Ensure retrieved_documents is a list of strings
    if not isinstance(retrieved_documents, list) or not all(isinstance(doc, str) for doc in retrieved_documents):
        print("Error: Retrieved documents are not in the expected format!")
        retrieved_documents = ["No relevant context available."]

    # Concatenate the retrieved documents to provide context
    context = " ".join(retrieved_documents)

    # Limit the context length
    max_context_length = 1024 - 50  # Reserving tokens for the generated answer
    context_words = context.split()
    if len(context_words) > max_context_length:
        context = " ".join(context_words[:max_context_length])

    # Construct the prompt
    prompt = f"Given the following context, answer the question concisely:\n\nContext: {context}\n\nQuestion: {query}"

    # Generate response using the LLM
    response = llm(prompt, max_new_tokens=50, truncation=True, pad_token_id=50256)

    # Extract and return the generated text
    return response[0]['generated_text'].strip()

@app.route("/query", methods=["POST"])  # Changed to POST
def query():
    """Handles API query requests."""
    # Get the query from the POST body (JSON)
    data = request.get_json()
    user_query = data.get("query", "")
    
    if not user_query:
        return jsonify({"error": "No query provided!"}), 400

    expanded_q = expand_query(user_query)
    
    # Retrieve documents
    bm25_results = bm25_search(expanded_q)
    semantic_results = semantic_search(expanded_q)
    
    # Combine results & rerank
    top_results = bm25_results + semantic_results
    reranked_results = rerank_results(expanded_q, top_results)
    
    # Generate final answer
    answer = generate_answer(user_query)

    return jsonify({
        # "query": user_query,
        # "expanded_query": expanded_q,
        # "top_results": reranked_results,
        "answer": answer
    })

if __name__ == "__main__":
    app.run(port=5000, debug=True)
