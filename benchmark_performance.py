
import time
import hashlib
import os

# Simulating expensive operations
def simulate_load_embedding_model():
    print("Simulating: Loading embedding model...")
    time.sleep(2)  # Simulate 2 seconds to load model
    return "model_object"

def simulate_pdf_parsing():
    print("Simulating: Parsing PDF...")
    time.sleep(1)  # Simulate 1 second to parse PDF
    return ["doc1", "doc2"]

def simulate_embedding_creation():
    print("Simulating: Creating embeddings and FAISS index...")
    time.sleep(3)  # Simulate 3 seconds to create embeddings
    return "vector_store"

def simulate_get_response():
    # print("Simulating: Getting response from LLM...")
    time.sleep(0.5) # Simulate LLM response time
    return "Answer"

# Current Implementation (Simulated)
def current_ask_simulation(file_content, question):
    start_time = time.time()

    # Every time it loads the model
    embed = simulate_load_embedding_model()

    # Every time it parses the PDF
    docs = simulate_pdf_parsing()

    # Every time it creates embeddings
    vectorstore = simulate_embedding_creation()

    # Getting response
    response = simulate_get_response()

    end_time = time.time()
    return end_time - start_time

# Optimized Implementation (Simulated with simple global cache)
cache = {}
cached_model = None

def optimized_ask_simulation(file_content, question):
    global cached_model
    start_time = time.time()

    file_hash = hashlib.md5(file_content).hexdigest()

    # Model is cached
    if cached_model is None:
        cached_model = simulate_load_embedding_model()

    # Vector store is cached by file hash
    if file_hash not in cache:
        docs = simulate_pdf_parsing()
        vectorstore = simulate_embedding_creation()
        cache[file_hash] = vectorstore
    else:
        # print("Using cached vector store")
        pass

    # Getting response
    response = simulate_get_response()

    end_time = time.time()
    return end_time - start_time

def run_benchmark():
    file_content = b"fake pdf content"
    questions = ["What is this?", "Tell me more.", "Summarize it."]

    print("--- Running Baseline (Current) ---")
    total_baseline_time = 0
    for i, q in enumerate(questions):
        t = current_ask_simulation(file_content, q)
        print(f"Question {i+1} took {t:.2f}s")
        total_baseline_time += t

    print(f"\nTotal Baseline Time: {total_baseline_time:.2f}s")
    print(f"Average Time per Question: {total_baseline_time/len(questions):.2f}s")

    print("\n--- Running Optimized ---")
    total_optimized_time = 0
    for i, q in enumerate(questions):
        t = optimized_ask_simulation(file_content, q)
        print(f"Question {i+1} took {t:.2f}s")
        total_optimized_time += t

    print(f"\nTotal Optimized Time: {total_optimized_time:.2f}s")
    print(f"Average Time per Question: {total_optimized_time/len(questions):.2f}s")

    improvement = (total_baseline_time - total_optimized_time) / total_baseline_time * 100
    print(f"\nOverall Improvement: {improvement:.2f}%")

if __name__ == "__main__":
    run_benchmark()
