import os
import json
import requests
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase

# --- Configuration ---
# Ensure the API endpoint matches the one you are running locally
FASTAPI_ENDPOINT = "http://localhost:8000/generate"
EVALUATION_DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'eval.json')

def get_rag_output(question: str):
    """
    Function to query your running FastAPI endpoint.
    It should return the generated answer and the retrieved context.
    
    IMPORTANT: You may need to adjust the keys ('answer', 'retrieved_context')
    to match the actual JSON response format of your FastAPI application.
    """
    payload = {"question": question}
    try:
        response = requests.post(FASTAPI_ENDPOINT, json=payload, timeout=60)
        response.raise_for_status()  # Raise an exception for bad status codes
        response_data = response.json()
        
        # --- ADJUST KEYS HERE ---
        # For example, if your API returns {"result": "...", "context_docs": [...]},
        # change these keys to "result" and "context_docs".
        actual_answer = response_data.get("answer", "Key 'answer' not found in response")
        retrieved_context = response_data.get("retrieved_context", ["Key 'retrieved_context' not found in response"])
        
        return actual_answer, retrieved_context
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API Request Failed: {e}")
        return f"API Error: {e}", [f"Failed to retrieve context due to API error: {e}"]

def main():
    """
    Main function to run the DeepEval evaluation suite.
    """
    # 1. Load Evaluation Dataset
    try:
        with open(EVALUATION_DATASET_PATH, 'r') as f:
            eval_cases = json.load(f)
        print(f"✅ Loaded {len(eval_cases)} evaluation cases from '{EVALUATION_DATASET_PATH}'")
    except FileNotFoundError:
        print(f"❌ Error: Evaluation dataset not found at '{EVALUATION_DATASET_PATH}'")
        return

    # 2. Create Test Cases and Run Evaluation
    print("\n🚀 Starting RAG pipeline evaluation with DeepEval...")
    print("-" * 50)
    
    test_results = []

    for i, case in enumerate(eval_cases):
        question = case['question']
        expected_answer = case['expected_answer']
        ground_truth_context = case['retrieval_context']
        
        print(f"Running Test Case {i+1}/{len(eval_cases)}: '{question}'")

        # Get the actual output from your RAG pipeline
        actual_output, retrieval_context = get_rag_output(question)

        # Create the test case for DeepEval
        test_case = LLMTestCase(
            input=question,
            actual_output=actual_output,
            expected_output=expected_answer,
            retrieval_context=retrieval_context,
            context=ground_truth_context # Ground truth context for comparison
        )
        
        # Define the metrics for this test
        # Note: ContextualPrecision requires `retrieval_context` and `context`.
        metrics = [
            FaithfulnessMetric(threshold=0.7),
            AnswerRelevancyMetric(threshold=0.7),
            ContextualPrecisionMetric(threshold=0.7)
        ]

        # Run the evaluation for this single test case
        assert_test(test_case, metrics)
        test_results.append(test_case)

    print("-" * 50)
    print("✅ Evaluation complete. See the detailed report above for scores on each test case.")
    # DeepEval automatically calculates and displays overall scores at the end.

if __name__ == "__main__":
    main()
    
    
    
    
#     Set the Environment Variable in your Terminal (before running the script):

# For Linux/WSL (your current environment):

# Bash

# export OPENAI_API_KEY="your_openai_api_key_here"
# IMPORTANT: Replace "your_openai_api_key_here" with your actual API key.
# This command sets the variable only for the current terminal session. If you close the terminal and open a new one, you'll need to set it again.

# For permanent setting (Linux/WSL): To make it permanent across sessions, you can add the export line to your shell's configuration file (e.g., ~/.bashrc, ~/.zshrc).

# Bash

# echo 'export OPENAI_API_KEY="your_openai_api_key_here"' >> ~/.bashrc
# source ~/.bashrc # or source ~/.zshrc
# Remember to replace the placeholder.