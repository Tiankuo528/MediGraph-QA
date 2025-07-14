
# MediGraph-QA: End-to-end LLMs-based Advanced Medical Question Answering System

*A production-grade, local deployed, and end-to-end medical question answering system powered by a fine-tuned LLaMA3 model, GraphRAG, and a full MLOps workflow.*

This is a reproducible personal-scale version of my internship project

![Python Version](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)

---

**MediGraph-QA** is an open-source project designed to provide accurate, context-aware answers to complex medical questions. It leverages a LLaMA3-8B model fine-tuned on medical data and enhances its responses using a powerful **Graph-based Retrieval-Augmented Generation (GraphRAG)** pipeline. The entire system is containerized and ready for scalable deployment.

This repository is a demonstration of a full MLOps lifecycle, from data preparation and model fine-tuning to automated evaluation, interactive demonstration, and CI/CD.

## 🏛️ System Architecture

The system operates in two main phases: an offline indexing pipeline that builds the knowledge base, and an online inference pipeline that answers user questions.

```
Offline: Indexing Pipeline
[Medical PDFs] -> [Text Chunker] -> [LLM Embedder] -> [FAISS Vector Store]
-> [Knowledge Graph]

Online: Inference Pipeline
[User Query] -> [FAISS Retriever] -> [Graph Context Expander] -> [RAG Prompt] -> [Fine-Tuned LLaMA3] -> [Generated Answer]
```

## ✨ Key Features

-   🔬 **Domain-Specific Fine-Tuning**: LLaMA3-8B-Instruct fine-tuned with **LoRA** on a medical QA dataset for domain-specific accuracy and tone.
-   🧠 **GraphRAG Retriever**: Goes beyond simple semantic search. Uses a **FAISS** vector index for initial retrieval, then expands context using a **Knowledge Graph** to find nuanced, interconnected information.
-   🚀 **High-Performance Inference**: Deployed with the **vLLM** inference engine for state-of-the-art throughput and low latency, utilizing features like continuous batching.
-   📊 **Automated RAG Evaluation**: Integrated with **DeepEval** to systematically measure the RAG pipeline's performance on metrics like Faithfulness, Answer Relevancy, and Contextual Precision.
-   🔧 **Robust API Server**: A clean, scalable RESTful API built with **FastAPI** provides a standard interface for the QA service.
-   🖥️ **Interactive Web Demo**: A user-friendly **Gradio** web UI for easy, interactive demonstration of the QA system.
-   📦 **Containerized & Scalable**: Fully containerized with **Docker** for reproducibility and includes **Kubernetes** manifests for scalable, production-grade deployment.
-   ⚙️ **Automated MLOps**: Features a **CI/CD pipeline** using GitHub Actions to automate testing, building, and deployment.

## 🛠️ Technology Stack

| Component            | Technology                                |
| -------------------- | ----------------------------------------- |
| **Base Model** | `meta-llama/Meta-Llama-3.1-8B-Instruct`   |
| **Fine-Tuning** | LoRA (PEFT, bitsandbytes)                 |
| **Retriever** | GraphRAG (FAISS + Knowledge Graph)        |
| **Inference Engine** | vLLM                                      |
| **API Server & UI** | FastAPI, Uvicorn, Gradio                  |
| **Evaluation** | DeepEval                                |
| **Deployment** | Docker, Kubernetes                        |
| **CI/CD** | GitHub Actions                            |


Local Development & Validation ----> Packaging, Automation & Production Deployment


🚀 Getting Started: Local Deployment (on A6000 GPU)

# Local Development & Validation

### 1. Environment Setup

#### 1. Prerequisites
Before starting, ensure you have the following installed on your system:

Python: 3.10 or higher.

NVIDIA Drivers and CUDA: Required for vLLM and GPU acceleration.

Docker: For containerization and local testing.

Kubernetes Cluster (Optional): Configured with GPU nodes for production deployment.

Git

#### 2. Local Setup and Installation

Clone the repository and install the necessary Python dependencies.

```bash
# Clone the repo
git clone https://github.com/Tiankuo528/test_llm.git
cd LLaMA3-MedicalQA-GraphRAG

# Create and activate conda environment
conda create -n med-llm python=3.10 
conda activate med-llm  #llm

# Install dependencies
pip install -r requirements.txt
```

#### 3. Base Model Download and Preparation

```bash
# Download the base model and embedding model from Hugging Face
# You will need to log in with your access token
pip install huggingface_hub
huggingface-cli login
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct --exclude "original/*" --local-dir model/meta-llama/Meta-Llama-3.1-8B-Instruct
huggingface-cli download nomic-ai/nomic-embed-text-v1  --local-dir model/nomic-ai/nomic-embed-text-v1  #embedding llm
```



### 2. LLMs Fine-Tuning
First, we fine-tune the base LLaMA3 model on our medical dataset and merge the weights.

#### Step 1: Prepare dataset in `data/train.json`

```json
[
  {
    "instruction": "please explain the common causes of osteoposis",
    "input": "",
    "output": "Osteoposis is commanly cuased by reduced female hormone, lack of calcium absorption..."
  }
]
```

#### Step 2: Run the LoRA fine-tuning script
```bash
python scripts/train_lora.py
```
Output: LoRA adapter weights will be saved in `model/lora_adapter/`

#### Step 3: Merge LoRA weights with the base model for optimal inference
Merge our trained LoRA adapter with base LLaMA3.1–8B-Instruct weights. vLLM performs best with fully merged models. 

```bash
python scripts/merge_weights.py
```
Output: Final, deployment-ready model saved in `model/merged_model/`



### 3. Build GraphRAG Pipeline
Next, we process our source documents (e.g., medical publications in PDF) to create the vector index and knowledge graph for the RAG pipeline.


#### Step 1: Place your medical PDF files into the `data/medical_pdfs/` directory

#### Step 2: Run the embedding and storage script
Create and save the FAISS index and the knowledge graph.

```bash
python scripts/embed_and_store.py
```
Output: FAISS index and Graph data will be saved to disk.

#### Step 3: Knowledge Graph Retrieval + RAG Prompt Assembly 
Construct a chain that takes a question, performs graph-augmented retrieval, and calls the LLM to generate an answer.

```bash
python scripts/rag_chain.py
```



### 4. Launching the Inference API (Local)

Now that the environment and models are ready, launch the FastAPI server which uses vLLM and the GraphRAG pipeline for high-throughput inference to answer questions.

```bash
# Launch the API server from the root directory
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open a new terminal and test the endpoint with a medical question:

```bash
# Launch the API server from the root directory
curl -X POST "http://localhost:8000/generate" \
-H "Content-Type: application/json" \
-d '{
  "question": "how to identify the tumor bone metastasis from X-day image?"
}'
```
You should receive a detailed, medically relevant response in JSON format generated by the system.


### 5. Interactive Demo with Gradio
To provide a user-friendly interface for demonstrations, this project includes a Gradio web UI. It communicates with the FastAPI backend (which must be running).

Launch the Gradio App
```bash
# In a new terminal, launch the Gradio script
python app/gradio_ui.py
```
Now, open your web browser and navigate to ```http://127.0.0.1:7860``` to interact with the medical QA system.


### 6.  Evaluating System with DeepEval
To ensure the reliability and accuracy of our RAG pipeline, we use DeepEval to run a comprehensive evaluation suite. This tests the model's ability to generate faithful, relevant, and contextually grounded answers.

Prepare Evaluation Dataset
Create a test set in ```data/eval.json``` with questions, ground-truth contexts, and expected answers.

Run the Evaluation
Execute the evaluation script, which will run the test cases against your running API and generate a report.

```bash
# Ensure the FastAPI server from Step 4 is running
python deepeval/evaluate_rag.py

```
The script will output a detailed report with scores for each metric, allowing you to track performance and identify areas for improvement.

```
✨ You're using DeepEval!
-------------------------------------------------------------------------
Overall Score: 0.89 | Average Latency: 2.34s
-------------------------------------------------------------------------
Metrics:
- Faithfulness: 0.92
- Answer Relevancy: 0.95
- Contextual Precision: 0.85
-------------------------------------------------------------------------
```




# Packaging, Automation & Production Deployment

### 7.  Containerization (Docker)

To package the application for production, we use Docker. This ensures the application, including the fine-tuned model and knowledge base, is portable.

Ensure Docker is running (Docker Desktop or the Docker daemon).

#### 1. Build the Docker Image
The ```Dockerfile``` packages the entire application, including the fine-tuned model and knowledge base, into a portable container image.

```bash
## Replace 'chutk' with your actual Docker Hub username
docker build -f docker/Dockerfile -t chutk/medical-qa:v1 .
```
 

#### 2. Run the Container Locally
Before deploying to a cluster, test the container locally, ensuring it has GPU access.

```bash
# --gpus all: Grants the container access to all host GPUs
# -p 8000:8000: Maps port 8000 on the host to port 8000 in the container
# Replace 'chutk' with your actual Docker Hub username
docker run --gpus all -p 8000:8000 chutk/medical-qa:v1
```
You can now test this containerized instance using the ```curl``` command from Step 4.




### 8. CI/CD & MLOps Pipeline (GitHub Actions)

To ensure code quality, reproducibility, and automated deployments, this project includes a CI/CD pipeline using GitHub Actions. This setup configures GitHub to automatically test and build your Docker image.

Pipeline Flow:
```
Git Push -> Run Tests -> Build Docker Image -> Push to Registry

┌──────────────────┐   ┌───────────────────────┐   ┌────────────────────────┐
│ Git Push to main ├─► │ Lint & Unit Tests Job ├─► │  Build & Push Docker   │
└──────────────────┘   └───────────────────────┘   └────────────────────────┘
                                                     (Pushes to Docker Hub)
```
#### 1. Configure the Workflow⚙️

Modify Image Name: In the ```ci-cd.yml``` file, locate the line specifying the Docker tags:
```YAML
tags: chutk/medical-qa:latest
```
Crucially, change ```chutk``` to your actual Docker Hub username.


#### 2. Set up GitHub Secrets

This workflow requires credentials to push the image to Docker Hub.

1. Go to your GitHub repository page.

2. Navigate to Settings > Secrets and variables > Actions.

3. Click New repository secret and create the following two secrets:

3.1. DOCKERHUB_USERNAME: Your Docker Hub username.

3.2. DOCKERHUB_TOKEN: A Docker Hub Access Token (Use an Access Token, not your password).


#### 3. Trigger the Pipeline

Push the new workflow file to the ```main``` branch.

```Bash
git add .github/workflows/ci-cd.yml
git commit -m "Add CI/CD workflow"
git push origin main
```  
The pipeline will now trigger automatically on every subsequent push to the ```main``` branch.


### 9.  Deployment on Kubernetes


The ```k8s/``` directory contains manifests for deploying the application on a Kubernetes cluster with GPU nodes.

```bash
k8s/deployment.yaml   #This file defines how to run container, including requesting a GPU and setting the model to use.

k8s/service.yaml  #This file creates a stable network endpoint to access LLM pods.
```
Note: Ensure you update ```k8s/deployment.yaml``` to use the Docker image name you built and pushed in the previous steps (e.g., ```chutk/medical-qa:v1```).


Apply the manifests to your cluster:

```bash
# Deploy the application pods
kubectl apply -f k8s/deployment.yaml
# Expose the application via a service
kubectl apply -f k8s/service.yaml
```

Check the status of the deployment:

```bash
kubectl get pods
kubectl get service
```
You should see a pod with a name like ```vllm-falcon-7b-deployment-... ```with a status of``` Running```.

3. Test the Service:
To access the service from your local machine, use port-forward. Open a new terminal and run:

```bash
kubectl port-forward svc/vllm-falcon-7b-service 8080:80
```


Now, you can send a request to your local localhost:8080, which will be forwarded to the service running in Kubernetes.

```bash
curl http://localhost:8080/v1/completions \
-H "Content-Type: application/json" \
-d '{
    "model": "tiiuae/falcon-7b",
    "prompt": "San Francisco is a",
    "max_tokens": 7,
    "temperature": 0
}'
```
ou have now successfully deployed the LLM on Kubernetes. The HPA will automatically add more pods if the CPU load increases and remove them when the load decreases. 


---

## 💡 Example Q&A

**User Instruction:**
> “Why there is a higher risk of re-fracture after a previous bone fracture？”

**Generated Response:**
> Factors that contribute to a higher risk of re-fracture after a previous bone fracture include untreated osteoporosis, weakened bone structure, and the bone not having fully recovered its weight-bearing capacity after healing.

---

## 🙋 FAQ

**Q: Can I run this on a single A6000?**  
✅ Yes — you can fine-tune with LoRA + 4-bit, and run vLLM inference in quantized mode.

**Q: Can this be extended to multi-turn dialog?**  
🧩 Yes — chain-of-thought and history memory modules are modular and pluggable.

---

## 📎 Credits

- [Meta AI](https://ai.meta.com/research/llama3/) — LLaMA3-70B
- [vLLM Project](https://github.com/vllm-project/vllm)
- [LangChain](https://www.langchain.com/)
- [PEFT by HuggingFace](https://github.com/huggingface/peft)
- [CD/CI] (https://www.xugj520.cn/archives/deploy-llm-app-cicd-guide.html)
---

## 🔐 Disclaimer

This system is intended for research and educational use only. It does not constitute medical advice. Consult licensed professionals for any clinical decisions.

---

## 📬 Contact

**Author**: Tiankuo Chu  
**GitHub**: [github.com/Tiankuo528](https://github.com/Tiankuo528)  
**Email**: *[chutk@udel.edu]*

<!-- #    watch -n 1 nvidia-smi --> 











