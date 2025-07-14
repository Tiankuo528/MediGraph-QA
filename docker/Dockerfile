# Start from the official vllm image which includes CUDA, Python, and vllm.
FROM vllm/vllm-openai:latest

# Set the working directory
WORKDIR /app

# (Optional but recommended)
# Copy only the requirements file first to leverage Docker's build cache.
COPY requirements.txt .

# Install any ADDITIONAL Python dependencies your custom API needs.
# The base image already has most of what you'll need.
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of your application code
COPY . /app

# Expose the port your API will run on
EXPOSE 8000

# Set the command to run your FastAPI app
ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]