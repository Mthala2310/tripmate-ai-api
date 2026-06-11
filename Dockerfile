FROM python:3.10

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Hugging Face Spaces uses port 7860 by default
EXPOSE 7860

# Run FastAPI server
CMD ["uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "7860"]
