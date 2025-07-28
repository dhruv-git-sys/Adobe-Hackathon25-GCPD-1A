FROM --platform=linux/amd64 python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY pdf2segment.py .
COPY main_extractor.py .
COPY main_entry.py .

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Set the entry point to process all PDFs
CMD ["python", "main_entry.py"]
