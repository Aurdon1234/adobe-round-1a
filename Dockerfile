# This explicitly sets the platform, as requested in the challenge guidelines.
FROM --platform=linux/amd64 python:3.10-slim

# Best Practice: Set a working directory.
WORKDIR /app

# Copy only what's needed first for faster caching
COPY requirements.txt .

# Install dependencies (add --no-cache-dir to speed up without caching)
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the necessary application code into the container.
COPY . .

# Specify the command to run your application.
CMD ["python", "process_pdfs.py"]