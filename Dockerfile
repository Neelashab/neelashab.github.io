FROM python:3.11-slim

WORKDIR /app

# Install system packages if needed
RUN apt-get update && apt-get install -y build-essential

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Set environment variables if needed
ENV PORT=8000

# Expose port
EXPOSE 8000

# Start LangGraph API
CMD ["langgraph", "serve"]
