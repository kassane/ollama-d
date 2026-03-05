# syntax=docker/dockerfile:1

# ============================================================
# Stage 1: build — compile the library and run unit tests
# ============================================================
FROM dlang2/ldc-ubuntu:latest AS builder

WORKDIR /app
COPY . .

# Build the library in release mode
RUN dub build -b release

# Run the unit test suite (struct serialization, no network required)
RUN dub test -b unittest

# ============================================================
# Stage 2: integration — requires a running Ollama service
#           Used by docker-compose.yml for full integration tests
# ============================================================
FROM builder AS integration

# Wait for Ollama to be available, pull a model, then run samples
ENV OLLAMA_HOST=http://ollama:11434

CMD ["sh", "-c", \
    "until curl -sf $OLLAMA_HOST/api/version; do echo 'Waiting for Ollama...'; sleep 2; done && \
     ollama pull llama3.1:8b && \
     dub run -b release :simple && \
     dub run -b release :coder -- --prompt 'Write a D function to reverse a string' --model llama3.1:8b --verbose"]
