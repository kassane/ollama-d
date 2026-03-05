# syntax=docker/dockerfile:1

# ============================================================
# Stage 1: build — install LDC2 via ldcup, compile library,
#           and run the unit test suite (no Ollama needed).
# ============================================================
FROM ubuntu:24.04 AS builder

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates xz-utils libcurl4 && \
    rm -rf /var/lib/apt/lists/*

# Install ldcup (rustup-like manager for LDC2) and use it to
# install LDC2 1.42.0 (based on DMD 2.112.1 / LLVM 21).
RUN curl -sSf https://raw.githubusercontent.com/kassane/ldcup/main/scripts/install.sh \
        | sh -s -- --no-bootstrap && \
    /root/.dlang/ldcup install ldc2-1.42.0

ENV PATH="/root/.dlang/ldc2-1.42.0/ldc2-1.42.0-linux-x86_64/bin:${PATH}"

WORKDIR /app
COPY . .

# Build the library in release mode (includes -preview=safer)
RUN dub build -b release

# Run the unit test suite (struct serialization — no network needed)
RUN dub test

# ============================================================
# Stage 2: integration — used by docker-compose.yml together
#           with a running Ollama service for live API tests.
# ============================================================
FROM builder AS integration

ENV OLLAMA_HOST=http://ollama:11434

CMD ["sh", "-c", \
    "until curl -sf $OLLAMA_HOST/api/version; do echo 'Waiting for Ollama...'; sleep 2; done && \
     curl -s -X POST $OLLAMA_HOST/api/pull -d '{\"name\":\"llama3.1:8b\",\"stream\":false}' > /dev/null && \
     dub run -b release :simple && \
     dub run -b release :coder -- --prompt 'Write a D function to reverse a string' \
         --model llama3.1:8b --host $OLLAMA_HOST --verbose"]
