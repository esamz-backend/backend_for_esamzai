FROM rust:1.78-slim-bookworm AS builder

# Install libclang + build deps
RUN apt-get update && apt-get install -y \
    libclang-dev \
    clang \
    build-essential \
    cmake \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN cargo build --release

# Download Qwen Q2 model during build
RUN wget -q "https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat-GGUF/resolve/main/qwen1_5-0_5b-chat-q2_k.gguf" \
    -O qwen-0.5b.gguf

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /app/target/release/esamz .
COPY --from=builder /app/qwen-0.5b.gguf .

ENV QWEN_MODEL_PATH=/app/qwen-0.5b.gguf
ENV PORT=3000
EXPOSE 3000
CMD ["./esamz"]
