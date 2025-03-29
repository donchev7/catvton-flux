FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime as builder
WORKDIR /build

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "huggingface_hub[cli]"

ARG HF_TOKEN
ENV HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}

RUN mkdir -p /models/cat-tryoff-flux /models/flux-dev && \
    echo "Downloading cat-tryoff-flux model..." && \
    huggingface-cli download xiaozaa/cat-tryoff-flux --local-dir /models/cat-tryoff-flux --local-dir-use-symlinks False && \
    echo "Downloading FLUX.1-dev model..." && \
    huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir /models/flux-dev --local-dir-use-symlinks False

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY . .

COPY --from=builder /models/cat-tryoff-flux /app/models/cat-tryoff-flux
COPY --from=builder /models/flux-dev /app/models/flux-dev

COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["python", "api.py"]
