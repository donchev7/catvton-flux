FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install git and huggingface-cli
# RUN apt-get update && apt-get install -y \
#     git \
#     && rm -rf /var/lib/apt/lists/* && \
#     pip install --no-cache-dir "huggingface_hub[cli]"

# Copy application code
COPY . .

# ARG HF_TOKEN
# ENV HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}

# RUN echo "Downloading cat-tryoff-flux model..." && \
#     huggingface-cli download xiaozaa/cat-tryoff-flux && \
#     echo "Downloading FLUX.1-dev model..." && \
#     huggingface-cli download black-forest-labs/FLUX.1-dev

COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["python", "api.py"]
