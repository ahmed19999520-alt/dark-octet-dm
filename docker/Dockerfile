FROM python:3.11-slim

LABEL maintainer="ahmed.ali.mpi@hotmail.com"
LABEL description="Dark Octet DM — EQST-GP Series Paper II"

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ gfortran libopenblas-dev liblapack-dev git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e ".[dev,notebooks]"

RUN mkdir -p /workspace/results /workspace/notebooks

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", \
     "--no-browser", "--allow-root", \
     "--notebook-dir=/workspace/notebooks"]
