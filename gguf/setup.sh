#!/bin/bash
set -e

echo "🚀 [Native C++] NLP 추론 전용 환경 구축 (Only /data/ephemeral)..."


echo "[0/5] 작업 공간 및 캐시 경로 설정..."

WORK_DIR="/data/ephemeral/nlp_workspace"
mkdir -p "$WORK_DIR"
chmod 777 "$WORK_DIR"

export TMPDIR="/data/ephemeral/tmp"
mkdir -p "$TMPDIR"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

export XDG_CACHE_HOME="/data/ephemeral/.cache"
export PIP_CACHE_DIR="/data/ephemeral/.cache/pip"
export UV_CACHE_DIR="/data/ephemeral/.cache/uv"
export HF_HOME="/data/ephemeral/.cache/huggingface"

mkdir -p "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$HF_HOME"

cd "$WORK_DIR"
echo "📂 현재 작업 위치: $(pwd)"

export DEBIAN_FRONTEND=noninteractive
export TZ=Asia/Seoul
ln -snf /usr/share/zoneinfo/$TZ /etc/localtime


echo "📦 [1/5] 시스템 패키지 설치 (Build tools)..."
apt-get update
apt-get install -y vim wget build-essential cmake git curl

apt-get clean
rm -rf /var/cache/apt/archives/*


echo "📦 [2/5] CUDA 12.2 확인 및 설치..."

CUDA_RUN="/data/ephemeral/cuda_installer.run"

if [ ! -d "/usr/local/cuda-12.2" ]; then
    echo "⬇️ CUDA 다운로드..."
    wget -O "$CUDA_RUN" https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
    chmod +x "$CUDA_RUN"
    
    echo "⚙️ CUDA 설치 중..."
    sh "$CUDA_RUN" --silent --toolkit
    
    ln -sf /usr/local/cuda-12.2 /usr/local/cuda
    rm -f "$CUDA_RUN"
    echo "✅ CUDA 설치 완료"
else
    echo "✅ CUDA가 이미 존재합니다."
fi

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDACXX=/usr/local/cuda/bin/nvcc


echo "🐍 [3/5] Python Client 환경 구축..."

pip install uv
uv init --python 3.12.12 . 
uv sync

source .venv/bin/activate

echo "🔥 Python 라이브러리 설치 (OpenAI, Pandas)..."
uv pip install --no-cache-dir \
    openai \
    pandas \
    huggingface_hub \
    jupyter \
    ipykernel


echo "🔨 [4/5] Llama.cpp 원본 엔진 빌드 (GPU 가속)..."

if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp
fi

cd llama.cpp

rm -rf build

cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j 6

echo "✅ 빌드 완료: $(pwd)/build/bin/llama-server 생성됨"
cd ..


echo "💾 [5/5] 모델 저장소 준비..."
mkdir -p models


echo "🎉 [설치 완료]"
