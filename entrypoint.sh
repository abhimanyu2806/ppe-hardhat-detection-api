#!/bin/sh
set -e

MODEL_PATH="model/best.pt"
MODEL_URL="https://drive.google.com/uc?id=1nNNt4hMk5O-czmStw9_6LK-CqRCVh9RF"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Model weights not found locally -- downloading from Drive..."
    mkdir -p model
    gdown "$MODEL_URL" -O "$MODEL_PATH"
    echo "Model download complete."
else
    echo "Model weights already present, skipping download."
fi

exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"