#!/bin/bash
# One-time setup for Othello AlphaZero training on a cloud GPU instance.
#
# Works both with git clone and with archive extraction:
#   git clone:  bash scripts/cloud_setup.sh
#   archive:    tar -xzf open_spiel_junqi.tar.gz && cd open_spiel_junqi && bash scripts/cloud_setup.sh
#
# After completion, save this instance as a custom image in your cloud console.
set -euo pipefail

PROJ_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJ_DIR"

# ── 1. System deps (Python & PyTorch are pre-installed on this image) ────────
echo "=== [1/4] System packages ==="
sudo apt-get update -y
sudo apt-get install -y build-essential cmake git

# ── 2. C++ deps (skip if already present from archive) ───────────────────────
echo "=== [2/5] C++ dependencies ==="
if [ ! -d pybind11/include ]; then
    git clone --single-branch --depth 1 https://github.com/pybind/pybind11.git pybind11
fi
if [ ! -f open_spiel/pybind11_json/include/pybind11_json/pybind11_json.hpp ]; then
    rm -rf open_spiel/pybind11_json
    git clone --single-branch --depth 1 https://github.com/pybind/pybind11_json.git open_spiel/pybind11_json
fi
if [ ! -d open_spiel/abseil-cpp/absl ]; then
    rm -rf open_spiel/abseil-cpp
    git clone --single-branch --depth 1 https://github.com/abseil/abseil-cpp.git open_spiel/abseil-cpp
fi
if [ ! -f open_spiel/json/include/nlohmann/json.hpp ]; then
    rm -rf open_spiel/json
    git clone --single-branch --depth 1 https://github.com/nlohmann/json.git open_spiel/json
fi
if [ ! -d open_spiel/pybind11_abseil ]; then
    git clone https://github.com/pybind/pybind11_abseil.git open_spiel/pybind11_abseil
fi

# ── 3. Build pyspiel ─────────────────────────────────────────────────────────
echo "=== [3/5] Building pyspiel ==="
rm -rf build && mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DOPEN_SPIEL_BUILD_WITH_PYTHON=ON \
      -DOPEN_SPIEL_BRIDGE_ENABLED=OFF \
      -DPython3_EXECUTABLE=$(which python3) \
      ../open_spiel
cmake --build . -j$(nproc)
cd ..

# ── 4. Python deps & pyspiel ─────────────────────────────────────────────────
echo "=== [4/4] Python deps ==="
pip install numpy attrs absl-py scipy ml-collections nvidia-ml-py psutil -q

# Copy pyspiel .so to site-packages (no rebuild)
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
cp build/python/pyspiel*.so "$SITE_PACKAGES/" 2>/dev/null || true
# Fallback: first-time build from source
if ! python -c "import pyspiel" 2>/dev/null; then
    echo "Building pyspiel from source..."
    pip install -e .
fi

# ── 5. Verify ────────────────────────────────────────────────────────────────
echo "=== Verify ==="
python -c "import pyspiel; s=pyspiel.load_game('othello').observation_tensor_shape(); print(f'Othello obs shape: {s}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

echo ""
echo "=== Setup complete. ==="
echo "Start training:  screen -S train && source venv/bin/activate && python train/train_othello.py"
echo ""
echo "Now shut down this instance and create a custom image from its system disk."
