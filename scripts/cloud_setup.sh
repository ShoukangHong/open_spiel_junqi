#!/bin/bash
# One-time setup for Othello AlphaZero training on a cloud GPU instance.
# Run:  bash scripts/cloud_setup.sh
# After completion, save this instance as a custom image in your cloud console.
set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
GIT_REPO="https://github.com/<YOUR_USER>/open_spiel_junqi.git"  # <-- CHANGE
GIT_BRANCH="exp"
PYTHON=python3.11

# ── 1. System deps ───────────────────────────────────────────────────────────
echo "=== [1/5] System packages ==="
sudo apt-get update -y
sudo apt-get install -y build-essential cmake git \
    ${PYTHON} ${PYTHON}-dev ${PYTHON}-venv

# ── 2. Clone repo + deps ─────────────────────────────────────────────────────
echo "=== [2/5] Cloning repo ==="
cd ~
if [ ! -d open_spiel_junqi ]; then
    git clone -b "$GIT_BRANCH" "$GIT_REPO" open_spiel_junqi
else
    cd open_spiel_junqi && git pull && cd ..
fi
cd ~/open_spiel_junqi

# Clone OpenSpiel C++ dependencies (skip if already present)
echo "=== OpenSpiel deps ==="
[ -d pybind11 ] || git clone --single-branch --depth 1 https://github.com/pybind/pybind11.git pybind11
[ -d open_spiel/pybind11_json ] && rm -rf open_spiel/pybind11_json
git clone --single-branch --depth 1 https://github.com/pybind/pybind11_json.git open_spiel/pybind11_json
[ -d open_spiel/abseil-cpp ] && rm -rf open_spiel/abseil-cpp
git clone --single-branch --depth 1 https://github.com/abseil/abseil-cpp.git open_spiel/abseil-cpp
[ -d open_spiel/json ] && rm -rf open_spiel/json
git clone --single-branch --depth 1 https://github.com/nlohmann/json.git open_spiel/json
[ -d open_spiel/pybind11_abseil ] || git clone https://github.com/pybind/pybind11_abseil.git open_spiel/pybind11_abseil

# ── 3. Build pyspiel ─────────────────────────────────────────────────────────
echo "=== [3/5] Building pyspiel ==="
rm -rf build && mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DOPEN_SPIEL_BUILD_WITH_PYTHON=ON \
      -DOPEN_SPIEL_BRIDGE_ENABLED=OFF \
      -DPython3_EXECUTABLE=$(which $PYTHON) \
      ../open_spiel
cmake --build . --config Release -j$(nproc)
cd ..

# ── 4. Python env ────────────────────────────────────────────────────────────
echo "=== [4/5] Python venv ==="
$PYTHON -m venv venv
source venv/bin/activate
pip install --upgrade pip -q

# PyTorch with CUDA (adjust cuXXX to match GPU driver: 121=cuda12.1, 118=cuda11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Training deps
pip install numpy attrs absl-py scipy ml-collections

# Install pyspiel
pip install -e .

# ── 5. Verify ────────────────────────────────────────────────────────────────
echo "=== [5/5] Verify ==="
python -c "import pyspiel; s=pyspiel.load_game('othello').observation_tensor_shape(); print(f'Othello obs shape: {s}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "=== Setup complete. ==="
echo "Start training:  screen -S train && source venv/bin/activate && python train/train_othello.py"
echo ""
echo "Now shut down this instance and create a custom image from its system disk."
