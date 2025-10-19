# ===========================================================
# Makefile for installing dependencies for IPBO + PFN + NePS
# ===========================================================

ENV_NAME = ipbo_env
PYTHON_VERSION = 3.11

.PHONY: install
install:
	@echo "Checking if environment '$(ENV_NAME)' exists..."
	@if conda env list | grep -q $(ENV_NAME); then \
		echo "Environment '$(ENV_NAME)' already exists, skipping creation."; \
	else \
		echo "Creating conda environment: $(ENV_NAME)"; \
		conda create -n $(ENV_NAME) python=$(PYTHON_VERSION) -y; \
		echo "Environment created."; \
	fi

	@echo "Installing dependencies..."

	@echo "Trying GPU-compatible PyTorch first..."
	@if conda run -n $(ENV_NAME) pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121; then \
		echo "Installed CUDA 12.1 build of PyTorch."; \
	else \
		echo "Falling back to CPU build of PyTorch..."; \
		conda run -n $(ENV_NAME) pip install torch torchvision torchaudio; \
	fi

	@echo "Installing remaining dependencies..."
	conda run -n $(ENV_NAME) pip install -U ifBO neural-pipeline-search tqdm tensorboard numpy matplotlib
	@echo "All dependencies installed successfully in environment '$(ENV_NAME)'!"

.PHONY: test
test:
	@echo "Testing imports..."
	conda run -n $(ENV_NAME) python -c "import torch, neps, ifbo, torchvision; print('All imports successful!')"
