# Ollama
brew install ollama
brew services start ollama
ollama --version
curl -fsS http://127.0.0.1:11434/api/version

# ChromaDB（Python 3.13 虛擬環境，裝喺 ~/rag）
brew install python@3.13
mkdir -p "$HOME/rag"
/opt/homebrew/bin/python3.13 -m venv "$HOME/rag/.venv"
"$HOME/rag/.venv/bin/python" -m pip install -U pip
"$HOME/rag/.venv/bin/python" -m pip install chromadb
"$HOME/rag/.venv/bin/python" -c "import chromadb; print(chromadb.__version__)"

# Luanch 
cd ~/rag
source .venv/bin/activate