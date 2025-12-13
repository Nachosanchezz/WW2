from pathlib import Path

# Ruta base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# Carpetas
DATA_RAW = BASE_DIR / "data" / "raw" 
DATA_PROCESSED = BASE_DIR / "data" / "processed"
INDEX_DIR = BASE_DIR / "index"

# Ficheros estándar
WIKI_FILE = DATA_PROCESSED / "wiki_docs.jsonl"
DOCUMENTS_FILE = DATA_PROCESSED / "documentos.jsonl"

# Chunking (igual que tu compa)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
