import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer


from config import DOCUMENTS_FILE, INDEX_DIR

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed" 
INDEX_DIR = BASE_DIR / "index"
DOCUMENTS_FILE = DATA_PROCESSED / "documentos.jsonl"


def split_text(text: str, max_chars: int = 1200, overlap: int = 200):
    """Divide un texto largo en chunks de tamaño max_chars,
    solapados 'overlap' caracteres para no cortar ideas a la mitad."""

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end]
        chunks.append(chunk)
        start += max_chars - overlap
    return chunks


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    textos = []
    metadatos = []

    if not DOCUMENTS_FILE.exists():
        raise FileNotFoundError(f"No se encuentra el archivo de documentos: {DOCUMENTS_FILE}")
    
    print(f"[INFO] Cargando documentos desde {DOCUMENTS_FILE}...")
    with open(DOCUMENTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            texto = doc.get("texto", "")
            if not texto:
                continue
            
            chunks = split_text(texto, max_chars=1200, overlap=200)
            for i, chunk in enumerate(chunks):
                nuevo_meta = dict(doc)
                nuevo_meta["texto"] = chunk
                nuevo_meta["chunk_id"] = i
                textos.append(chunk)
                metadatos.append(nuevo_meta)
    print(f"[INFO] Total de chunks creados: {len(textos)}")

    print("[INFO] Generando embeddings con SentenceTransformer...")
    modelo = SentenceTransformer("all-MiniLM-L6-v2")

    print
    embeddings = modelo.encode(textos, batch_size=32, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"[INFO] Total de vectores en el índice: {index.ntotal}")

    index_path = INDEX_DIR / "faiss_index.bin"
    meta_path = INDEX_DIR / "metadata.jsonl"

    faiss.write_index(index, str(index_path))
    with open(meta_path, "w", encoding="utf-8") as f_meta:
        json.dump(metadatos, f_meta, ensure_ascii=False)

    print(f"[DONE] Índice FAISS guardado en: {index_path}")
    print(f"[DONE] Metadatos guardados en: {meta_path}")

if __name__ == "__main__":
    main()
            
    