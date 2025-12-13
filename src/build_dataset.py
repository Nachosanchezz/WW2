import json
from pathlib import Path

from config import DATA_PROCESSED, CHUNK_SIZE, CHUNK_OVERLAP

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

def load_jsonl(path: Path):
    """Carga documentos desde un archivo JSONL, con mensajes de depuración."""
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra el archivo: {path}")
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                print(f"[WARN] Línea vacía en {path} en la línea {i}, se omite.")
                continue
            try:
                obj = json.loads(line)
                docs.append(obj)
            except json.JSONDecodeError as e:
                print(f"[WARN] Línea {i} no válida en {path.name}: {e}")
    print(f"[INFO] Cargados {len(docs)} documentos desde {path}")
    if docs:
        ejemplo = docs[0]
        print(f"[DEBUG] Ejemplo de doc en {path.name}: keys={list(ejemplo.keys())}")
    return docs

def chunk_text(text:str, size: int, overlap: int):
    """Divide un texto largo en chunks solapados."""
    text = text or ""
    length = len(text)
    if length == 0:
        return []

    chunks = []
    start = 0

    while start < length:
        end = min(start + size, length)
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    wiki_path = DATA_PROCESSED / "wiki_docs.jsonl"

    wiki_docs = load_jsonl(wiki_path)

    all_docs = wiki_docs
    print(f"[INFO] Total documentos antes de chunking: {len(all_docs)}")

    if len(all_docs) == 0:
        print("[WARN] No hay documentos para procesar. Saliendo.")
        return
    
    out_path = DATA_PROCESSED / "documentos.jsonl"

    with open(out_path, "w", encoding="utf-8") as f_out:
        total_chunks = 0
        for idx, doc in enumerate(all_docs, start=1):
            texto = doc.get("texto", "")
            if not texto:
                print(f"[WARN] Documento {doc.get('id', idx)} sin texto, se omite.")
                continue

            chunks = chunk_text(texto, CHUNK_SIZE, CHUNK_OVERLAP)

            for i, chunk in enumerate(chunks):
                chunk_doc = {
                    "id": f"{doc.get('id', idx)}_chunk{i+1}",
                    "texto": chunk,
                    "fuente": doc.get("fuente", "desconocida"),
                    "metadata": doc.get("metadata", {}),
                }
                f_out.write(json.dumps(chunk_doc, ensure_ascii=False) + "\n")
                total_chunks += 1
        print(f"[INFO] Total de chunks escritos en {out_path}: {total_chunks}")
if __name__ == "__main__":
    main()