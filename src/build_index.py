import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config import DOCUMENTS_FILE, INDEX_DIR  # usa config, no redefinas rutas


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if not DOCUMENTS_FILE.exists():
        raise FileNotFoundError(f"No se encuentra el archivo de documentos: {DOCUMENTS_FILE}")

    textos = []
    metadatos = []

    print(f"[INFO] Cargando documentos desde {DOCUMENTS_FILE}...")
    with open(DOCUMENTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            texto = (doc.get("texto") or "").strip()
            if not texto:
                continue

            # ✅ NO vuelvas a chunkear: documentos.jsonl ya viene chunked de build_dataset.py
            textos.append(texto)
            metadatos.append(doc)

    print(f"[INFO] Total de chunks cargados: {len(textos)}")
    if not textos:
        print("[WARN] No hay textos para indexar. Saliendo.")
        return

    print("[INFO] Generando embeddings con SentenceTransformer...")
    modelo = SentenceTransformer("all-MiniLM-L6-v2")

    # ✅ batch_size más alto = más rápido (si te falta RAM, baja a 64)
    embeddings = modelo.encode(textos, batch_size=128, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"[INFO] Total de vectores en el índice: {index.ntotal}")

    index_path = INDEX_DIR / "faiss_index.bin"
    meta_path = INDEX_DIR / "metadata.jsonl"  # ✅ de verdad JSONL

    faiss.write_index(index, str(index_path))

    # ✅ guarda metadatos en JSONL para no reventar memoria con un json gigante
    with open(meta_path, "w", encoding="utf-8") as f_meta:
        for m in metadatos:
            f_meta.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"[DONE] Índice FAISS guardado en: {index_path}")
    print(f"[DONE] Metadatos guardados en: {meta_path}")


if __name__ == "__main__":
    main()
