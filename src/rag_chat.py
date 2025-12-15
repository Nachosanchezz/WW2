import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from config import INDEX_DIR

# ==========================
# RUTAS
# ==========================

INDEX_PATH = INDEX_DIR / "faiss_index.bin"
META_PATH = INDEX_DIR / "metadata.jsonl"  # JSONL real (una línea por doc)

# ==========================
# MODELOS
# ==========================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # mismo que build_index
OLLAMA_URL = "http://localhost:11434/api/chat"
LLAMA_MODEL = "llama3.1:latest"  # cambia si usas otro: "llama3.2", "mistral", etc.


# ==========================
# CARGA DE ÍNDICE Y METADATOS
# ==========================

def load_metadata_jsonl(path: Path) -> List[Dict[str, Any]]:
    meta: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                meta.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Línea {i} inválida en {path.name}: {e}")
    return meta


print("[INFO] Cargando índice FAISS y metadatos...")

if not INDEX_PATH.exists():
    raise FileNotFoundError(f"No se encuentra el índice: {INDEX_PATH}")
if not META_PATH.exists():
    raise FileNotFoundError(f"No se encuentran los metadatos: {META_PATH}")

index = faiss.read_index(str(INDEX_PATH))
METADATOS = load_metadata_jsonl(META_PATH)

print(f"[INFO] Vectores en índice: {index.ntotal}")
print(f"[INFO] Metadatos cargados: {len(METADATOS)}")

if index.ntotal != len(METADATOS):
    print("[WARN] OJO: index.ntotal != len(METADATOS). "
          "Puede pasar si cambiaste algo entre indexado y metadatos.")


print("[INFO] Cargando modelo de embeddings...")
# opcional si quieres forzar offline:
# os.environ["HF_HUB_OFFLINE"] = "1"
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)


# ==========================
# RETRIEVAL
# ==========================

def retrieve_context(question: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Devuelve los k chunks más parecidos según FAISS.
    Para IndexFlatL2: distances es distancia (menor = mejor).
    Para IndexFlatIP: distances es similitud (mayor = mejor).
    """
    q_vec = embedder.encode([question], show_progress_bar=False)
    q_vec = np.asarray(q_vec, dtype="float32")

    distances, indices = index.search(q_vec, k)

    results: List[Dict[str, Any]] = []
    for rank, idx in enumerate(indices[0]):
        if 0 <= idx < len(METADATOS):
            doc = dict(METADATOS[idx])
            doc["_rank"] = rank + 1
            doc["_score"] = float(distances[0][rank])
            results.append(doc)
    return results


# ==========================
# OLLAMA
# ==========================

def call_llama(prompt: str, system_prompt: Optional[str] = None) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": LLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.2,
        },
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, dict) and "message" in data:
        return data["message"].get("content", "").strip()

    return str(data)


# ==========================
# PROMPT RAG
# ==========================

def build_rag_prompt(question: str, context_docs: List[Dict[str, Any]]) -> str:
    context_parts = []

    for doc in context_docs:
        fuente = doc.get("fuente", "desconocida")
        meta = doc.get("metadata", {}) or {}
        title = meta.get("title") or meta.get("filename") or ""
        url = meta.get("url") or ""

        header = f"[Fuente: {fuente}"
        if title:
            header += f" | Título: {title}"
        if url:
            header += f" | URL: {url}"
        header += "]\n"

        context_parts.append(header + (doc.get("texto") or ""))

    context_str = "\n\n---\n\n".join(context_parts)

    return f"""
Usa EXCLUSIVAMENTE la siguiente información de contexto para responder a la pregunta.
Si la respuesta no está claramente en el contexto, di que no aparece en los documentos.

Si la pregunta pide un número (personas, muertos, años, fechas), responde PRIMERO con la cifra o la fecha,
y luego añade como máximo una breve explicación de 1–2 frases. No des contexto general si no se pide.

Contexto:
{context_str}

Pregunta:
{question}

Responde en español, claro y breve.
""".strip()


# ==========================
# RAG END-TO-END
# ==========================

def answer_with_rag(question: str, k: int = 5) -> Dict[str, Any]:
    context_docs = retrieve_context(question, k=k)
    prompt = build_rag_prompt(question, context_docs)

    system_prompt = (
        "Eres un asistente experto en Segunda Guerra Mundial. "
        "Respondes SIEMPRE en español. "
        "Tu prioridad es responder directo y conciso. "
        "No inventes: usa solo el contexto proporcionado. "
        "Si el contexto no tiene la respuesta, dilo claramente."
    )

    answer = call_llama(prompt, system_prompt=system_prompt)

    return {
        "question": question,
        "answer": answer,
        "context_docs": context_docs,
    }


# ==========================
# CLI
# ==========================

if __name__ == "__main__":
    print(">>> Chat RAG (WW2) con Ollama. Escribe 'salir' para terminar.")
    while True:
        q = input("\nTú: ").strip()
        if not q:
            continue
        if q.lower() in {"salir", "exit", "quit"}:
            break

        result = answer_with_rag(q, k=5)

        print("\nAsistente:\n")
        print(result["answer"])

        print("\n[Fuentes usadas]")
        for i, doc in enumerate(result["context_docs"], start=1):
            fuente = doc.get("fuente", "desconocida")
            meta = doc.get("metadata", {}) or {}
            title = meta.get("title") or meta.get("filename") or ""
            score = doc.get("_score")
            print(f"{i}. {fuente} - {title} (score={score})")
