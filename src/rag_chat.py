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

EAST_KEYS = [
    "barbarossa", "eastern front", "stalingrad", "leningrad", "moscow",
    "caucasus", "case blue", "fall blau", "kalach", "volga", "typhoon",
    "reichsbahn", "railway gauge", "operation uranus"
]

WEST_KEYS = [
    "overlord", "normandy", "d-day", "market garden", "bulge",
    "torch", "husky", "italian campaign", "north african campaign"
]


def retrieve_context(question: str, k_final: int = 10, k_retrieve: int = 60) -> List[Dict[str, Any]]:
    """
    Recupera k_retrieve candidatos del índice FAISS y devuelve solo los k_final mejores,
    aplicando un rerank heurístico para evitar mezclar frentes (East vs West).
    """
    q_vec = embedder.encode([question], show_progress_bar=False)
    q_vec = np.asarray(q_vec, dtype="float32")

    # Normaliza igual que los documentos (coseno)
    faiss.normalize_L2(q_vec)

    distances, indices = index.search(q_vec, k_retrieve)

    q = question.lower()
    query_is_east = any(k in q for k in EAST_KEYS)

    results: List[Dict[str, Any]] = []
    for rank, idx in enumerate(indices[0]):
        if 0 <= idx < len(METADATOS):
            doc = dict(METADATOS[idx])

            title = ((doc.get("metadata", {}) or {}).get("title") or "").lower()
            text_head = (doc.get("texto") or "")[:2500].lower()

            raw_dist = float(distances[0][rank])  # L2: menor = mejor

            # Base score: invertimos distancia para que "más alto = mejor"
            score = -raw_dist

            # Heurística: si la query es del Este, premia docs del Este y penaliza docs del Oeste
            if query_is_east:
                if any(k in title or k in text_head for k in EAST_KEYS):
                    score += 0.25
                if any(k in title for k in WEST_KEYS):
                    score -= 0.35

            doc["_rank"] = rank + 1
            doc["_raw_dist"] = raw_dist
            doc["_score"] = score
            results.append(doc)

    # Más alto = mejor (porque ya invertimos la distancia y aplicamos bonus/penalty)
    results.sort(key=lambda d: d["_score"], reverse=True)
    return results[:k_final]



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
        "num_predict": 500,   # 👈 MÁS TOKENS DE SALIDA
        "top_p": 0.9
    },
}


    resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, dict) and "message" in data:
        return data["message"].get("content", "").strip()

    return str(data)

# ==========================
# TRADUCCIÓN PARA RETRIEVAL
# ==========================

def translate_question_to_english(question: str) -> str:
    """
    Traduce la pregunta al inglés SOLO para mejorar el retrieval.
    La respuesta final seguirá siendo en español.
    """
    system = (
        "You are a translation engine. Translate to English. "
        "Return ONLY the translated question. No explanations."
    )
    return call_llama(question, system_prompt=system).strip()


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
    Responde usando EXCLUSIVAMENTE la información del CONTEXTO.
    No uses conocimiento externo ni inventes datos.

    PRIMERO, identifica el tipo de pregunta:

    TIPO A — Pregunta factual concreta
    - Pide un dato único (nombre, fecha, año, número, lugar).
    - Ejemplos: 
    "¿En qué año nació Hitler?"
    "¿Con quién se casó Adolf Hitler?"
    "¿Cuántos murieron en X?"

    Responde SOLO con el dato o una frase muy corta.
    NO añadas introducción, viñetas ni contexto adicional.

    TIPO B — Pregunta explicativa
    - Pide causas, consecuencias, desarrollo o explicación.
    - Ejemplos:
    "¿Cómo fue el desembarco de Normandía?"
    "¿Por qué fracasó Barbarroja?"
    "¿Qué papel jugó X?"

    Responde con:
    - 1 párrafo introductorio breve
    - 4–8 viñetas con hechos clave (causa → efecto)
    - 1 párrafo final de conclusión

    REGLAS IMPORTANTES:
    - Si la respuesta literal NO aparece en el contexto, dilo claramente.
    - Si la pregunta es factual y el dato NO aparece, di: 
        "No aparece ese dato en los documentos disponibles."
    - No mezcles estilos: factual = corto, explicativa = desarrollada.

CONTEXTO:
{context_str}

PREGUNTA:
{question}

RESPUESTA (en español):
""".strip()


# ==========================
# RAG END-TO-END
# ==========================

def answer_with_rag(question: str, k: int = 5) -> Dict[str, Any]:
    question_en = translate_question_to_english(question)
    context_docs = retrieve_context(question_en, k_final=10, k_retrieve=60)   # retrieval en inglés
    prompt = build_rag_prompt(question, context_docs)   # pregunta original (ES)

    system_prompt = (
        "Eres un asistente experto en Segunda Guerra Mundial. "
        "Respondes SIEMPRE en español. "
        "No inventes: usa solo el contexto proporcionado. "
        "Si falta la respuesta literal, dilo y responde con lo más cercano del contexto. "
        "No inventes."
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

        result = answer_with_rag(q, k=8)

        print("\nAsistente:\n")
        print(result["answer"])

        print("\n[Fuentes usadas]")
        for i, doc in enumerate(result["context_docs"], start=1):
            fuente = doc.get("fuente", "desconocida")
            meta = doc.get("metadata", {}) or {}
            title = meta.get("title") or meta.get("filename") or ""
            score = doc.get("_score")
            print(f"{i}. {fuente} - {title} (score={score})")
