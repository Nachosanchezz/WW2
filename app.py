import streamlit as st
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ==========================
# STREAMLIT CONFIG
# ==========================
st.set_page_config(page_title="RAG WW2", page_icon="🪖", layout="wide")

CUSTOM_CSS = """
<style>
.stApp {
    background: radial-gradient(circle at 20% 20%, rgba(255,255,255,0.06), transparent 40%),
                radial-gradient(circle at 80% 0%, rgba(255,255,255,0.05), transparent 35%),
                linear-gradient(180deg, rgba(18,18,18,1) 0%, rgba(10,10,10,1) 100%);
    color: #f5f5f5;
}
h1,h2,h3,h4 { color: #f5f5f5 !important; letter-spacing: 0.3px; }
.card {
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 14px;
    background: rgba(255,255,255,0.03);
}
.badge {
    display:inline-block; padding:4px 10px; border-radius:999px; font-size:12px;
    border:1px solid rgba(255,255,255,0.14); background: rgba(255,255,255,0.03);
    margin-right: 8px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==========================
# IMPORTA TU PROYECTO
# ==========================
# Importamos funciones/constantes de TU rag_chat.py
# (así la app siempre usa exactamente tu lógica)
try:
    import rag_chat as rc
except Exception as e:
    st.error(f"No puedo importar rag_chat.py. Error: {e}")
    st.stop()


# ==========================
# CARGA RECURSOS (cacheados)
# ==========================
@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer(rc.EMBEDDING_MODEL_NAME)

@st.cache_resource(show_spinner=False)
def load_index_and_meta():
    if not rc.INDEX_PATH.exists():
        raise FileNotFoundError(f"No se encuentra el índice: {rc.INDEX_PATH}")
    if not rc.META_PATH.exists():
        raise FileNotFoundError(f"No se encuentra metadata: {rc.META_PATH}")

    index = faiss.read_index(str(rc.INDEX_PATH))

    # usa el loader de tu rag_chat si quieres
    metadatos = rc.load_metadata_jsonl(rc.META_PATH)

    return index, metadatos

def safe_head(text: str, n: int = 900) -> str:
    text = text or ""
    return text[:n] + ("..." if len(text) > n else "")

# ==========================
# STATE
# ==========================
if "messages" not in st.session_state:
    st.session_state.messages = []  # {role, content, sources(optional)}

# ==========================
# SIDEBAR
# ==========================
with st.sidebar:
    st.markdown("## 🪖 RAG — Segunda Guerra Mundial")
    st.markdown(
        '<div class="badge">FAISS</div><div class="badge">Ollama</div><div class="badge">Streamlit</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    st.markdown("### Modelo / Generación")
    model_name = st.text_input("Modelo Ollama", value=rc.LLAMA_MODEL)
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.2, 0.05)
    num_predict = st.slider("Max tokens (num_predict)", 200, 1200, 500, 50)
    top_p = st.slider("top_p", 0.1, 1.0, 0.9, 0.05)

    st.markdown("---")
    st.markdown("### Retrieval")
    k_final = st.slider("k_final (docs al LLM)", 3, 15, 10, 1)
    k_retrieve = st.slider("k_retrieve (candidatos FAISS)", 10, 120, 60, 5)

    use_translation = st.checkbox(
        "Traducir ES→EN para buscar (recomendado)",
        value=True,
        help="Usa tu translate_question_to_english() antes del retrieval."
    )
    use_rerank = st.checkbox(
        "Rerank East/West (tu heurística)",
        value=True,
        help="Si lo desactivas, hacemos retrieval simple sin bonus/penalty."
    )

    st.markdown("---")
    if st.button("🔄 Recargar índice/metadatos"):
        load_index_and_meta.clear()
        st.success("Recargado.")

    if st.button("🧹 Nueva conversación"):
        st.session_state.messages = []
        st.rerun()

# ==========================
# HEADER
# ==========================
c1, c2 = st.columns([1.3, 1])
with c1:
    st.markdown("# 📚 RAG WW2")
    st.caption("Tu RAG (Wikipedia + FAISS) con pregunta en español, retrieval en inglés, respuesta en español.")
with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Rutas**")
    st.caption(f"Índice: `{rc.INDEX_PATH}`")
    st.caption(f"Metadata: `{rc.META_PATH}`")
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================
# LOAD ONCE
# ==========================
try:
    index, METADATOS = load_index_and_meta()
    embedder = load_embedder()

    # 👇 IMPORTANTÍSIMO:
    # tu rag_chat.py ya crea variables globales index/METADATOS/embedder en import,
    # pero aquí aseguramos que SIEMPRE sean los que están cargados en esta app.
    rc.index = index
    rc.METADATOS = METADATOS
    rc.embedder = embedder

except Exception as e:
    st.error(f"Error cargando recursos: {e}")
    st.stop()

# ==========================
# RENDER CHAT HISTORY
# ==========================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📌 Ver fuentes usadas"):
                for i, doc in enumerate(msg["sources"], start=1):
                    meta = doc.get("metadata", {}) or {}
                    title = meta.get("title") or meta.get("filename") or "Sin título"
                    url = meta.get("url") or ""
                    score = doc.get("_score")
                    st.markdown(f"**{i}. {title}**  \nFuente: `{doc.get('fuente','?')}` · score: `{score}`")
                    if url:
                        st.markdown(f"- {url}")
                    st.caption(safe_head(doc.get("texto", ""), 900))

# ==========================
# INPUT
# ==========================
user_q = st.chat_input("Escribe tu pregunta…")

if user_q:
    # 1) usuario
    st.session_state.messages.append({"role": "user", "content": user_q})

    # 2) decide query para retrieval
    retrieval_q = user_q
    if use_translation:
        try:
            retrieval_q = rc.translate_question_to_english(user_q)
        except Exception:
            retrieval_q = user_q  # fallback

    # 3) retrieval
    if use_rerank:
        context_docs = rc.retrieve_context(
            retrieval_q,
            k_final=int(k_final),
            k_retrieve=int(k_retrieve)
        )
    else:
        # retrieval simple (sin heurística)
        q_vec = embedder.encode([retrieval_q], show_progress_bar=False).astype("float32")
        faiss.normalize_L2(q_vec)
        distances, indices = index.search(q_vec, int(k_retrieve))

        tmp = []
        for r, idx in enumerate(indices[0][:int(k_final)]):
            if 0 <= idx < len(METADATOS):
                d = dict(METADATOS[idx])
                d["_rank"] = r + 1
                d["_raw_dist"] = float(distances[0][r])
                d["_score"] = -float(distances[0][r])
                tmp.append(d)
        context_docs = tmp

    # 4) prompt
    prompt = rc.build_rag_prompt(user_q, context_docs)

    system_prompt = (
        "Eres un asistente experto en Segunda Guerra Mundial. "
        "Respondes SIEMPRE en español. "
        "No inventes: usa solo el contexto proporcionado. "
        "Si falta la respuesta literal, dilo y responde con lo más cercano del contexto. "
        "No inventes."
    )

    # 5) llama a Ollama pero con tus opciones personalizadas
    # (para no tocar tu call_llama original, lo replico aquí mínimamente)
    def call_llama_custom(prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(num_predict),
                "top_p": float(top_p)
            },
        }
        import requests
        resp = requests.post(rc.OLLAMA_URL, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "message" in data:
            return data["message"].get("content", "").strip()
        return str(data)

    with st.chat_message("assistant"):
        with st.spinner("Pensando…"):
            try:
                answer = call_llama_custom(prompt, system_prompt=system_prompt)
            except Exception as e:
                answer = f"❌ Error llamando a Ollama: {e}"

        st.markdown(answer)

        with st.expander("📌 Ver fuentes usadas"):
            for i, doc in enumerate(context_docs, start=1):
                meta = doc.get("metadata", {}) or {}
                title = meta.get("title") or meta.get("filename") or "Sin título"
                url = meta.get("url") or ""
                score = doc.get("_score")
                st.markdown(f"**{i}. {title}**  \nFuente: `{doc.get('fuente','?')}` · score: `{score}`")
                if url:
                    st.markdown(f"- {url}")
                st.caption(safe_head(doc.get("texto", ""), 900))

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": context_docs}
    )

    st.rerun()
