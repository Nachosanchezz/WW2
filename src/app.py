import streamlit as st
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import base64

# ==========================
# STREAMLIT CONFIG
# ==========================
st.set_page_config(page_title="WW2", page_icon="🪖", layout="centered")

CUSTOM_CSS = """
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==========================
# IMPORTA TU PROYECTO
# ==========================
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
    metadatos = rc.load_metadata_jsonl(rc.META_PATH)
    return index, metadatos

def safe_head(text: str, n: int = 900) -> str:
    text = text or ""
    return text[:n] + ("..." if len(text) > n else "")

# ==========================
# STATE
# ==========================
if "messages" not in st.session_state:
    st.session_state.messages = []  # {role, content}

# ==========================
# HEADER (logo grande)
# ==========================
st.markdown(
    """
    <div style="text-align:center; margin-top: 6px; margin-bottom: 10px;">
        <div style="font-size:72px; line-height: 1;">🪖</div>
        <div style="font-size:42px; font-weight:800;">WW2</div>
        <div style="opacity:0.7; margin-top: 6px;">
            Si quieres aprender y culturizarte sobre la Segunda Guerra Mundial, ¡pregúntame!
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.divider()

# ==========================
# LOAD ONCE
# ==========================
try:
    index, METADATOS = load_index_and_meta()
    embedder = load_embedder()

    # Asegurar que rag_chat usa lo mismo que la app
    rc.index = index
    rc.METADATOS = METADATOS
    rc.embedder = embedder

except Exception as e:
    st.error(f"Error cargando recursos: {e}")
    st.stop()

# ==========================
# (SIN SIDEBAR) - dejamos valores fijos por ahora, como ChatGPT
# ==========================
model_name = rc.LLAMA_MODEL
temperature = 0.2
num_predict = 500
top_p = 0.9

k_final = 10
k_retrieve = 60
use_translation = True
use_rerank = True

# ==========================
# RENDER CHAT HISTORY (sin fuentes)
# ==========================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==========================
# INPUT
# ==========================
user_q = st.chat_input("Escribe tu pregunta…")

if user_q:
    # 1) guarda usuario en memoria
    st.session_state.messages.append({"role": "user", "content": user_q})

    # 2) pinta inmediatamente el mensaje del usuario (sin esperar)
    with st.chat_message("user"):
        st.markdown(user_q)

    # 3) prepara la zona del asistente y un placeholder para ir actualizando
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        answer_placeholder.markdown("Pensando…")

        # ==========================================================
        # ✅ BLOQUE NUEVO: FILTRO DE ALCANCE (NO LLAMAR AL LLM SI NO WW2)
        # ==========================================================
        is_related = True
        try:
            # Si existe en tu rag_chat.py, usamos TU función
            if hasattr(rc, "is_ww2_related"):
                is_related = bool(rc.is_ww2_related(user_q))
        except Exception:
            is_related = True  # si falla, no bloqueamos por error

        if not is_related:
            answer = "Fuera del alcance del sistema."
            answer_placeholder.markdown(answer)

        else:
            # decide query para retrieval
            retrieval_q = user_q
            if use_translation:
                try:
                    retrieval_q = rc.translate_question_to_english(user_q)
                except Exception:
                    retrieval_q = user_q  # fallback

            # retrieval
            if use_rerank:
                context_docs = rc.retrieve_context(
                    retrieval_q,
                    k_final=int(k_final),
                    k_retrieve=int(k_retrieve)
                )
            else:
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

            # ==========================================================
            # ✅ BLOQUE NUEVO: SI EL CONTEXTO ESTÁ VACÍO O NO RELACIONADO,
            # NO LLAMAR AL LLM (evita “Napoleón”, “Revolución”, etc.)
            # ==========================================================
            def _looks_relevant(docs: List[Dict[str, Any]], q_es: str) -> bool:
                """
                Heurística simple (sin LLM):
                - Si no hay docs -> no relevante
                - Si hay docs pero TODOS tienen score muy malo -> no relevante
                """
                if not docs:
                    return False

                # si tienes _score (normalmente negativo), filtramos los muy malos
                scores = []
                for d in docs:
                    s = d.get("_score")
                    if isinstance(s, (int, float)):
                        scores.append(float(s))

                # Si no hay scores, asumimos que hay algo
                if not scores:
                    return True

                # Ajusta este umbral si lo necesitas:
                # - como usas -distancia, cuanto más cerca de 0 mejor (menos negativo)
                best = max(scores)
                return best > -1.25  # umbral conservador

            if not _looks_relevant(context_docs, user_q):
                answer = "La respuesta no aparece en las fuentes actuales."
                answer_placeholder.markdown(answer)
            else:
                # prompt + system_prompt (tuyo, intacto)
                prompt = rc.build_rag_prompt(user_q, context_docs)

                system_prompt = (
                    "Eres un asistente experto en Segunda Guerra Mundial. "
                    "Respondes SIEMPRE en español. "
                    "No inventes: usa solo el contexto proporcionado. "
                    "Si falta la respuesta literal, dilo y responde con lo más cercano del contexto. "
                    "No inventes."
                )

                # call_llama_custom (tu misma función)
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

                # llamada real (SOLO si hay contexto relevante)
                try:
                    answer = call_llama_custom(prompt, system_prompt=system_prompt)
                except Exception as e:
                    answer = f"❌ Error llamando a Ollama: {e}"

                # pinta respuesta en el placeholder
                answer_placeholder.markdown(answer)

    # 4) guarda respuesta en memoria y refresca
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
