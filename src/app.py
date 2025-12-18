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

def img_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

BG_PATH = "assets/bg.jpg"      # <-- tu foto
LOGO_PATH = "assets/logo.png"  # <-- opcional

bg_b64 = img_to_base64(BG_PATH)
logo_b64 = img_to_base64(LOGO_PATH) if Path(LOGO_PATH).exists() else None

CUSTOM_CSS = f"""
<style>
/* Layout general */
.block-container {{
  max-width: 980px;
  padding-top: 2.2rem;
  padding-bottom: 2rem;
}}

/* Fondo con imagen + overlay */
.stApp {{
  background:
    linear-gradient(180deg, rgba(255,255,255,0.92) 0%, rgba(255,255,255,0.92) 100%),
    url("data:image/jpg;base64,{bg_b64}");
  background-size: cover;
  background-position: center;
  background-attachment: fixed;
}}

/* “Hero” arriba */
.hero {{
  text-align: center;
  padding: 22px 18px 14px 18px;
  margin-bottom: 18px;
}}

.hero-card {{
  display: inline-block;
  padding: 18px 22px;
  border-radius: 18px;
  background: rgba(255,255,255,0.78);
  border: 1px solid rgba(0,0,0,0.06);
  backdrop-filter: blur(8px);
  box-shadow: 0 10px 30px rgba(0,0,0,0.08);
}}

.hero-title {{
  font-size: 44px;
  line-height: 1.05;
  margin: 0;
  font-weight: 800;
  letter-spacing: -0.6px;
  color: #111827;
}}

.hero-sub {{
  margin-top: 8px;
  font-size: 15px;
  color: rgba(17,24,39,0.72);
}}

/* Logo grande */
.hero-logo {{
  width: 86px;
  height: 86px;
  margin: 0 auto 10px auto;
  border-radius: 22px;
  background: rgba(255,255,255,0.85);
  border: 1px solid rgba(0,0,0,0.06);
  display:flex;
  align-items:center;
  justify-content:center;
  box-shadow: 0 8px 20px rgba(0,0,0,0.08);
}}

.hero-logo img {{
  width: 58px;
  height: 58px;
  object-fit: contain;
}}

/* Chat container en tarjeta */
.chat-shell {{
  border-radius: 18px;
  background: rgba(255,255,255,0.86);
  border: 1px solid rgba(0,0,0,0.06);
  backdrop-filter: blur(10px);
  box-shadow: 0 12px 34px rgba(0,0,0,0.10);
  padding: 10px 10px 6px 10px;
}}

/* Input más tipo ChatGPT */
div[data-testid="stChatInput"] textarea {{
  border-radius: 999px !important;
  padding: 14px 16px !important;
  border: 1px solid rgba(0,0,0,0.10) !important;
}}

/* Mensajes: un pelín más suaves */
div[data-testid="stChatMessage"] {{
  padding-top: 8px;
  padding-bottom: 8px;
}}
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
APP_NAME = "ARCHIVO 39-45"
TAGLINE = "Si quieres cultivar tu mente con historia, estás en el lugar correcto."

st.markdown('<div class="hero">', unsafe_allow_html=True)
st.markdown('<div class="hero-card">', unsafe_allow_html=True)

if logo_b64:
    st.markdown(
        f'<div class="hero-logo"><img src="data:image/png;base64,{logo_b64}" /></div>',
        unsafe_allow_html=True
    )
else:
    st.markdown('<div class="hero-logo">🪖</div>', unsafe_allow_html=True)

st.markdown(f'<h1 class="hero-title">{APP_NAME}</h1>', unsafe_allow_html=True)
st.markdown(f'<div class="hero-sub">{TAGLINE}</div>', unsafe_allow_html=True)

st.markdown('</div></div>', unsafe_allow_html=True)


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

st.markdown('<div class="chat-shell">', unsafe_allow_html=True)


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

        # llamada real
        try:
            answer = call_llama_custom(prompt, system_prompt=system_prompt)
        except Exception as e:
            answer = f"❌ Error llamando a Ollama: {e}"

        # pinta respuesta en el placeholder
        answer_placeholder.markdown(answer)

    # 4) guarda respuesta en memoria y refresca
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

