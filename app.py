# app.py — Text Insight Lab (versión corregida con traducción robusta)
import re
import json
from datetime import datetime
from collections import Counter
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from textblob import TextBlob
from googletrans import Translator

# ───────────────────────────────────────────────────────────────
# CONFIGURACIÓN GENERAL
# ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Text Insight Lab",
    page_icon="💬",
    layout="wide",
    menu_items={"About": "Text Insight Lab – versión con traducción robusta"},
)

# Paleta de colores (puedes ajustar a tu brand)
PRIMARY = "#4F46E5"  # Indigo
ACCENT  = "#06B6D4"  # Cyan
OK      = "#22C55E"  # Green
WARN    = "#F59E0B"  # Amber
BAD     = "#EF4444"  # Red
MUTED   = "#94A3B8"  # Slate

st.markdown(
    f"""
    <style>
      .card {{
        background: linear-gradient(180deg, rgba(255,255,255,.75), rgba(255,255,255,.55));
        backdrop-filter: blur(8px);
        border: 1px solid rgba(148,163,184,.25);
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 10px 28px rgba(17,24,39,.08);
      }}
      @media (prefers-color-scheme: dark) {{
        .card {{
          background: linear-gradient(180deg, rgba(15,23,42,.7), rgba(15,23,42,.55));
        }}
      }}
      .badge {{
        display:inline-flex; align-items:center; gap:.5rem; font-weight:600;
        font-size:.85rem; padding:.3rem .65rem; border-radius:999px;
        border:1px solid rgba(148,163,184,.3); background: rgba(148,163,184,.12);
        color: #e2e8f0;
      }}
      textarea, .stTextInput > div > div > input {{
        border-radius: 14px !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ───────────────────────────────────────────────────────────────
# ESTADO DE SESIÓN
# ───────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history: List[Dict] = []
translator = Translator(service_urls=["translate.googleapis.com"])

# ───────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ───────────────────────────────────────────────────────────────
def normalize_spaces(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()

def translate_es_to_en(text: str) -> str:
    """Detecta idioma y traduce al inglés si no lo está"""
    try:
        if not text.strip():
            return text
        detection = translator.detect(text)
        src_lang = detection.lang
        if src_lang != "en":
            translated = translator.translate(text, src=src_lang, dest="en")
            return translated.text
        else:
            return text
    except Exception as e:
        st.warning(f"⚠️ Error al traducir: {e}")
        return text

def sentiment_blob(en_text: str) -> Tuple[float, float]:
    blob = TextBlob(en_text)
    return float(blob.sentiment.polarity), float(blob.sentiment.subjectivity)

def split_sentences(t: str) -> List[str]:
    return [s.strip() for s in re.split(r"[.!?]+", t) if s.strip()]

def count_words(text: str) -> Dict[str, int]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    tokens = [t for t in tokens if len(t) > 2 and not t.isnumeric()]
    return dict(Counter(tokens).most_common(15))

def label_from_polarity(p: float) -> Tuple[str, str, str]:
    if p >= 0.5:  return "Positivo", "😊", OK
    if p <= -0.5: return "Negativo", "😔", BAD
    return "Neutral", "😐", WARN

def add_history(payload: Dict):
    st.session_state.history.insert(0, payload)

# ───────────────────────────────────────────────────────────────
# INTERFAZ PRINCIPAL
# ───────────────────────────────────────────────────────────────
st.title("💬 Text Insight Lab")
st.caption("Analiza sentimiento, subjetividad y palabras clave con una interfaz moderna.")

tab_an, tab_hist = st.tabs(["🔎 Analizar texto", "🗂️ Historial"])

# ───────────────────────────────────────────────────────────────
# TAB 1 — ANÁLISIS
# ───────────────────────────────────────────────────────────────
with tab_an:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    texto = st.text_area("✍️ Escribe tu texto (en español preferiblemente):", height=180)
    if st.button("Analizar texto", type="primary"):
        if texto.strip():
            with st.spinner("Traduciendo y analizando..."):
                texto_es = normalize_spaces(texto)
                texto_en = translate_es_to_en(texto_es)
                pol, sub = sentiment_blob(texto_en)
                label, emoji, color = label_from_polarity(pol)

                st.markdown(
                    f'<span class="badge" style="background:{color}; color:white;">{emoji} {label}</span>',
                    unsafe_allow_html=True,
                )
                st.metric("Polaridad", f"{pol:+.2f}")
                st.metric("Subjetividad", f"{sub:.2f}")

                st.write("**Polaridad (−1 → 1)**")
                st.progress((pol + 1) / 2)
                st.write("**Subjetividad (0 → 1)**")
                st.progress(sub)

                # Mostrar traducción
                with st.expander("Ver textos"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Original (ES)**")
                        st.text(texto_es)
                    with col2:
                        st.markdown("**Usado para análisis (EN)**")
                        st.text(texto_en)

                # Frases detectadas
                st.subheader("Frases detectadas")
                frases_es = split_sentences(texto_es)
                frases_en = split_sentences(texto_en)
                filas = []
                for i in range(min(len(frases_es), len(frases_en))):
                    p_i, _ = sentiment_blob(frases_en[i])
                    lbl, emo, _c = label_from_polarity(p_i)
                    filas.append({
                        "#": i + 1,
                        "Original (ES)": frases_es[i],
                        "Traducción (EN)": frases_en[i],
                        "Polaridad": round(p_i, 2),
                        "Etiqueta": lbl,
                        "Icono": emo
                    })
                if filas:
                    st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)
                else:
                    st.info("No se detectaron frases.")

                # Guardar historial
                add_history({
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "texto_es": texto_es,
                    "texto_en": texto_en,
                    "pol": round(pol, 3),
                    "sub": round(sub, 3),
                    "label": label,
                    "emoji": emoji
                })
                st.toast("Análisis guardado ✅", icon="💾")

        else:
            st.warning("Por favor, ingresa algún texto para analizar.")

    st.markdown('</div>', unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────
# TAB 2 — HISTORIAL
# ───────────────────────────────────────────────────────────────
with tab_hist:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Historial de análisis previos")
    if not st.session_state.history:
        st.info("No hay análisis previos aún.")
    else:
        for h in st.session_state.history:
            st.markdown(f"**{h['timestamp']}** — {h['emoji']} *{h['label']}*")
            st.write(f"Polaridad: {h['pol']:+.2f} · Subjetividad: {h['sub']:.2f}")
            with st.expander("Ver texto analizado"):
                st.text(h["texto_es"])
    st.markdown('</div>', unsafe_allow_html=True)
