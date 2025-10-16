# app.py — Text Insight Lab (UX/UI mejorado)
# Reqs: streamlit, textblob, googletrans==4.0.0-rc1, pandas
# pip install streamlit textblob "googletrans==4.0.0-rc1" pandas

import re
import json
from datetime import datetime
from collections import Counter
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from textblob import TextBlob
from googletrans import Translator

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Text Insight Lab",
    page_icon="💬",
    layout="wide",
    menu_items={"About": "Text Insight Lab – UI/UX mejorado para análisis con TextBlob."},
)

# Paleta rápida (ajústala a tu brand)
PRIMARY = "#4F46E5"  # Indigo 600
ACCENT  = "#06B6D4"  # Cyan 500
OK      = "#22C55E"  # Green 500
WARN    = "#F59E0B"  # Amber 500
BAD     = "#EF4444"  # Red 500
MUTED   = "#94A3B8"  # Slate 400

st.markdown(
    f"""
    <style>
      .card {{
        background: linear-gradient(180deg, rgba(255,255,255,.75), rgba(255,255,255,.55));
        backdrop-filter: blur(8px);
        border: 1px solid rgba(148,163,184,.25);
        border-radius: 18px; padding: 18px; box-shadow: 0 10px 28px rgba(17,24,39,.08);
      }}
      @media (prefers-color-scheme: dark) {{
        .card {{ background: linear-gradient(180deg, rgba(15,23,42,.70), rgba(15,23,42,.5)); }}
      }}
      .badge {{
        display:inline-flex; align-items:center; gap:.5rem; font-weight:600; font-size:.85rem;
        padding:.3rem .65rem; border-radius:999px; border:1px solid rgba(148,163,184,.3);
        background: rgba(148,163,184,.12); color: #e2e8f0;
      }}
      .chip {{ display:inline-block; margin:.25rem .35rem .35rem 0; padding:.45rem .75rem;
               border-radius:999px; border:1px dashed rgba(148,163,184,.45); cursor:pointer; }}
      .chip:hover {{ border-style: solid; }}
      textarea, .stTextInput > div > div > input {{ border-radius: 14px !important; }}
      .muted {{ color:{MUTED}; font-size:.95rem; }}
      .metric-wrap > div[data-testid="stMetric"] {{
        background: rgba(148,163,184,.08); border: 1px solid rgba(148,163,184,.2);
        border-radius: 14px; padding:.6rem .8rem;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# ESTADO
# ──────────────────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history: List[Dict] = []
if "prefill" not in st.session_state:
    st.session_state.prefill = ""

translator = Translator(service_urls=["translate.googleapis.com"])

# ──────────────────────────────────────────────────────────────────────────────
# STOPWORDS (ES + EN) — tu set extendido + limpieza
# ──────────────────────────────────────────────────────────────────────────────
STOPWORDS = set("""
a al algo algunas algunos ante antes como con contra cual cuando de del desde donde durante e el ella
ellas ellos en entre era eras es esa esas ese eso esos esta estas este esto estos ha había han has hasta
he la las le les lo los me mi mía mías mío míos mis mucho muchos muy nada ni no nos nosotras nosotros
nuestra nuestras nuestro nuestros o os otra otras otro otros para pero poco por porque que quien quienes
qué se sea sean según si sido sin sobre sois somos son soy su sus suya suyas suyo suyos también tanto te
tenéis tenemos tener tengo ti tiene tienen todo todos tu tus tuya tuyas tuyo tuyos tú un una uno unos
vosotras vosotros vuestra vuestras vuestro vuestros y ya yo
about above after again against all am an and any are aren't as at be because been before being below
between both but by can't cannot could couldn't did didn't do does doesn't doing don't down during each
few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers
herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me
more most mustn't my myself no nor not of off on once only or other ought our ours ourselves out over own
same shan't she she'd she'll she's should shouldn't so some such than that that's the their theirs them
themselves then there there's these they they'd they'll they're they've this those through to too under
until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where where's
which while who who's whom why why's with would wouldn't you you'd you'll you're you've your yours yourself yourselves
""".split())

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def normalize_spaces(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()

def translate_es_to_en(text: str) -> str:
    try:
        if not text.strip():
            return text
        detection = translator.detect(text)
        src_lang = detection.lang
        # Fuerza traducción solo si no está ya en inglés
        if src_lang != "en":
            translated = translator.translate(text, src=src_lang, dest="en")
            return translated.text
        else:
            return text
    except Exception as e:
        st.warning(f"⚠️ Error al traducir ({e}). Se usará el texto original.")
        return text

def sentiment_blob(en_text: str) -> Tuple[float, float]:
    blob = TextBlob(en_text)
    return float(blob.sentiment.polarity), float(blob.sentiment.subjectivity)

def split_sentences(t: str) -> List[str]:
    return [s.strip() for s in re.split(r"[.!?]+", t) if s.strip()]

def tokenize_basic(t: str) -> List[str]:
    return re.findall(r"\b\w+\b", t.lower())

def count_words(text: str) -> Tuple[Dict[str, int], List[str]]:
    tokens = tokenize_basic(text)
    filtered = [w for w in tokens if w not in STOPWORDS and len(w) > 2 and not w.isnumeric()]
    counts = dict(Counter(filtered))
    counts = dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))
    return counts, filtered

def top_ngrams(tokens: List[str], n: int = 2, topk: int = 10) -> Dict[str, int]:
    if len(tokens) < n:
        return {}
    grams = [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    grams = [g for g in grams if all(part not in STOPWORDS for part in g.split())]
    counts = dict(Counter(grams).most_common(topk))
    return counts

def label_from_polarity(p: float) -> Tuple[str, str, str]:
    if p >= 0.5:  return "Positivo", "😊", OK
    if p <= -0.5: return "Negativo", "😔", BAD
    return "Neutral", "😐", WARN

def add_history(payload: Dict):
    st.session_state.history.insert(0, payload)

def export_history_json() -> str:
    return json.dumps(st.session_state.history, ensure_ascii=False, indent=2)

# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────
col_l, col_r = st.columns([0.75, 0.25])
with col_l:
    st.markdown("### 💬 Text Insight Lab")
    st.markdown('<div class="muted">Analiza sentimiento, subjetividad y keywords con una UI pulida.</div>', unsafe_allow_html=True)
with col_r:
    st.markdown('<div style="text-align:right;"><span class="badge">TextBlob</span> <span class="badge">googletrans</span> <span class="badge">v2.0</span></div>', unsafe_allow_html=True)

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🧭 Guía")
    st.markdown("**Polaridad**: −1 (neg) → 1 (pos)\n\n**Subjetividad**: 0 (hecho) → 1 (opinión)")
    st.markdown("#### ⚡ Ejemplos")
    for ex in [
        "Hoy la app fluyó excelente, el equipo respondió rápido.",
        "El servicio estuvo aceptable, pero puede mejorar el soporte.",
        "Pésima experiencia: tardaron demasiado y no resolvieron."
    ]:
        if st.button(ex, use_container_width=True):
            st.session_state.prefill = ex
    st.markdown("#### ⬇️ Exportar")
    st.download_button(
        "Descargar historial (JSON)",
        data=export_history_json(),
        file_name=f"history_{datetime.now().date()}.json",
        mime="application/json",
        use_container_width=True,
    )

# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────
tab_an, tab_file, tab_kw, tab_hist, tab_info = st.tabs(
    ["🔎 Analizar texto", "📄 Analizar archivo", "🏷️ Keywords & N-grams", "🗂️ Historial", "ℹ️ Info"]
)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — Analizar texto
# ──────────────────────────────────────────────────────────────────────────────
with tab_an:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Analiza sentimiento y subjetividad")
    text_input = st.text_area(
        "Escribe o pega tu texto (ES recomendado):",
        value=st.session_state.prefill,
        height=180,
        placeholder="Ej: “Me encantó la experiencia con el nuevo flujo, fue muy claro y rápido.”",
    )
    c1, c2 = st.columns([1, 1])
    with c1:
        run = st.button("Analizar ahora", type="primary", use_container_width=True)
    with c2:
        if st.button("Limpiar", use_container_width=True):
            st.session_state.prefill = ""
            st.rerun()

    if run and text_input.strip():
        raw_es = normalize_spaces(text_input)
        en_text = translate_es_to_en(raw_es)
        pol, sub = sentiment_blob(en_text)
        label, emoji, color = label_from_polarity(pol)

        # header badges
        st.markdown(
            f"""
            <div style="display:flex; gap:.6rem; align-items:center; margin:.5rem 0 1rem 0;">
              <span class="badge" style="background:{color}; color:white;">{emoji} {label}</span>
              <span class="badge">EN len: {len(en_text)}</span>
              <span class="badge">Modo: {"traducido ES→EN" if raw_es != en_text else "sin traducción"}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Métricas + barras
        met1, met2, met3 = st.columns(3, gap="small")
        with met1:
            st.metric("Polaridad", f"{pol:+.2f}", delta="↑ positivo" if pol>0 else ("↓ negativo" if pol<0 else "—"))
        with met2:
            st.metric("Subjetividad", f"{sub:.2f}", delta="opinión" if sub>=0.5 else "hechos")
        with met3:
            wc = len(raw_es.split())
            st.metric("Palabras (ES)", f"{wc}")

        st.markdown('<div class="metric-wrap">', unsafe_allow_html=True)
        pnorm = (pol + 1) / 2
        st.write("**Polaridad (−1 → 1)**")
        st.progress(max(0.0, min(1.0, pnorm)))
        st.write("**Subjetividad (0 → 1)**")
        st.progress(max(0.0, min(1.0, sub)))
        st.markdown('</div>', unsafe_allow_html=True)

        # Traducción / texto
        with st.expander("Ver textos"):
            a, b = st.columns(2)
            with a:
                st.markdown("**Original (ES)**")
                st.text(raw_es)
            with b:
                st.markdown("**Usado para análisis (EN)**")
                st.text(en_text)

        # Frases
        st.markdown("#### Frases detectadas")
        es_sent = split_sentences(raw_es)
        en_sent = split_sentences(en_text)
        rows = []
        for i in range(min(len(es_sent), len(en_sent))):
            p_i, _ = sentiment_blob(en_sent[i])
            tag, e_i, _c = label_from_polarity(p_i)
            rows.append({"#": i+1, "Original (ES)": es_sent[i], "Traducción (EN)": en_sent[i], "Polaridad": round(p_i, 2), "Etiqueta": tag, "Icono": e_i})
        if rows:
            df_sent = pd.DataFrame(rows)
            st.dataframe(df_sent, use_container_width=True, hide_index=True)
        else:
            st.info("No se detectaron frases.")

        # Diagnóstico rápido
        st.markdown("#### Diagnóstico UX")
        tips = []
        if -0.1 < pol < 0.1: tips.append("El sentimiento es **casi neutral**; añade matices (adjetivos/emoción).")
        if sub < 0.25: tips.append("Texto muy **objetivo**; agrega opinión si buscas subjetividad.")
        if sub > 0.8: tips.append("Texto muy **subjetivo**; agrega hechos para balance.")
        if tips:
            for t in tips: st.warning("• " + t)
        else:
            st.success("Buen balance entre hechos y opinión. ✅")

        # Historial
        add_history({
            "type": "analysis",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "label": label, "emoji": emoji, "polarity": round(pol,3), "subjectivity": round(sub,3),
            "text_es": raw_es, "text_en": en_text
        })
        st.toast("Análisis guardado en historial", icon="💾")

    st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — Analizar archivo
# ──────────────────────────────────────────────────────────────────────────────
with tab_file:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Sube un archivo (.txt, .md, .csv)")
    up = st.file_uploader(" ", type=["txt", "md", "csv"])
    if up is not None:
        try:
            content = up.getvalue().decode("utf-8")
            with st.expander("Ver muestra de contenido"):
                st.text(content[:1000] + ("..." if len(content) > 1000 else ""))
            if st.button("Analizar archivo", type="primary"):
                raw_es = normalize_spaces(content)
                en_text = translate_es_to_en(raw_es)
                pol, sub = sentiment_blob(en_text)
                label, emoji, color = label_from_polarity(pol)

                st.markdown(f'<span class="badge" style="background:{color}; color:white;">{emoji} {label}</span>', unsafe_allow_html=True)
                st.metric("Polaridad", f"{pol:+.2f}")
                st.metric("Subjetividad", f"{sub:.2f}")

                st.write("**Polaridad (−1→1)**")
                st.progress((pol+1)/2)
                st.write("**Subjetividad (0→1)**")
                st.progress(sub)

                add_history({
                    "type": "file",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "filename": up.name,
                    "label": label, "emoji": emoji,
                    "polarity": round(pol,3), "subjectivity": round(sub,3)
                })
                st.toast("Archivo analizado y guardado", icon="📄")
        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — Keywords & N-grams
# ──────────────────────────────────────────────────────────────────────────────
with tab_kw:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Extracción simple de palabras clave y N-grams")
    kw_text = st.text_area(
        "Pega texto (o reutiliza el último análisis desde el historial):",
        height=140, placeholder="Texto para extraer keywords…"
    )
    ckw1, ckw2, ckw3 = st.columns([1,1,1])
    with ckw1:
        topk = st.number_input("Top N palabras", min_value=5, max_value=50, value=15, step=1)
    with ckw2:
        use_bi = st.checkbox("Bigramas", value=True)
    with ckw3:
        use_tri = st.checkbox("Trigramas", value=False)

    if st.button("Extraer", type="primary"):
        if not kw_text.strip() and st.session_state.history:
            # fallback: último texto del historial
            for item in st.session_state.history:
                if item.get("type") == "analysis":
                    kw_text = item["text_en"] or item["text_es"]
                    break

        if kw_text.strip():
            counts, tokens = count_words(kw_text)
            top_words = list(counts.items())[:topk]
            if top_words:
                st.markdown("**Top palabras (filtradas):**")
                st.bar_chart(dict(top_words))
                st.dataframe(pd.DataFrame(top_words, columns=["Palabra", "Frecuencia"]), use_container_width=True, hide_index=True)
                st.download_button(
                    "Descargar palabras (CSV)",
                    pd.DataFrame(top_words, columns=["word", "freq"]).to_csv(index=False).encode("utf-8"),
                    file_name="keywords.csv", mime="text/csv"
                )
            else:
                st.info("No se encontraron palabras significativas.")

            # N-grams
            if use_bi:
                bi = top_ngrams(tokens, n=2, topk=10)
                if bi:
                    st.markdown("**Top 10 Bigramas:**")
                    st.bar_chart(bi)
            if use_tri:
                tri = top_ngrams(tokens, n=3, topk=10)
                if tri:
                    st.markdown("**Top 10 Trigramas:**")
                    st.bar_chart(tri)
        else:
            st.warning("Pega texto o ejecuta un análisis primero.")

    st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — Historial
# ──────────────────────────────────────────────────────────────────────────────
with tab_hist:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Tus últimos análisis")
    if not st.session_state.history:
        st.info("Sin entradas todavía.")
    else:
        for item in st.session_state.history:
            with st.container(border=True):
                if item["type"] == "analysis":
                    st.markdown(f"**Análisis** {item['emoji']} · {item['label']} · {item['timestamp']}")
                    c1, c2, c3 = st.columns([0.5, 0.25, 0.25])
                    c1.write(item["text_es"])
                    c2.metric("Pol.", f"{item['polarity']:+.2f}")
                    c3.metric("Subj.", f"{item['subjectivity']:.2f}")
                elif item["type"] == "file":
                    st.markdown(f"**Archivo** {item['emoji']} · {item['label']} · {item['timestamp']}")
                    st.caption(item.get("filename","(sin nombre)"))
                    c2, c3 = st.columns([0.5, 0.5])
                    c2.metric("Pol.", f"{item['polarity']:+.2f}")
                    c3.metric("Subj.", f"{item['subjectivity']:.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 5 — Info
# ──────────────────────────────────────────────────────────────────────────────
with tab_info:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Acerca del análisis")
    st.markdown(
        """
- **Sentimiento (TextBlob)**: −1 (muy negativo) → 1 (muy positivo)  
- **Subjetividad**: 0 (objetivo) → 1 (subjetivo)  
- **Traducción**: `googletrans` (fallback sin traducir si falla).  
- **Limitaciones**: No capta sarcasmo/ironía compleja; sensible a longitud/contexto.
- **Tip**: Para producción considera modelos más robustos (transformers) y datasets propios.
        """
    )
    st.caption("UI con enfoque en claridad, feedback inmediato y exportables.")
    st.markdown('</div>', unsafe_allow_html=True)
