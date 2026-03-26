import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from gtts import gTTS
from io import BytesIO
import os
from dotenv import load_dotenv
import speech_recognition as sr
import docx

st.set_page_config(page_title="LinguaFlow AI", page_icon="◈", layout="wide")

load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")

# ─── Embedded CSS ─────────────────────────────────────────
st.markdown("""
<style>
:root {
    --bg-void: #060608;
    --bg-deep: #0d0d14;
    --bg-glass: rgba(255,255,255,0.04);
    --bg-glass-hover: rgba(255,255,255,0.07);
    --bg-glass-active: rgba(99,102,241,0.18);
    --border-glass: rgba(255,255,255,0.08);
    --border-accent: rgba(99,102,241,0.5);
    --ink-primary: #f0f0f5;
    --ink-secondary: #8888a8;
    --ink-muted: #44445a;
    --violet: #6366f1;
    --violet-glow: rgba(99,102,241,0.3);
    --violet-dim: rgba(99,102,241,0.12);
    --font-display: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    --font-mono: 'Consolas', 'Courier New', monospace;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-void) !important;
    color: var(--ink-primary) !important;
    font-family: var(--font-mono) !important;
}
[data-testid="stApp"] { background: var(--bg-void) !important; }

[data-testid="stSidebar"] {
    background: var(--bg-deep) !important;
    border-right: 1px solid var(--border-glass) !important;
}
[data-testid="stSidebar"] > div { padding: 2rem 1.5rem !important; }
[data-testid="stSidebar"] h1 {
    font-family: var(--font-display) !important;
    font-size: 1.3rem !important;
    font-weight: 900 !important;
    color: var(--ink-primary) !important;
    letter-spacing: -0.01em !important;
}
[data-testid="stSidebar"] h2 {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    font-weight: 400 !important;
    color: var(--violet) !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] .stAlert {
    background: var(--violet-dim) !important;
    border: 1px solid var(--border-accent) !important;
    border-radius: 12px !important;
    color: var(--ink-secondary) !important;
    font-size: 0.78rem !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
    color: var(--ink-secondary) !important;
    cursor: pointer !important;
}
[data-testid="stSidebar"] .stRadio label:hover { color: var(--ink-primary) !important; }

[data-testid="stMainBlockContainer"] h1 {
    font-family: var(--font-display) !important;
    font-size: clamp(2rem, 5vw, 3.2rem) !important;
    font-weight: 900 !important;
    letter-spacing: -0.03em !important;
    color: #f0f0f5 !important;
    text-shadow: 0 0 40px rgba(99,102,241,0.4);
}
[data-testid="stMainBlockContainer"] p {
    font-family: var(--font-mono) !important;
    color: var(--ink-secondary) !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.02em !important;
}
[data-testid="stMainBlockContainer"] h2 {
    font-family: var(--font-display) !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    color: var(--ink-primary) !important;
}
[data-testid="stMarkdownContainer"] h3 {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: var(--violet) !important;
    margin-bottom: 0.75rem !important;
}

[data-testid="stSelectbox"] > label {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    color: var(--ink-secondary) !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
[data-testid="stSelectbox"] > div > div {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 12px !important;
    color: var(--ink-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.88rem !important;
    transition: border-color 0.2s !important;
}
[data-testid="stSelectbox"] > div > div:hover {
    border-color: var(--border-accent) !important;
    background: var(--bg-glass-hover) !important;
}
[data-testid="stSelectbox"] ul {
    background: #12121e !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 12px !important;
}
[data-testid="stSelectbox"] li {
    font-family: var(--font-mono) !important;
    font-size: 0.85rem !important;
    color: var(--ink-secondary) !important;
}
[data-testid="stSelectbox"] li:hover {
    background: var(--violet-dim) !important;
    color: var(--ink-primary) !important;
}

[data-testid="stTextArea"] label {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    color: var(--ink-secondary) !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
textarea {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 12px !important;
    color: var(--ink-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.9rem !important;
    line-height: 1.7 !important;
    transition: border-color 0.2s, background 0.2s !important;
    resize: none !important;
}
textarea:focus {
    border-color: var(--border-accent) !important;
    background: var(--bg-glass-hover) !important;
    box-shadow: 0 0 0 3px var(--violet-glow) !important;
    outline: none !important;
}
textarea::placeholder {
    color: var(--ink-muted) !important;
    font-style: italic !important;
}

.stButton > button {
    width: 100% !important;
    border-radius: 12px !important;
    height: auto !important;
    padding: 0.7rem 1.5rem !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    background: var(--violet-dim) !important;
    border: 1px solid var(--border-accent) !important;
    color: var(--violet) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: var(--bg-glass-active) !important;
    border-color: var(--violet) !important;
    color: #fff !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px var(--violet-glow) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

.translation-box {
    background: var(--bg-glass);
    border: 1px solid var(--border-glass);
    border-left: 3px solid var(--violet);
    border-radius: 16px;
    padding: 1.5rem 1.75rem;
    font-family: var(--font-mono);
    font-size: 0.92rem;
    line-height: 1.8;
    color: var(--ink-primary);
    position: relative;
    overflow: hidden;
    animation: fadeSlideUp 0.4s ease forwards;
}
.translation-box::before {
    content: 'OUTPUT';
    position: absolute;
    top: 0.75rem;
    right: 1rem;
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    color: var(--violet);
    opacity: 0.6;
}
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

.mode-badge {
    display: inline-block;
    padding: 0.25rem 0.85rem;
    border-radius: 100px;
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    background: var(--violet-dim);
    border: 1px solid var(--border-accent);
    color: var(--violet);
    margin-bottom: 1.25rem;
}
.swap-arrow {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    padding-top: 1.8rem;
    color: var(--ink-muted);
    font-size: 1.2rem;
}
.voice-ring {
    width: 80px;
    height: 80px;
    border-radius: 50%;
    border: 2px solid var(--border-accent);
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 1.5rem auto;
    animation: pulse-ring 2s cubic-bezier(0.4,0,0.6,1) infinite;
}
@keyframes pulse-ring {
    0%, 100% { box-shadow: 0 0 0 0 var(--violet-glow); }
    50%       { box-shadow: 0 0 0 16px transparent; }
}

[data-testid="stFileUploader"] {
    border: 1px dashed var(--border-glass) !important;
    border-radius: 16px !important;
    padding: 2rem !important;
    background: var(--bg-glass) !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--border-accent) !important; }
[data-testid="stFileUploader"] label {
    font-family: var(--font-mono) !important;
    color: var(--ink-secondary) !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

.stSuccess {
    background: rgba(16,185,129,0.1) !important;
    border: 1px solid rgba(16,185,129,0.3) !important;
    border-radius: 12px !important;
    color: #10b981 !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
}
.stInfo {
    background: var(--violet-dim) !important;
    border: 1px solid var(--border-accent) !important;
    border-radius: 12px !important;
    color: var(--violet) !important;
    font-family: var(--font-mono) !important;
}
.stError {
    background: rgba(244,63,94,0.1) !important;
    border: 1px solid rgba(244,63,94,0.3) !important;
    border-radius: 12px !important;
    color: #f43f5e !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
}

audio {
    width: 100% !important;
    border-radius: 12px !important;
    filter: invert(1) hue-rotate(180deg) !important;
    opacity: 0.85 !important;
}
hr { border: none !important; border-top: 1px solid var(--border-glass) !important; margin: 1.5rem 0 !important; }
[data-testid="column"] { padding: 0 0.75rem !important; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-glass); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--violet); }
</style>
""", unsafe_allow_html=True)


# ─── Translation Function ─────────────────────────────────
def translate_text(text, from_lang, to_lang):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are LinguaFlow, a professional AI translation engine. "
                   "Translate accurately from {source_language} to {target_language}. "
                   "Maintain the original tone and context. Return only the translated text."),
        ("user", "{text}")
    ])
    ai = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, groq_api_key=GROQ_KEY)
    pipe = prompt | ai | StrOutputParser()
    return pipe.invoke({"source_language": from_lang, "target_language": to_lang, "text": text})


# ─── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:0.25rem;">
        <span style="font-family:var(--font-mono); font-size:0.65rem;
                     letter-spacing:0.2em; text-transform:uppercase; color:#6366f1;">
            ◈ v2.0
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.title("LinguaFlow")
    st.subheader("Neural Translation Engine")
    st.markdown("---")

    mode = st.radio(
        "Input Channel",
        ["📝  Text", "📄  Document", "🎙️  Voice"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-family:var(--font-mono); font-size:0.72rem;
                color:#44445a; line-height:1.9;">
        <span style="color:#6366f1;">▸</span> Llama-3.3-70B<br>
        <span style="color:#6366f1;">▸</span> LangChain<br>
        <span style="color:#6366f1;">▸</span> Groq LPU<br>
        <span style="color:#6366f1;">▸</span> gTTS
    </div>
    """, unsafe_allow_html=True)


# ─── Header ───────────────────────────────────────────────
st.markdown("""
<div style="padding: 2.5rem 0 1rem;">
    <h1>LinguaFlow</h1>
    <p style="margin-top:0.5rem;">
        Seamless multi-modal translation &nbsp;·&nbsp;
        Voice, Text &amp; Documents &nbsp;·&nbsp;
        Powered by Generative AI
    </p>
</div>
""", unsafe_allow_html=True)


# ─── Language Selector ────────────────────────────────────
languages = {
    "🇬🇧  English":    "en",
    "🇮🇳  Hindi":      "hi",
    "🇪🇸  Spanish":    "es",
    "🇫🇷  French":     "fr",
    "🇩🇪  German":     "de",
    "🇯🇵  Japanese":   "ja",
    "🇨🇳  Chinese":    "zh",
    "🇧🇷  Portuguese": "pt",
    "🇸🇦  Arabic":     "ar",
}
lang_keys = list(languages.keys())

col1, col_arrow, col2 = st.columns([5, 1, 5])
with col1:
    from_lang = st.selectbox("Source Language", lang_keys, index=0)
with col_arrow:
    st.markdown('<div class="swap-arrow">⇄</div>', unsafe_allow_html=True)
with col2:
    to_lang = st.selectbox("Target Language", lang_keys, index=1)

st.markdown("---")


# ─────────────────────────────────────────────────────────
#  MODE: TEXT INPUT
# ─────────────────────────────────────────────────────────
if "📝" in mode:
    st.markdown('<div class="mode-badge">📝 Text Mode</div>', unsafe_allow_html=True)

    input_col, output_col = st.columns(2, gap="large")

    with input_col:
        st.markdown("### ◈ Source Text")
        text_input = st.text_area(
            "Input text",
            height=220,
            placeholder="Start typing what you want to translate...",
            label_visibility="collapsed"
        )
        char_count = len(text_input) if text_input else 0
        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',monospace; font-size:0.68rem; '
            f'color:#44445a; text-align:right; margin-top:-0.5rem;">{char_count} chars</div>',
            unsafe_allow_html=True
        )

    if text_input:
        with output_col:
            st.markdown("### ◈ Translation Output")
            with st.spinner("Running inference ..."):
                translation = translate_text(
                    text_input,
                    from_lang.split("  ")[-1],
                    to_lang.split("  ")[-1]
                )
            st.markdown(
                f'<div class="translation-box">{translation}</div>',
                unsafe_allow_html=True
            )
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            if st.button("🔊  Generate Audio"):
                with st.spinner("Synthesizing speech..."):
                    tts = gTTS(text=translation, lang=languages[to_lang])
                    audio_buf = BytesIO()
                    tts.write_to_fp(audio_buf)
                    audio_buf.seek(0)
                st.audio(audio_buf, format="audio/mp3")


# ─────────────────────────────────────────────────────────
#  MODE: DOCUMENT UPLOAD
# ─────────────────────────────────────────────────────────
elif "📄" in mode:
    st.markdown('<div class="mode-badge">📄 Document Mode</div>', unsafe_allow_html=True)
    st.markdown("### ◈ Upload Document")

    uploaded = st.file_uploader(
        "Drop a .txt or .docx file",
        type=["txt", "docx"],
        label_visibility="collapsed"
    )

    if uploaded:
        if uploaded.name.endswith(".txt"):
            content = uploaded.read().decode("utf-8", errors="ignore")
        else:
            doc = docx.Document(uploaded)
            content = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

        word_count = len(content.split())
        st.markdown(f"""
        <div style="display:flex; gap:2rem; margin: 1rem 0 1.5rem;
                    font-family:var(--font-mono); font-size:0.72rem;">
            <span><span style="color:#6366f1;">FILE </span>{uploaded.name}</span>
            <span><span style="color:#6366f1;">WORDS </span>{word_count:,}</span>
            <span><span style="color:#6366f1;">SIZE </span>{round(uploaded.size/1024, 1)} KB</span>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("▸ Preview source text"):
            st.markdown(f"""
            <div style="font-family:var(--font-mono); font-size:0.8rem;
                        color:#8888a8; line-height:1.7; white-space:pre-wrap;
                        max-height:180px; overflow:auto;">
            {content[:800]}{'...' if len(content) > 800 else ''}
            </div>
            """, unsafe_allow_html=True)

        if st.button("🚀  Process & Translate Document"):
            with st.spinner("Extracting and translating..."):
                result = translate_text(
                    content,
                    from_lang.split("  ")[-1],
                    to_lang.split("  ")[-1]
                )
            st.success("✓  Translation complete")
            st.markdown("### ◈ Translated Output")
            st.markdown(
                f'<div class="translation-box">{result}</div>',
                unsafe_allow_html=True
            )
            st.download_button(
                label="⬇  Download Translation (.txt)",
                data=result.encode("utf-8"),
                file_name=f"translated_{uploaded.name.rsplit('.',1)[0]}.txt",
                mime="text/plain"
            )


# ─────────────────────────────────────────────────────────
#  MODE: VOICE
# ─────────────────────────────────────────────────────────
elif "🎙️" in mode:
    st.markdown('<div class="mode-badge">🎙️ Voice Mode</div>', unsafe_allow_html=True)
    st.markdown("### ◈ Real-Time Voice Translation")

    st.markdown("""
    <div style="font-family:var(--font-mono); font-size:0.8rem;
                color:#8888a8; margin-bottom:1.5rem; line-height:1.8;">
        Ensure your microphone is active.<br>
        Speak clearly after clicking Record — you have 5 seconds.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; margin: 1rem 0 2rem;">
        <div class="voice-ring">
            <span style="font-size:1.6rem;">🎤</span>
        </div>
        <div style="font-family:var(--font-mono); font-size:0.68rem;
                    letter-spacing:0.15em; color:#44445a; text-transform:uppercase;">
            Ready to capture
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("⏺  Start Recording"):
        recognizer = sr.Recognizer()
        with sr.Microphone() as mic:
            with st.spinner("Listening... speak now"):
                try:
                    audio_data = recognizer.listen(mic, timeout=5)
                    spoken_text = recognizer.recognize_google(audio_data)

                    st.markdown(f"""
                    <div style="background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.25);
                                border-radius:10px; padding:1rem 1.25rem; margin:1rem 0;
                                font-family:var(--font-mono); font-size:0.85rem; color:#f0f0f5;">
                        <span style="font-size:0.65rem; letter-spacing:0.15em;
                                     text-transform:uppercase; color:#6366f1;">Detected Speech</span><br><br>
                        {spoken_text}
                    </div>
                    """, unsafe_allow_html=True)

                    with st.spinner("Translating voice..."):
                        translated = translate_text(
                            spoken_text,
                            from_lang.split("  ")[-1],
                            to_lang.split("  ")[-1]
                        )

                    st.markdown("### ◈ Translation Output")
                    st.markdown(
                        f'<div class="translation-box">{translated}</div>',
                        unsafe_allow_html=True
                    )

                    with st.spinner("Generating speech..."):
                        tts = gTTS(text=translated, lang=languages[to_lang])
                        audio_buf = BytesIO()
                        tts.write_to_fp(audio_buf)
                        audio_buf.seek(0)
                    st.audio(audio_buf, format="audio/mp3")

                except sr.WaitTimeoutError:
                    st.error("⚠  No speech detected. Please try again.")
                except sr.UnknownValueError:
                    st.error("⚠  Could not understand audio. Speak more clearly.")
                except Exception as e:
                    st.error(f"⚠  Audio Error: {str(e)}")