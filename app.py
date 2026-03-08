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

st.set_page_config(page_title="LinguaFlow AI", page_icon="🔄", layout="wide")

load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")


st.markdown("""
<style>
.main { background-color: #f8f9fa; }
.stButton>button { 
    width: 100%; 
    border-radius: 10px; 
    height: 3.5em; 
    background-color: #2E86C1; 
    color: white; 
    font-weight: bold;
    border: none;
}
.translation-box {
    background-color: #ffffff;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    border-left: 6px solid #2E86C1;
    font-size: 18px;
    color: #2c3e50;
}
</style>
""", unsafe_allow_html=True)

def translate_text(text, from_lang, to_lang):

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are LinguaFlow, a professional AI translation engine. Translate accurately from {source_language} to {target_language}. Maintain the original tone and context. Return only the translated text."),
        ("user", "{text}")
    ])
    
    
    ai = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, groq_api_key=GROQ_KEY)
    
    pipe = prompt | ai | StrOutputParser()
    
    return pipe.invoke({
        "source_language": from_lang,
        "target_language": to_lang,
        "text": text
    })

with st.sidebar:
    st.title("🔄 LinguaFlow AI")
    st.subheader("Smart Translation Hub")
    st.markdown("---")
    mode = st.radio("Select Input Channel", ["📝 Text Input", "📄 Document Upload", "🎙️ Voice Command"])
    st.markdown("---")
    st.info("**Tech Stack:**\n- Llama-3.3 LLM\n- LangChain\n- Groq LPU")

st.title("LinguaFlow: Seamless Multi-Modal Translation")
st.write("Translating Voice, Text, and Documents using Generative Intelligence.")


languages = {"English": "en", "Hindi": "hi", "Spanish": "es", "French": "fr", "German": "de"}

col1, col2 = st.columns(2)
with col1:
    from_lang = st.selectbox("Source Language", list(languages.keys()), index=0)
with col2:
    to_lang = st.selectbox("Target Language", list(languages.keys()), index=1)

st.markdown("---")


if mode == "📝 Text Input":
    input_col, output_col = st.columns(2)
    with input_col:
        st.markdown("### 📥 Source Text")
        text_input = st.text_area("What would you like to translate?", height=200, placeholder="Type here...")
    
    if text_input:
        with output_col:
            st.markdown("### 📤 LinguaFlow Output")
            with st.spinner("Processing through LLM..."):
                translation = translate_text(text_input, from_lang, to_lang)
                st.markdown(f'<div class="translation-box">{translation}</div>', unsafe_allow_html=True)
                
                st.write("") 
                if st.button("🔊 Generate Audio Feedback"):
                    tts = gTTS(text=translation, lang=languages[to_lang])
                    audio_file = BytesIO()
                    tts.write_to_fp(audio_file)
                    st.audio(audio_file)

elif mode == "📄 Document Upload":
    st.subheader("Document Translation Engine")
    uploaded = st.file_uploader("Upload your file (.txt or .docx)", type=["txt", "docx"])
    if uploaded:
        if uploaded.name.endswith(".txt"):
            content = uploaded.read().decode()
        else:
            doc = docx.Document(uploaded)
            content = "\n".join([p.text for p in doc.paragraphs])
        
        if st.button("🚀 Process & Translate"):
            with st.spinner("Extracting text and translating..."):
                result = translate_text(content, from_lang, to_lang)
                st.success("File Processed Successfully!")
                st.markdown(f'<div class="translation-box">{result}</div>', unsafe_allow_html=True)

elif mode == "🎙️ Voice Command":
    st.subheader("Real-Time Voice Translation")
    st.write("Click the button below and start speaking.")
    if st.button("🎤 Start Recording"):
        recognizer = sr.Recognizer()
        with sr.Microphone() as mic:
            st.write("Listening... Please speak clearly.")
            try:
               
                audio_data = recognizer.listen(mic, timeout=5)
                spoken_text = recognizer.recognize_google(audio_data)
                st.info(f"Detected Speech: {spoken_text}")
               
                
                with st.spinner("Translating voice..."):
                    translated = translate_text(spoken_text, from_lang, to_lang)
                    st.success(f"Translated: {translated}")
                
             
                tts = gTTS(text=translated, lang=languages[to_lang])
                audio_file = BytesIO()
                tts.write_to_fp(audio_file)
                st.audio(audio_file)
            except Exception as e:
                st.error("Audio Error: Please ensure your microphone is active.")