import os

# Disable TensorFlow to avoid DLL load errors on Windows if not needed
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gc
import uuid
import tempfile
import base64
import logging
from config import Config
from rag_code import Transcribe, EmbedData, QdrantVDB_QB, Retriever, RAG
import streamlit as st

# Configure logging
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(page_title="Audio RAG Chat", page_icon="🎙️", layout="wide")

# Initialize session state
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
    st.session_state.messages = []
    st.session_state.transcripts = []

# Validate configuration
is_valid, error_msg = Config.validate()
if not is_valid:
    st.error(error_msg)
    st.stop()

@st.cache_resource
def get_embed_model():
    return EmbedData(embed_model_name=Config.EMBED_MODEL_NAME, batch_size=Config.BATCH_SIZE)

@st.cache_resource
def get_vdb():
    vdb = QdrantVDB_QB(
        collection_name=Config.COLLECTION_NAME,
        batch_size=Config.BATCH_SIZE,
        vector_dim=Config.VECTOR_DIM
    )
    vdb.define_client()
    vdb.create_collection()
    return vdb

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

with st.sidebar:
    st.header("🎙️ Audio Upload")
    uploaded_file = st.file_uploader("Choose your audio file", type=["mp3", "wav", "m4a"])

    if uploaded_file:
        try:
            file_key = f"{st.session_state.id}-{uploaded_file.name}"

            if file_key not in st.session_state.file_cache:
                with st.status("Processing audio...", expanded=True) as status:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        
                        st.write("Transcribing...")
                        transcriber = Transcribe(api_key=Config.ASSEMBLYAI_API_KEY)
                        transcripts = transcriber.transcribe_audio(file_path)
                        st.session_state.transcripts = transcripts
                        
                        documents = [f"Speaker {t['speaker']}: {t['text']}" for t in transcripts]

                        st.write("Embedding & Indexing...")
                        embeddata = get_embed_model()
                        embeddata.embed(documents)

                        qdrant_vdb = get_vdb()
                        qdrant_vdb.ingest_data(embeddata=embeddata)

                        retriever = Retriever(vector_db=qdrant_vdb, embeddata=embeddata)
                        query_engine = RAG(retriever=retriever, llm_name=Config.LLM_NAME)
                        
                        st.session_state.file_cache[file_key] = query_engine
                    status.update(label="Audio processed successfully!", state="complete", expanded=False)
            else:
                query_engine = st.session_state.file_cache[file_key]

            st.success("Ready to Chat!")
            st.audio(uploaded_file)
            
            with st.expander("Show full transcript"):
                for t in st.session_state.transcripts:
                    st.markdown(f"**{t['speaker']}**: {t['text']}")
                
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            st.error(f"An error occurred: {e}")
            st.stop()

# Main UI
col1, col2 = st.columns([6, 1])

with col1:
    try:
        assembly_logo = base64.b64encode(open("assets/AssemblyAI.png", "rb").read()).decode()
        deepseek_logo = base64.b64encode(open("assets/deep-seek.png", "rb").read()).decode()
        st.markdown(f"""
            # RAG over Audio powered by 
            <img src="data:image/png;base64,{assembly_logo}" width="200" style="vertical-align: -15px; padding-right: 10px;"> 
            and 
            <img src="data:image/png;base64,{deepseek_logo}" width="200" style="vertical-align: -5px; padding-left: 10px;">
        """, unsafe_allow_html=True)
    except FileNotFoundError:
        st.title("🎙️ Chat with your Audio")

with col2:
    st.button("Clear Chat ↺", on_click=reset_chat, use_container_width=True)

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the audio conversation..."):
    if not st.session_state.file_cache:
        st.warning("Please upload an audio file first!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Get the query engine for the latest uploaded file
            latest_file_key = list(st.session_state.file_cache.keys())[-1]
            query_engine = st.session_state.file_cache[latest_file_key]
            
            try:
                streaming_response = query_engine.query(prompt)
                for chunk in streaming_response:
                    if chunk.delta:
                        full_response += chunk.delta
                        message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                logger.error(f"Error during chat: {e}")
                st.error(f"An error occurred while generating response: {e}")