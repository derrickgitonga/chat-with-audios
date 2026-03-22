import logging
from qdrant_client import models
from qdrant_client import QdrantClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.sambanovasystems import SambaNovaCloud
import assemblyai as aai
from typing import List, Dict, Optional
from config import Config

from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
)

logger = logging.getLogger(__name__)

def batch_iterate(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]

class EmbedData:
    def __init__(self, embed_model_name=Config.EMBED_MODEL_NAME, batch_size=Config.BATCH_SIZE):
        self.embed_model_name = embed_model_name
        self.embed_model = self._load_embed_model()
        self.batch_size = batch_size
        self.embeddings = []
        self.contexts = []
        
    def _load_embed_model(self):
        try:
            logger.info(f"Loading embedding model: {self.embed_model_name}")
            return HuggingFaceEmbedding(
                model_name=self.embed_model_name, 
                trust_remote_code=True, 
                cache_folder=Config.CACHE_DIR
            )
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise

    def generate_embedding(self, context):
        try:
            return self.embed_model.get_text_embedding_batch(context)
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
        
    def embed(self, contexts):
        self.contexts = contexts
        self.embeddings = []
        
        for batch_context in batch_iterate(contexts, self.batch_size):
            batch_embeddings = self.generate_embedding(batch_context)
            self.embeddings.extend(batch_embeddings)
        logger.info(f"Generated {len(self.embeddings)} embeddings")

class QdrantVDB_QB:
    def __init__(self, collection_name=Config.COLLECTION_NAME, vector_dim=Config.VECTOR_DIM, batch_size=Config.BATCH_SIZE):
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.vector_dim = vector_dim
        self.client: Optional[QdrantClient] = None
        
    def define_client(self):
        try:
            logger.info(f"Connecting to Qdrant at {Config.QDRANT_URL}")
            self.client = QdrantClient(url=Config.QDRANT_URL, prefer_grpc=True)
        except Exception as e:
            logger.error(f"Error connecting to Qdrant: {e}")
            raise
        
    def create_collection(self):
        if not self.client:
            self.define_client()
            
        try:
            if not self.client.collection_exists(collection_name=self.collection_name):
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_dim,
                        distance=models.Distance.DOT,
                        on_disk=True
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        default_segment_number=5,
                        indexing_threshold=0
                    ),
                    quantization_config=models.BinaryQuantization(
                        binary=models.BinaryQuantizationConfig(always_ram=True)
                    ),
                )
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
            
    def ingest_data(self, embeddata):
        if not self.client:
            self.define_client()
            
        try:
            for batch_context, batch_embeddings in zip(
                batch_iterate(embeddata.contexts, self.batch_size), 
                batch_iterate(embeddata.embeddings, self.batch_size)
            ):
                self.client.upload_collection(
                    collection_name=self.collection_name,
                    vectors=batch_embeddings,
                    payload=[{"context": context} for context in batch_context]
                )

            self.client.update_collection(
                collection_name=self.collection_name,
                optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000)
            )
            logger.info(f"Ingested {len(embeddata.embeddings)} points into {self.collection_name}")
        except Exception as e:
            logger.error(f"Error ingesting data: {e}")
            raise
        
class Retriever:
    def __init__(self, vector_db, embeddata):
        self.vector_db = vector_db
        self.embeddata = embeddata

    def search(self, query, limit=3):
        try:
            query_embedding = self.embeddata.embed_model.get_query_embedding(query)
            
            result = self.vector_db.client.search(
                collection_name=self.vector_db.collection_name,
                query_vector=query_embedding,
                search_params=models.SearchParams(
                    quantization=models.QuantizationSearchParams(
                        ignore=False,
                        rescore=True,
                        oversampling=2.0,
                    )
                ),
                limit=limit,
                timeout=1000,
            )
            return result
        except Exception as e:
            logger.error(f"Error searching in vector DB: {e}")
            raise
    
class RAG:
    def __init__(self, retriever, llm_name=Config.LLM_NAME):
        system_msg = ChatMessage(
            role=MessageRole.SYSTEM,
            content="You are a helpful assistant that answers questions about the user's document.",
        )
        self.messages = [system_msg, ]
        self.llm_name = llm_name
        self.llm = self._setup_llm()
        self.retriever = retriever
        self.qa_prompt_tmpl_str = (
            "Context information is below.\n"
            "---------------------\n"
            "{context}\n"
            "---------------------\n"
            "Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
            "Query: {query}\n"
            "Answer: "
        )

    def _setup_llm(self):
        try:
            logger.info(f"Setting up LLM: {self.llm_name}")
            return SambaNovaCloud(
                model=self.llm_name,
                temperature=0.7,
                context_window=100000,
                api_key=Config.SAMBANOVA_API_KEY
            )
        except Exception as e:
            logger.error(f"Error setting up LLM: {e}")
            raise

    def generate_context(self, query):
        try:
            result = self.retriever.search(query)
            context = [dict(data) for data in result]
            combined_prompt = []

            for entry in context:
                context_text = entry["payload"]["context"]
                combined_prompt.append(context_text)

            return "\n\n---\n\n".join(combined_prompt)
        except Exception as e:
            logger.error(f"Error generating context: {e}")
            raise

    def query(self, query):
        try:
            context = self.generate_context(query=query)
            prompt = self.qa_prompt_tmpl_str.format(context=context, query=query)
            return self.llm.stream_complete(prompt)
        except Exception as e:
            logger.error(f"Error during query: {e}")
            raise

class Transcribe:
    def __init__(self, api_key: str):
        aai.settings.api_key = api_key
        self.transcriber = aai.Transcriber()
        
    def transcribe_audio(self, audio_path: str, speakers_expected: int = 2) -> List[Dict[str, str]]:
        try:
            logger.info(f"Transcribing audio: {audio_path}")
            config = aai.TranscriptionConfig(
                speaker_labels=True,
                speakers_expected=speakers_expected
            )
            
            transcript = self.transcriber.transcribe(audio_path, config=config)
            
            if transcript.error:
                raise Exception(f"AssemblyAI Error: {transcript.error}")

            speaker_transcripts = []
            for utterance in transcript.utterances:
                speaker_transcripts.append({
                    "speaker": f"Speaker {utterance.speaker}",
                    "text": utterance.text
                })
            logger.info(f"Transcription complete. Found {len(speaker_transcripts)} utterances.")
            return speaker_transcripts
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            raise