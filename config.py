import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    # API Keys
    ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
    SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")
    
    # Models
    EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-large-en-v1.5")
    LLM_NAME = os.getenv("LLM_NAME", "DeepSeek-R1-Distill-Llama-70B")
    
    # Vector DB
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "chat_with_audios")
    VECTOR_DIM = int(os.getenv("VECTOR_DIM", "1024"))
    
    # App Settings
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
    CACHE_DIR = os.getenv("CACHE_DIR", "./hf_cache")

    @classmethod
    def validate(cls):
        missing = []
        if not cls.ASSEMBLYAI_API_KEY:
            missing.append("ASSEMBLYAI_API_KEY")
        if not cls.SAMBANOVA_API_KEY:
            missing.append("SAMBANOVA_API_KEY")
        
        if missing:
            error_msg = f"Missing environment variables: {', '.join(missing)}"
            logger.error(error_msg)
            return False, error_msg
        
        return True, ""
