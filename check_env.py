import logging
from config import Config

# Configure logging to write to file
logging.basicConfig(
    filename='env_check_result.txt',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    logger.info("Starting environment check...")
    
    is_valid, error_msg = Config.validate()
    
    if is_valid:
        logger.info("Environment check PASSED.")
        print("Environment check PASSED. See env_check_result.txt for details.")
    else:
        logger.error(f"Environment check FAILED: {error_msg}")
        print(f"Environment check FAILED: {error_msg}")
        
    # Additional info
    logger.info(f"Using Embedding Model: {Config.EMBED_MODEL_NAME}")
    logger.info(f"Using LLM: {Config.LLM_NAME}")
    logger.info(f"Qdrant URL: {Config.QDRANT_URL}")
    logger.info(f"Collection Name: {Config.COLLECTION_NAME}")

if __name__ == "__main__":
    check_environment()
