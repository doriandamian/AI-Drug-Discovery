import logging
import time
from langchain_ollama.chat_models import ChatOllama
from core.config import OLLAMA_BASE_URL, MANAGER_MODEL

logger = logging.getLogger(__name__)

def get_local_llm(model: str = MANAGER_MODEL):
    return ChatOllama(
        model=model,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1
    )

def wait_for_ollama(retries=20, delay=2):
    for i in range(retries):
        try:
            llm = get_local_llm()
            response = llm.invoke("Respond with a simple Yes if you can hear me.")
            if "yes" in response.content.lower():
                logger.info("Ollama connection successful.")
                return
            else:
                logger.warning("Ollama responded unexpectedly: %s. Retrying...", response.content)
        except Exception as e:
            logger.warning("Attempt %d/%d: Error connecting to Ollama: %s. Retrying in %ds...", i + 1, retries, e, delay)
        time.sleep(delay)
    raise ConnectionError("Failed to connect to Ollama after multiple retries.")