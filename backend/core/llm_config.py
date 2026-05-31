from langchain_ollama.chat_models import ChatOllama

def get_local_llm():
    return ChatOllama(
        model="qwen2.5",
        # base_url="http://ollama:11434",
        base_url="http://host.docker.internal:11434",
        temperature=0.1
    )

import time

def wait_for_ollama(retries=20, delay=2):
    for i in range(retries):
        try:
            llm = get_local_llm()
            response = llm.invoke("Respond with a simple Yes if you can hear me.")
            if "yes" in response.content.lower():
                print("Ollama connection successful.")
                return
            else:
                print(f"Ollama responded unexpectedly: {response.content}. Retrying...")
        except Exception as e:
            print(f"Attempt {i+1}/{retries}: Error connecting to Ollama: {e}. Retrying in {delay} seconds...")
        time.sleep(delay)
    raise ConnectionError("Failed to connect to Ollama after multiple retries.")