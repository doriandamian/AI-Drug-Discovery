import os
import sys
import platform
import subprocess
import time
import urllib.request
import atexit

def run_command(command, shell=False, check=True):
    try:
        subprocess.run(command, shell=shell, check=check, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

def check_docker():
    print("Verifying Docker installation...")
    if not run_command(["docker", "--version"]):
        print("Docker is not installed. Please install Docker and try again.")
        sys.exit(1)
    if not run_command(["docker", "info"]):
        print("Docker is installed but not running. Please start Docker and try again.")
        sys.exit(1)
    print("Docker is installed and running.")

def setup_ollama():
    os_name = platform.system()
    print("Verifying Ollama installation...")
    
    if not run_command(["ollama", "--version"]):
        print("Ollama is not installed. Installing Ollama...")
        if os_name in ["Darwin", "Linux"]:
            os.system("curl -fsSL https://ollama.com/install.sh | sh")
        elif os_name == "Windows":
            print("Downloading Ollama installer for Windows...")
            urllib.request.urlretrieve("https://ollama.com/download/OllamaSetup.exe", "OllamaSetup.exe")
            print("Installing Ollama... Please confirm the installation windows.")
            os.system("OllamaSetup.exe")
        else:
            print(f"Unsupported operating system: {os_name}.")
            sys.exit(1)
    else:
        print("Ollama is already installed.")

    print("Starting Ollama server in background...")
    if os_name == "Windows":
        subprocess.Popen(["ollama", "serve"], creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)

def pull_models():
    models = ["qwen2.5:14b", "qwen2.5:7b", "nomic-embed-text"]
    for model in models:
        print(f"Verifying/Downloading model: {model} (this might take a few minutes on first run)...")
        os.system(f"ollama pull {model}")

def configure_environment():
    print("\nConfiguring environment variables for the network...")
    defaults = {"OLLAMA_BASE_URL": "http://host.docker.internal:11434"}

    lines = []
    if os.path.exists(".env"):
        with open(".env") as f:
            lines = f.read().splitlines()

    present = {
        line.split("=", 1)[0].strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#") and "=" in line
    }

    missing = [(k, v) for k, v in defaults.items() if k not in present]
    if missing:
        with open(".env", "a") as f:
            if lines and lines[-1].strip() != "":
                f.write("\n")
            for k, v in missing:
                f.write(f"{k}={v}\n")
        print("File .env updated (added missing defaults, existing values preserved).")
    else:
        print("File .env is ready (existing values preserved).")

def start_docker_compose():
    print("\nStarting Docker infrastructure (FastAPI & Neo4j)...")
    os.system("docker compose up --build -d")

def stop_docker_compose():
    print("\nStopping the system...")
    print("Shutting down Docker containers (please wait a few seconds)...")
    os.system("docker compose down")
    print("System stopped successfully and resources released")

atexit.register(stop_docker_compose)

def main():
    print("="*50)
    print("STARTING AI DRUG DISCOVERY ENVIRONMENT")
    print("="*50)
    
    check_docker()
    setup_ollama()
    pull_models()
    configure_environment()
    start_docker_compose()
    
    print("\n" + "="*50)
    print("SYSTEM IS FULLY OPERATIONAL")
    print("FastAPI backend is running at: http://localhost:8000")
    print("Neo4j database is running at: http://localhost:7474")
    print("="*50)
    print("ATENȚIE: Press Ctrl+C to stop the system and release resources.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Ctrl+C detected]")
        sys.exit(0)

if __name__ == "__main__":
    main()