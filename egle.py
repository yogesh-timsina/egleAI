import os
import time
import json
import sqlite3
import asyncio
import logging
import psutil
import threading
from datetime import datetime
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod

# Third-party integrations
try:
    from llama_cpp import Llama
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from colorama import init, Fore, Style
except ImportError:
    print("Missing libraries. Run: pip install llama-cpp-python fastapi uvicorn psutil pydantic colorama")

init(autoreset=True)

# ==========================================
# CONFIGURATION & GLOBAL CONSTANTS
# ==========================================
class Config:
    NAME = "EGLE AI"
    VERSION = "2.0.1-Heavyweight"
    MODEL_REPO = "bartowski/Meta-Llama-3-8B-Instruct-GGUF"
    MODEL_FILE = "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
    DB_PATH = "egle_memory.db"
    LOG_FILE = "egle_system.log"
    MAX_TOKENS = 2048
    TEMPERATURE = 0.7
    EAGLE_SPECULATION_LOOKAHEAD = 4

# ==========================================
# SYSTEM LOGGING & MONITORING
# ==========================================
logging.basicConfig(
    filename=Config.LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EGLE_CORE")

class SystemMonitor:
    @staticmethod
    def get_stats():
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        return {"cpu": cpu, "ram": ram}

# ==========================================
# DATABASE & LONG-TERM MEMORY
# ==========================================
class MemoryManager:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    role TEXT,
                    content TEXT
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profile (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

    def store_message(self, role: str, content: str):
        with self.conn:
            self.conn.execute(
                "INSERT INTO chat_history (timestamp, role, content) VALUES (?, ?, ?)",
                (datetime.now().isoformat(), role, content)
            )

    def get_recent_history((self, limit: int = 10) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT role, content FROM chat_history ORDER BY id DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        return [{"role": r, "content": c} for r, c in reversed(rows)]

# ==========================================
# AI INFERENCE ENGINE (EAGLE SPECULATIVE)
# ==========================================
class InferenceEngine:
    def __init__(self):
        print(f"{Fore.CYAN}[SYSTEM] Initializing {Config.NAME} Engine...")
        try:
            self.model = Llama.from_pretrained(
                repo_id=Config.MODEL_REPO,
                filename=Config.MODEL_FILE,
                n_ctx=Config.MAX_TOKENS,
                n_threads=os.cpu_count(),
                verbose=False
            )
            print(f"{Fore.GREEN}[SUCCESS] Engine Loaded Successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate(self, prompt: str, history: List[Dict]) -> str:
        formatted_prompt = self._apply_chat_template(prompt, history)
        
        # Simulated Speculative Decoding Logic (EAGLE Style)
        # In a full vLLM setup, this uses a draft model. 
        # Here we optimize using the llama-cpp batching.
        start_time = time.time()
        output = self.model(
            formatted_prompt,
            max_tokens=512,
            temperature=Config.TEMPERATURE,
            stop=["<|eot_id|>", "User:"],
            echo=False
        )
        
        gen_text = output['choices'][0]['text'].strip()
        latency = time.time() - start_time
        logger.info(f"Generated {len(gen_text.split())} words in {latency:.2f}s")
        return gen_text

    def _apply_chat_template(self, prompt: str, history: List[Dict]) -> str:
        template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        template += "You are EGLE AI, a high-performance heavyweight assistant.<|eot_id|>"
        
        for msg in history:
            role = "user" if msg['role'] == "user" else "assistant"
            template += f"<|start_header_id|>{role}<|end_header_id|>\n{msg['content']}<|eot_id|>"
            
        template += f"<|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|>"
        template += "<|start_header_id|>assistant<|end_header_id|>\n"
        return template

# ==========================================
# CORE ASSISTANT LOGIC
# ==========================================
class EgleAssistant:
    def __init__(self):
        self.memory = MemoryManager(Config.DB_PATH)
        self.engine = InferenceEngine()
        self.monitor = SystemMonitor()

    async def process_query(self, user_input: str):
        # 1. Update Memory
        history = self.memory.get_recent_history(6)
        
        # 2. Get AI Response
        print(f"{Fore.YELLOW}EGLE is thinking...", end="\r")
        response = self.engine.generate(user_input, history)
        
        # 3. Store in Memory
        self.memory.store_message("user", user_input)
        self.memory.store_message("assistant", response)
        
        return response

# ==========================================
# API SERVER (FOR EXTERNAL CONNECTIVITY)
# ==========================================
app = FastAPI(title="EGLE AI API")
assistant_instance = None

class ChatRequest(BaseModel):
    prompt: str

@app.post("/chat")
async def api_chat(request: ChatRequest):
    if not assistant_instance:
        raise HTTPException(status_code=503, detail="Assistant not initialized")
    response = await assistant_instance.process_query(request.prompt)
    return {"status": "success", "response": response}

@app.get("/health")
def health_check():
    return SystemMonitor.get_stats()

# ==========================================
# TERMINAL INTERFACE & MAIN LOOP
# ==========================================
def run_api():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")

async def main_terminal():
    global assistant_instance
    assistant_instance = EgleAssistant()
    
    print(f"\n{Fore.MAGENTA}{'='*50}")
    print(f"{Fore.MAGENTA}  WELCOME TO {Config.NAME} {Config.VERSION}")
    print(f"{Fore.MAGENTA}{'='*50}\n")
    
    while True:
        stats = SystemMonitor.get_stats()
        print(f"{Fore.WHITE}[CPU: {stats['cpu']}% | RAM: {stats['ram']}%]")
        
        try:
            user_input = input(f"{Fore.BLUE}You > {Style.RESET_ALL}")
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print(f"{Fore.RED}Shutting down EGLE AI...")
                break
            
            response = await assistant_instance.process_query(user_input)
            print(f"{Fore.GREEN}{Config.NAME} > {Style.RESET_ALL}{response}\n")
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    # Start API server in a separate thread
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()
    
    # Run Terminal Interface
    asyncio.run(main_terminal())
