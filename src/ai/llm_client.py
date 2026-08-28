import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

class LLMClient:
    def __init__(self):
        self.client = Anthropic(api_key=os.environ["LLM_API_KEY"])
        self.model = os.environ["LLM_MODEL"]

    def ask(self, prompt: str, context: str = "", max_tokens: int = 1024) -> str:
        system = "Responde únicamente con base en el contexto entregado."
        content = f"Contexto:\n{context}\n\nPregunta:\n{prompt}" if context else prompt
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")