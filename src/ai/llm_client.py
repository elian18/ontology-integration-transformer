import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class LLMClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ["LLM_API_KEY"],
            base_url=os.environ["LLM_BASE_URL"],
        )
        self.model = os.environ["LLM_MODEL"]

    def ask(self, prompt: str, context: str = "", max_tokens: int = 1024) -> str:
        system = "Responde únicamente con base en el contexto entregado."
        content = f"Contexto:\n{context}\n\nPregunta:\n{prompt}" if context else prompt
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
        )
        return resp.choices[0].message.content