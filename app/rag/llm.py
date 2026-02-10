from app.config import settings
import httpx

class LLM:
    def __init__(self):
        self.backend = settings.llm_backend

    async def _openai_chat(self, prompt: str) -> str:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY missing")
        headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
        payload = {
            "model": settings.openai_model,
            "messages": [
                {"role": "system", "content": "You are a concise, factual assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()

    def _mock(self, prompt: str) -> str:
        return "MOCK_ANSWER: " + (prompt[:240].replace("\n", " ") + ("..." if len(prompt) > 240 else ""))

    def generate(self, prompt: str) -> str:
        if self.backend == "mock":
            return self._mock(prompt)

        if self.backend == "openai":
            import asyncio
            return asyncio.run(self._openai_chat(prompt))

        raise ValueError(f"Unsupported LLM_BACKEND={self.backend}")
