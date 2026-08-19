import json
import httpx
from fastapi import HTTPException
from app.config import get_settings

SYSTEM_PROMPT = """You are a helpful personal AI assistant that answers questions about the user's journal entries.
Only answer based on the context provided from their past entries. If the context doesn't contain
enough information to answer, say so honestly. When possible, mention the date of the entry
you are referencing.

Respond using markdown formatting where appropriate (code blocks, lists, bold, etc.)."""


class LLMClient:
    def __init__(self):
        settings = get_settings()
        self.base_url = settings.LLM_BASE_URL.rstrip("/")
        self.api_key = settings.LLM_API_KEY
        self.model = settings.LLM_MODEL

    async def chat_completion(self, messages: list[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1024,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except httpx.ConnectError:
            raise HTTPException(status_code=502, detail="Cannot connect to LLM service")
        except httpx.TimeoutException:
            raise HTTPException(status_code=502, detail="LLM service timed out")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"LLM service error: {e.response.status_code}")

    async def chat_completion_stream(self, messages: list[dict]):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1024,
            "stream": True,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data["choices"][0]["delta"]
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
        except httpx.ConnectError:
            raise HTTPException(status_code=502, detail="Cannot connect to LLM service")
        except httpx.TimeoutException:
            raise HTTPException(status_code=502, detail="LLM service timed out")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"LLM service error: {e.response.status_code}")


_llm_client = None


def get_llm() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
