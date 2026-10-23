"""LLM 接口封装：默认本地 Ollama；接口抽象后便于换成 OpenAI 兼容 API。"""
import ollama


class OllamaLLM:
    def __init__(self, model, base_url=None, temperature=0.2, max_tokens=512):
        self.model = model
        self.client = ollama.Client(host=base_url) if base_url else ollama.Client()
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt, system=None):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self.client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": self.temperature, "num_predict": self.max_tokens},
        )
        return resp["message"]["content"]
