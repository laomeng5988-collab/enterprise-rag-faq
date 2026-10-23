"""LLM 接口封装：默认本地 Ollama；接口抽象后便于换成 OpenAI 兼容 API。"""
import ollama


class OllamaLLM:
    def __init__(self, model, base_url=None, temperature=0.2, max_tokens=512, num_gpu=None):
        self.model = model
        self.client = ollama.Client(host=base_url) if base_url else ollama.Client()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_gpu = num_gpu

    def generate(self, prompt, system=None):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        options = {"temperature": self.temperature, "num_predict": self.max_tokens}
        if self.num_gpu is not None:
            options["num_gpu"] = self.num_gpu  # 0=纯CPU；-1=自动
        resp = self.client.chat(
            model=self.model,
            messages=messages,
            options=options,
        )
        return resp["message"]["content"]
