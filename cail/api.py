import json
import random
import time
import urllib.error
import urllib.request


class ChatClient:
    def __init__(self, base_url, model, api_key, timeout=120, retries=4):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.api_key = model, api_key
        self.timeout, self.retries = timeout, retries

    def complete(self, messages, temperature=0.7, max_tokens=2048):
        payload = json.dumps({"model": self.model, "messages": messages, "temperature": temperature,
                              "max_tokens": max_tokens}, ensure_ascii=False).encode("utf-8")
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(self.url, data=payload, method="POST", headers={
                "Content-Type": "application/json", "Authorization": "Bearer " + self.api_key})
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    result = json.loads(response.read())
                choice = result["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise ValueError("输出被 max_tokens 截断")
                content = choice["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("API 返回空文本")
                return content
            except urllib.error.HTTPError as exc:
                if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == self.retries:
                    raise RuntimeError(f"API HTTP {exc.code}") from None
            except (urllib.error.URLError, TimeoutError):
                if attempt == self.retries:
                    raise RuntimeError("API 连接超时或不可达") from None
            time.sleep(min(30, 2 ** attempt) + random.random())
