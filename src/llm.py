from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt


class LLM:
    def __init__(self, model: str):
        self.client = OpenAI()
        self.model = model

    @retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5), reraise=True)
    def generate_text(self, system_prompt: str, prompt: str, temperature: float, max_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
