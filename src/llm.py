from openai import AsyncOpenAI
from tenacity import retry, wait_exponential, stop_after_attempt
from langfuse import observe, propagate_attributes, get_client
from config import CONFIG
from logger import logger
from asyncio import Semaphore

langfuse = get_client()

class LLM:
    def __init__(self, model: str):
        self.client = AsyncOpenAI()
        self.model = model
        self.semaphore = Semaphore(32)

    @retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5), reraise=True)
    @observe(as_type="generation")
    async def generate_text(self, system_prompt: str, prompt: str, temperature: float, max_tokens: int) -> str:
        async with self.semaphore:
            with propagate_attributes(session_id=CONFIG["openai"]["model"]):
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body={
                        "usage": {
                            "include": True
                        }
                    }
                )

                langfuse.update_current_generation(
                    usage_details={
                        "input": response.usage.prompt_tokens,
                        "output": response.usage.completion_tokens,
                        },
                    cost_details={
                        "total": response.usage.cost
                    }
                )
                if response.choices[0].finish_reason != "stop":
                    logger.warning(f"LLM response finished unexpectedly. Reason: {response.choices[0].finish_reason}")
                return response.choices[0].message.content.strip()


llm = LLM(CONFIG["openai"]["model"])
