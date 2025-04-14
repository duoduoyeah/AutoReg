from langchain_openai import ChatOpenAI
from ..errors import ChainInvocationError

async def basic_chat(
    prompt: str,
    model: ChatOpenAI,
    ) -> str:
    try:
        output = await model.ainvoke(prompt)
        return output.content
    except Exception:
        extra_info = {
            "prompt": prompt,
            "model": model,
        }
        raise ChainInvocationError(extra_info=extra_info)
    