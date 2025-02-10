from ..errors import ChainError, DataClassError
from typing import Optional

async def run_chain(chain, 
              query,
              output_data,
              chain_purpose: Optional[str] = None):
    """
    Ainvoke the chain and return the output.
    Raise ChainInvocationError if the chain is not properly configured.
    Raise DataClassError if the output is not properly validated.
    """
    try:
        output = await chain.ainvoke({"query": query})
    except Exception:
        extra_info = {
            "chain_purpose": chain_purpose,
            "query": query,
            "output": output_data,
        }
        raise ChainError(extra_info=extra_info)

    try:
        output = output_data.model_validate(output)
    except Exception:
        extra_info = {
            "chain_purpose": chain_purpose,
            "query": query,
            "output": output_data,
        }
        raise DataClassError(extra_info=extra_info)
    
    return output