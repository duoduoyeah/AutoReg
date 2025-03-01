from ..errors import ChainInvocationError, DataClassError
from typing import Optional

async def run_chain(chain, 
              query,
              output_data,
              chain_purpose: Optional[str] = None):
    """
    Inputs:
        chain: The chain to invoke.
        query: The query to pass to the chain.
        output_data: The data class to validate the output.
        chain_purpose: The purpose of the chain.
    Returns:
        The output of the chain.
    Raises:
        ChainInvocationError: If the chain is not properly configured.
        DataClassError: If the output is not properly validated.
    """
    try:
        output = await chain.ainvoke({"query": query})
    except Exception:
        extra_info = {
            "chain_purpose": chain_purpose,
            "query": query,
            "output": output_data,
        }
        raise ChainInvocationError(extra_info=extra_info)

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