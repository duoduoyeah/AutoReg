class Messages:
    """User-facing messages"""
    DATAFILEFAILED = """
    The input csv file has problem.
    Possible senarios:
        1. the data path is incorrect.
        2. the data index is mispelled or incorrect.
        3. the data file has problem.
    """
    MISSING_ENVIRONMENT_VARIABLES = """
    The Language Model object is not correctly set up.
    Possible senarios:
        1. API or API base error.
    """
    
    JSONFILEERROR = """
    The input json file has porblem.
    Possible senarios:
        1. the json file path is incorrect.
        2. the json file doesn't match with the proper format.
    """
    
    CHAIN_INVOCATION_ERROR = """
    The langchain chain has problem when invoking the model.
    Possible senarios:
        1. the chain is not properly configured
    """
    
    DATACLASSERROR = """
    The pydantic data class has problem validating the output of the language model.
    Possible senarios:
        1. the data class itself.
        2. the input prompt is not good.
        3. The language model can't generate the output in the proper format.
    """

    CHAIN_CONFIGURATION_ERROR = """
    The langchain chain is not properly configured.
    Possible senarios:
        1. langchain parse error
        2. the query is not formatted correctly.
        3. the prompt is not formatted correctly.
    """

    DESIGNERROR = """
    The table design is invalid.
    """

    RESULTTABLEERROR = """
    The result table has inconsistent data.
    Possible scenarios:
        1. The lengths of tables, descriptions, and analysis lists are not equal.
    """
    
    OUTPUTFILEERROR = """
    The output file failed to generate.
    Possible scenarios:
        1. The output file path is incorrect.
        2. The output file convertion failed.
    """
