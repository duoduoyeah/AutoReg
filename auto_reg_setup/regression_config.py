


from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RegressionConfig:
    """Data class to store regression configurations"""
    dependent_vars: Optional[str] = None
    dependent_var_description: Optional[str] = None
    independent_vars: Optional[str] = None
    independent_var_description: Optional[str] = None
    effects: list[str] = None 
    control_vars: list[str] = None
    control_vars_description: list[str] = None
    constant: bool = True
    other_vars: list[str] = None
    other_vars_description: list[str] = None

