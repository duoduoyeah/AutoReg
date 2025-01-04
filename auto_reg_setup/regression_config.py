from pydantic import Field, BaseModel, AfterValidator
import pandas as pd
from typing import Annotated, List


regression_models: dict[str, str] = {
    "basic_regression": "panel data regression model",
    "robustness": "panel data regression model for robustness test",
    "mediating_effect": "panel data regression model for examining mediating effect",
    "moderating_effect": "panel data regression model for examining moderating effect", 
    "heterogeneity": "panel data regression model for examining heterogeneity",
    "endogeneity": "panel data regression model for examining endogeneity"
}

class BaseRegressionConfig(BaseModel):
    """Base configuration with common variables across all regressions"""
    dependent_vars: list[str]
    dependent_var_description: list[str] 
    independent_vars: list[str]
    independent_var_description: list[str]
    control_vars: list[str] | None = None
    control_vars_description: list[str] | None = None
    constant: bool = True


class RegressionConfig(BaseRegressionConfig):
    """Data class to store regression configurations"""
    regression_type: str = Field(default="basic_regression", description="Type of regression model")
    effects: list[str] | None = None

    # extra settings
    run_another_regression_without_controls: bool = False

    # extra variables
    instrument_var: str | None = None
    instrument_var_description: str | None = None
    group_var: str | None = None
    group_var_description: str | None = None

    @classmethod
    def create_with_base(cls, 
                        base_config: BaseRegressionConfig, 
                        **kwargs) -> 'RegressionConfig':
        """Factory method to create RegressionConfig with base configuration"""
        return cls(
            dependent_vars=base_config.dependent_vars,
            dependent_var_description=base_config.dependent_var_description,
            independent_vars=base_config.independent_vars,
            independent_var_description=base_config.independent_var_description,
            **kwargs
        )
    
    def __str__(self) -> str:
        """String representation of RegressionConfig, excluding None values"""
        attributes = []
        for key, value in self.__dict__.items():
            if value is not None:
                attributes.append(f"{key}: {value}")
        return "\n".join(attributes)


class ResearchConfig(BaseModel):
    """Data class to store research configurations
    
    All variables in this class are lists of strings, which allows them to be concatenated together when needed.
    For example, you can combine control_vars with extra_control_vars to create a larger list of control variables.
    
    effects: list[str] | None = None # fixed effects used for all regression except robustness test by adding extra control variables
    effects at most 2 effects. effect must use "entity" and "time" when denote index vars.

    extra_effects: list[str] | None = None # additional fixed effects to include in regression
    extra_effects at most 2 effects. When use extra_effects, the basic regression effects are ommited.
    effect must use "entity" and "time" when denote index vars.

    """
    # research topic
    research_topic: str = Field(description="research topic", default="")
    
    # core variables
    dependent_vars: list[str] | None = None
    dependent_var_description: list[str] | None = None
    independent_vars: list[str] | None = None
    independent_var_description: list[str] | None = None

    # control variables
    control_vars: list[str] | None = None
    control_vars_description: list[str] | None = None

    # instrument variables
    instrument_vars: list[str] | None = None
    instrument_vars_description: list[str] | None = None

    # group variables for heterogeneity analysis
    group_vars: list[str] | None = None
    group_vars_description: list[str] | None = None

    # mediating variables
    mediating_vars: list[str] | None = None
    mediating_vars_description: list[str] | None = None

    # supplementary variables for robustness test
    extra_control_vars: list[str] | None = None
    extra_control_vars_description: list[str] | None = None
    
    extra_effects: list[str] | None = None 
    extra_effects_vars: list[str] | None = None

    replacement_x_vars: list[str] | None = None
    replacement_x_vars_description: list[str] | None = None
    replacement_y_vars: list[str] | None = None
    replacement_y_vars_description: list[str] | None = None

    # basic regression settings
    effects: list[str] | None = None  
    effects_vars: list[str] | None = None 

    constant: bool = True
    regression_with_wihout_controls: list[str] = Field(
        description="if list here, will run another regression without controls",
        default=["basic_regression"]
    )

    def _all_vars(self) -> list[str]:
        """Return all variables in the research config"""
        return self.dependent_vars + \
                self.independent_vars + \
                self.control_vars + \
                self.instrument_var + \
                self.group_vars + \
                self.mediating_var + \
                self.extra_control_vars + \
                self.replacement_x_vars + \
                self.replacement_y_vars + \
                self.effects_vars + \
                self.extra_effects_vars


    def validate_research_config(self, df: pd.DataFrame) -> None:
        """Validate the research config"""
        # Check variables are defined
        if self.dependent_vars is None or self.independent_vars is None:
            raise ValueError("Dependent and independent variables are required")
        if self.control_vars is not None:
            raise ValueError("Control variables are required")
        if self.effects is not None:
            raise ValueError("Fixed effects are required")
        
        #check variables are in the dataframe
        for var in self._all_vars():
            if var not in df.columns:
                raise ValueError(f"Don't have variable {var} in the dataframe")


    def generate_regression_configs(self) -> dict[str, list[RegressionConfig]]:
        """
        Generate a dictionary of regression configurations based on the research config.
        
        Args:
            research_config: ResearchConfig object containing research configuration
            
        Returns:
            Dictionary mapping regression type strings to RegressionConfig objects
        """
        # the key of configs is to describe the type of regression.
        # the value of configs is a list of RegressionConfig objects.
        configs: dict[str, list[RegressionConfig]] = {}
        # Configuration Base

        base_config = BaseRegressionConfig(
            dependent_vars=self.dependent_vars,
            dependent_var_description=self.dependent_var_description,
            independent_vars=self.independent_vars,
            independent_var_description=self.independent_var_description,
            control_vars=self.control_vars,
            control_vars_description=self.control_vars_description,
            constant=self.constant,
        )

        # Basic regression config
        basic_regression_config = RegressionConfig.create_with_base(
            base_config,
            regression_type="basic regression",
            effects=self.effects_vars,
            run_another_regression_without_controls=self.regression_with_wihout_controls,
        )
        
        if self.regression_with_wihout_controls is not None:
            regression_types = f"Two Basic regressions.with and without controls"
        else:
            regression_types = "One Basic regression"

        configs[regression_types].append(basic_regression_config)

        # Robustness test configs

        # - Alternative measures of dependent/independent variables
        if self.replacement_x_vars is not None:
            for i, x_var in enumerate(self.replacement_x_vars):
                temp_config = RegressionConfig.create_with_base(
                    base_config,
                    regression_type="robustness",
                    effects=self.effects_vars,
                )
                temp_config.independent_vars = [x_var]
                temp_config.independent_var_description = [self.replacement_x_vars_description[i]]
                configs[f"robustness test - alternative independent variable: {x_var}"].append(temp_config)

        if self.replacement_y_vars is not None:
            for i, y_var in enumerate(self.replacement_y_vars):
                temp_config = RegressionConfig.create_with_base(
                    base_config,
                    regression_type="robustness",
                    effects=self.effects_vars,
                )
                temp_config.dependent_vars = [y_var]
                temp_config.dependent_var_description = [self.replacement_y_vars_description[i]]
                configs[f"robustness test - alternative dependent variable: {y_var}"].append(temp_config)

        # - Alternative fixed effects specifications
        if self.extra_effects_vars is not None:
            temp_config = RegressionConfig.create_with_base(
                base_config,
                regression_type="robustness",
                effects=self.extra_effects_vars,
            )

            configs[f"robustness test - alternative fixed effects: {self.extra_effects_vars}"].append(temp_config)

        # robustness test by adding extra control variables
        if self.extra_control_vars is not None:
            temp_config = RegressionConfig.create_with_base(
                base_config,
                regression_type="robustness",
                effects=self.effects_vars,
            )
            temp_config.control_vars.extend(self.extra_control_vars)
            temp_config.control_vars_description.extend(self.extra_control_vars_description)
            configs[f"robustness test - adding extra control variables: {self.extra_control_vars}"].append(temp_config)


        # Endogeneity test config 
        # - Instrumental variables regression (2SLS)
        if self.instrument_vars is not None:
            for i, instrument_var in enumerate(self.instrument_vars):
                temp_config = RegressionConfig.create_with_base(
                    base_config,
                    regression_type="endogeneity",
                    effects=self.effects_vars,
                    instrument_var=instrument_var,
                    instrument_var_description=self.instrument_vars_description[i],
                )
                configs[f"2SLS endogeneity test - instrument variables: {instrument_var}"].append(temp_config)

        # Mediating effect config
        if self.mediating_vars is not None:
            for i, mediating_var in enumerate(self.mediating_vars):
                temp_config = RegressionConfig.create_with_base(
                    base_config,
                    regression_type="mediating_effect",
                    effects=self.effects_vars,
                )
                temp_config.dependent_vars = [mediating_var]
                temp_config.dependent_var_description = [self.mediating_vars_description[i]]

                configs[f"mediating effect test by \
                        test the correlation between \
                        independent variables and mediating variables: {mediating_var}"].append(temp_config)


        # Moderating effect config
        # TODO: Add configs for:
        # - Interaction terms



        # Heterogeneity analysis config
        if self.group_vars is not None:
            for i, group_var in enumerate(self.group_vars):
                temp_config = RegressionConfig.create_with_base(
                    base_config,
                    regression_type="heterogeneity",
                    effects=self.effects_vars,
                    group_var=group_var,
                    group_var_description=self.group_vars_description[i],
                )
                configs[f"heterogeneity test by group variable: {group_var}"].append(temp_config)

        return configs

    def __str__(self) -> str:
        """String representation of RegressionConfig, excluding None values"""
        attributes = []
        for key, value in self.__dict__.items():
            if value is not None:
                attributes.append(f"{key}: {value}")
        return "\n".join(attributes)

if __name__ == '__main__':
    research_config = ResearchConfig(
        research_topic="abnormal stock return",
        dependent_vars=["stock_revenue"],
        dependent_var_description=["abnormal stock return"],
        independent_vars=["extreme_temperature"],
        independent_var_description=["the times of extreme weather events of the year"],
    )

    print(str(research_config))