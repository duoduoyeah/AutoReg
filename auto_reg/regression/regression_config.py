# data pipeline:
# dataframe and research config -> regression config

from pydantic import Field, BaseModel
import pandas as pd
from ..errors import JsonFileError

# TODO: this is not used
regression_models: dict[str, str] = {
    "basic_regression": "panel data regression model",
    "robustness": "panel data regression model for robustness test",
    "mediating_effect": "panel data regression model for examining mediating effect",
    "moderating_effect": "panel data regression model for examining moderating effect",
    "heterogeneity": "panel data regression model for examining heterogeneity",
    "endogeneity": "panel data regression model for examining endogeneity",
}


class BaseRegressionConfig(BaseModel):
    """Base configuration with common variables across all regressions"""

    dependent_vars: list[str] = []
    dependent_var_description: list[str] = []
    independent_vars: list[str] = []
    independent_var_description: list[str] = []
    control_vars: list[str] = []
    control_vars_description: list[str] = []
    constant: bool = True


class RegressionConfig(BaseRegressionConfig):
    """Data class to store regression configurations"""

    regression_type: str = Field(
        default="basic regression", description="Description of regression model"
    )
    effects: list[str] = []

    # extra settings
    run_another_regression_without_controls: bool = False

    # extra variables
    instrument_var: str = ""
    instrument_var_description: str = ""
    group_var: str = ""
    group_var_description: str = ""

    @classmethod
    def create_with_base(
        cls, base_config: BaseRegressionConfig, **kwargs
    ) -> "RegressionConfig":
        """Factory method to create RegressionConfig with base configuration"""
        return cls(
            dependent_vars=base_config.dependent_vars,
            dependent_var_description=base_config.dependent_var_description,
            independent_vars=base_config.independent_vars,
            independent_var_description=base_config.independent_var_description,
            control_vars=base_config.control_vars,
            control_vars_description=base_config.control_vars_description,
            constant=base_config.constant,
            **kwargs,
        )

    def __str__(self) -> str:
        """String representation of RegressionConfig, excluding None values and empty lists"""
        attributes = []
        for key, value in self.__dict__.items():
            if value:
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
    dependent_vars: list[str] = []
    dependent_var_description: list[str] = []
    independent_vars: list[str] = []
    independent_var_description: list[str] = []

    # control variables
    control_vars: list[str] = []
    control_vars_description: list[str] = []

    # instrument variables
    instrument_vars: list[str] = []
    instrument_vars_description: list[str] = []

    # group variables for heterogeneity analysis
    group_vars: list[str] = []
    group_vars_description: list[str] = []

    # mediating variables
    mediating_vars: list[str] = []
    mediating_vars_description: list[str] = []

    # supplementary variables for robustness test
    extra_control_vars: list[str] = []
    extra_control_vars_description: list[str] = []

    extra_effects: list[str] = []
    extra_effects_vars: list[str] = []

    replacement_x_vars: list[str] = []
    replacement_x_vars_description: list[str] = []
    replacement_y_vars: list[str] = []
    replacement_y_vars_description: list[str] = []

    # basic regression settings
    effects: list[str] = []
    effects_vars: list[str] = []

    constant: bool = True
    run_another_regression_without_controls: bool = True

    def _all_vars(self) -> list[str]:
        """Return all variables in the research config"""
        return (
            self.dependent_vars
            + self.independent_vars
            + self.control_vars
            + self.instrument_vars
            + self.group_vars
            + self.mediating_vars
            + self.extra_control_vars
            + self.replacement_x_vars
            + self.replacement_y_vars
        )

    def validate_research_config(self, df: pd.DataFrame) -> None:
        """Validate the research config"""
        # Check variables are defined
        if self.dependent_vars is None or self.independent_vars is None:
            raise JsonFileError(extra_info={"error": "Dependent and independent variables are required"})
        if self.control_vars is None:
            raise JsonFileError(extra_info={"error": "Control variables are required"})
        if self.effects is None:
            raise JsonFileError(extra_info={"error": "Fixed effects are required"})

        # check variables are in the dataframe
        for var in self._all_vars():
            if var not in df.columns:
                raise JsonFileError(extra_info={"error": f"Don't have variable {var} in the dataset"})

        if self.effects_vars:
            all_effects = (self.effects_vars or []) + (self.extra_effects_vars or [])
            for effect in all_effects:
                if effect not in df.index.names and effect not in df.columns:
                    raise JsonFileError(extra_info={"error": f"Don't have variable {effect} in the dataset"})

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
        configs: dict[str, RegressionConfig] = {}
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
            regression_type=f"basic regression, the dependent variable is: {self.dependent_vars}. The independent variable is: {self.independent_vars}",
            effects=self.effects,
            run_another_regression_without_controls=self.run_another_regression_without_controls,
        )

        if self.run_another_regression_without_controls:
            regression_description = f"Two Basic regressions.with and without controls"
        else:
            regression_description = "One Basic regression"

        configs[regression_description] = basic_regression_config

        # Robustness test configs

        # - Alternative measures of dependent/independent variables
        if self.replacement_x_vars is not None:
            for i, x_var in enumerate(self.replacement_x_vars):
                temp_config = RegressionConfig.create_with_base(
                    base_config,
                    regression_type="robustness",
                    effects=self.effects,
                )
                temp_config.independent_vars = [x_var]
                temp_config.independent_var_description = [
                    self.replacement_x_vars_description[i]
                ]
                regression_description = f"robustness test - alternative independent variable: {x_var} to replace the independent variable {self.independent_vars[0]}"
                configs[regression_description] = temp_config

        if self.replacement_y_vars is not None:
            for i, y_var in enumerate(self.replacement_y_vars):
                temp_config = RegressionConfig.create_with_base(
                    base_config,
                    regression_type="robustness",
                    effects=self.effects,
                )
                temp_config.dependent_vars = [y_var]
                temp_config.dependent_var_description = [
                    self.replacement_y_vars_description[i]
                ]
                regression_description = f"robustness test - alternative dependent variable: {y_var} to replace the dependent variable {self.dependent_vars[0]}"
                configs[regression_description] = temp_config

        # - Alternative fixed effects specifications
        if self.extra_effects is not None:
            temp_config = RegressionConfig.create_with_base(
                base_config,
                regression_type="robustness",
                effects=self.extra_effects,
            )
            regression_description = f"robustness test - alternative fixed effects: {self.extra_effects_vars} to replace the fixed effects {self.effects_vars}"
            configs[regression_description] = temp_config

        # robustness test by adding extra control variables
        if self.extra_control_vars is not None:
            temp_config = RegressionConfig.create_with_base(
                base_config,
                regression_type="robustness",
                effects=self.effects,
            )
            temp_config.control_vars.extend(self.extra_control_vars)
            temp_config.control_vars_description.extend(
                self.extra_control_vars_description
            )
            regression_description = f"robustness test - adding extra control variables: {self.extra_control_vars}"
            configs[regression_description] = temp_config

        # Endogeneity test config
        # - Instrumental variables regression (2SLS)
        if self.instrument_vars is not None:
            for i, instrument_var in enumerate(self.instrument_vars):
                temp_config = RegressionConfig.create_with_base(
                    base_config,
                    regression_type="endogeneity",
                    effects=self.effects,
                    instrument_var=instrument_var,
                    instrument_var_description=self.instrument_vars_description[i],
                )
                regression_description = f"2SLS endogeneity test - instrument variables: {instrument_var}. The explanatory variable is: {self.independent_vars[0]}. The explained variable is: {self.dependent_vars[0]}"
                configs[regression_description] = temp_config

        # Mediating effect config
        if self.mediating_vars is not None:
            for i, mediating_var in enumerate(self.mediating_vars):
                temp_config = RegressionConfig.create_with_base(
                    base_config,
                    regression_type="mediating_effect",
                    effects=self.effects,
                )
                temp_config.dependent_vars = [mediating_var]
                temp_config.dependent_var_description = [
                    self.mediating_vars_description[i]
                ]
                regression_description = f"mediating effect test by test the correlation between independent variables and mediating variables. The mediating variable is: {mediating_var}. The independent variable is: {self.independent_vars[0]}."
                configs[regression_description] = temp_config

        # Moderating effect config
        # TODO: Add configs for:
        # - Interaction terms

        # Heterogeneity analysis config
        if self.group_vars is not None:
            for i, group_var in enumerate(self.group_vars):
                temp_config = RegressionConfig.create_with_base(
                    base_config,
                    regression_type="heterogeneity",
                    effects=self.effects,
                    group_var=group_var,
                    group_var_description=self.group_vars_description[i],
                )
                configs[f"heterogeneity test by group variable: {group_var}"] = (
                    temp_config
                )

        return configs

    def __str__(self) -> str:
        """String representation of RegressionConfig, excluding None values and empty lists"""
        attributes = []
        for key, value in self.__dict__.items():
            if value:
                attributes.append(f"{key}: {value}")
        return "\n".join(attributes)


def get_descriptions(regression_config: dict[str, RegressionConfig]) -> dict[str, str]:
    """Get the descriptions of the regression config"""
    return {key: value.regression_type for key, value in regression_config.items()}


if __name__ == "__main__":
    research_config = ResearchConfig(
        research_topic="abnormal stock return",
        dependent_vars=["stock_revenue"],
        dependent_var_description=["abnormal stock return"],
        independent_vars=["extreme_temperature"],
        independent_var_description=["the times of extreme weather events of the year"],
    )

    print(str(research_config))
