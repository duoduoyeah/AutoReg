from linearmodels.panel.results import PanelEffectResults

# Using LM JSON response to set up the regression setting
user_description: dict = {}

def languagemodel_process():
    pass

reg_setting: dict = {}

# Run regression
def run_reg(reg_setting: dict) -> PanelEffectResults:
    pass

# Store the results
reg_results: list[PanelEffectResults] = []

# convert results to latex format table
def results_to_latex(results: list[PanelEffectResults]) -> str:
    pass

# convert results to markdown format table
def results_to_markdown(results: list[PanelEffectResults]) -> str:
    pass

#  convert the latex tables to words format
def latex_to_words(latex_table: str) -> str:
    pass

#  Analyze the results by LM
def analyze_results(results: list[PanelEffectResults]) -> str:
    pass