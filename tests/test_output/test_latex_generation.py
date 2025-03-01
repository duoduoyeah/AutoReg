from auto_reg.output.document_generator import *
from auto_reg.analysis.models import ResultTables
from auto_reg.analysis.models import RegressionResultTable
from auto_reg.analysis.models import RegressionAnalysis

def case_1():
    result_tables = ResultTables()
    table = RegressionResultTable(latex_table="")
    table.latex_table = """
    \\begin{table}[htbp]\n\\caption{2SLS Endogeneity Test and Addition\\ of Extra Control Variables}\n\\label{Combined_Regression_Table}\n\\centering\n\\begin{tabular}{p{5cm}p{2.5cm}p{2.5cm}}\n\\toprule\n& (1) & (2) \\\\\nVariable Name  & Independent Variable & Dependent Variable \\\\\n\\midrule\nCPRI  & -0.0003 &  \\\\\n                & [-0.3586] &  \\\\\nInstrumental Variable (NCSKEW 2)  & -0.0813** &  \\\\\n                & (-2.4117) &  \\\\\nPredicted Independent variable (CPRI predicted)  &  & -2.9085*** \\\\\n                &  & (-34.285) \\\\\nSize  & -0.0062 & 0.0695*** \\\\\n                & [-1.4543] & (18.449) \\\\\nLev  & 0.0131 & 0.1771*** \\\\\n                & [0.4630] & (7.7369) \\\\\nROA  & 0.2084*** & -0.0528 \\\\\n                & [3.2986] & (-0.9935) \\\\\nROE  & -0.0002 & 0.0430*** \\\\\n                & [-0.1611] & (25.785) \\\\\nBM  & -0.0004 & -0.0033*** \\\\\n                & [-0.5486] & (-5.5904) \\\\\nPB  & -0.0001 & 0.0056*** \\\\\n                & [-0.5722] & (23.390) \\\\\nDturn  & -0.0223** & 0.0492*** \\\\\n                & [-2.8363] & (6.2438) \\\\\nTobinQ  & 0.0126** & \\\\\n                & [2.5526] &  \\\\\nExecutives  & 0.0210* & \\\\\n                & [1.6554] &  \\\\\nConstant    & -0.2320** &  \\\\\n                & [-2.3270] &  \\\\\nConstant    & 43.937*** &  \\\\\n                & (119.18) &  \\\\\nNumber of id  & 4,052 &  \\\\\nIndividual FE  & Yes &  \\\\\nYear FE  & Yes &  \\\\\nOther FE (PROVINCECODE) & Yes &  \\\\\nAll Effects        & Yes            & Yes     \\\\\nObservations  & 36,654         & 36,654 \\\\\nR-squared & 0.0031 & 0.0572 \\\\\n\\bottomrule\n\\end{tabular}\n\\begin{tablenotes}\n\\small\n\\item \\textit{Note:} t-statistics are in parentheses; *, **, *** denote significance at the 10\\%, 5\\%, and 1\\% levels, respectively.\n\\end{tablenotes}\n\\end{table}

    """
    analysis = RegressionAnalysis(latex_analysis="")
    analysis.latex_analysis = """
    This is a test analysis.
    """

    description = "This is a test description."

    result_tables.tables = [table]
    result_tables.analysis = [analysis]
    result_tables.description = [description]

    output_path = "/workspaces/AutoReg/examples/ping/output"
    import os
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    doc = create_tex(result_tables)
    generate_pdf(doc, output_path + "/test")

def case_2():
    result_tables = ResultTables()
    table = RegressionResultTable(latex_table="")
    table.latex_table = """this is a test table."""
    analysis = RegressionAnalysis(latex_analysis="")
    analysis.latex_analysis = """this is a test analysis. """
    description = "This is a test description.中文的"

    result_tables.tables = [table]
    result_tables.analysis = [analysis]
    result_tables.description = [description]
    
    output_path = "/workspaces/AutoReg/examples/ping/output"
    import os
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    doc = create_tex(result_tables)
    generate_tex(doc, output_path + "/test_2")

    generate_word(output_path + "/test_2.tex", output_path + "/test_2")
if __name__ == "__main__":
    case_1()