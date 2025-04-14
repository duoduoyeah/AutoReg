# This is a good test but not a test under framework
from auto_reg.output.document_generator import *
from auto_reg.analysis.models import ResultTables
from auto_reg.analysis.models import RegressionResultTable
from auto_reg.analysis.models import RegressionAnalysis

if __name__ == "__main__":
    output_path = "/workspaces/AutoReg/examples/ping/output"
    tables = ResultTables()
    tables.tables = [RegressionResultTable(latex_table="test table")]
    tables.analysis = [RegressionAnalysis(latex_analysis="test analysis")]
    tables.description = ["test description"]
    doc = create_tex(tables)
    generate_tex(doc, output_path + "/test.tex")
    latex_file = output_path + "/test.tex"
    generate_word(latex_file, output_path + "/test.pdf")
    generate_pdf(doc, output_path + "/test.docx")
    