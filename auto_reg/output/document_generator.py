from pylatex import Command, Document, Section, Subsection
from pylatex.utils import NoEscape
from auto_reg.analysis.models import ResultTables
import pandoc

def fill_document(doc, tables, with_analysis):
    """Add a section, a subsection and some text to the document.

    :param doc: the document
    :type doc: :class:`pylatex.document.Document` instance
    """
    with doc.create(Section("Regression Analysis Part")):
        for table, description, analysis in tables.iterate_table():
            with doc.create(Subsection(description)):
                if with_analysis:
                    doc.append(analysis.latex_analysis)
                doc.append(table.latex_table)

def generate_latex(
    tables: ResultTables,
    with_analysis: bool = True,
    chinese: bool = False,
):
    if chinese:
        doc = Document("ctexart")
    else:
        doc = Document("article")
    fill_document(
        doc,
        tables,
        with_analysis,
    )
    
    return doc
    
def generate_pdf(
    doc, 
    filepath: str,
):
    # removethe .pdf extension
    if filepath.endswith(".pdf"):
        filepath = filepath.replace(".pdf", "")

    doc.generate_pdf(filepath, clean_tex=False, compiler="xelatex")


def generate_word(
    doc,
    filepath: str,
):
    # add the .docx extension
    if not filepath.endswith(".docx"):
        filepath = filepath + ".docx"

    doc_word = pandoc.write(doc, format="docx")
    with open(filepath, "wb") as f:
        f.write(doc_word)