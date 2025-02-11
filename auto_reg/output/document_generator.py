from pylatex import Document, Section, Subsection
from auto_reg.analysis.models import ResultTables
import pandoc
from ..errors import OutputFileError

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

def create_tex(
    tables: ResultTables,
    with_analysis: bool = True,
):
    try:
        doc = Document("ctexart")

        fill_document(
            doc,
            tables,
            with_analysis,
        )
        
        return doc
    except Exception as e:
        raise OutputFileError(extra_info={"extra_info": str(e)})
    
def gemerate_tex(
    doc,
    filepath: str,
):
    if filepath.endswith(".tex"):
        filepath = filepath.replace(".tex", "")

    try:
        doc.generate_tex(filepath)
    except Exception as e:
        raise OutputFileError(
            extra_info={
                "error": "The tex file failed to generate",
                "extra_info": str(e),
            })

def generate_pdf(
    doc, 
    filepath: str,
):
    # removethe .pdf extension
    if filepath.endswith(".pdf"):
        filepath = filepath.replace(".pdf", "")
    try:
        doc.generate_pdf(filepath, clean_tex=False, compiler="xelatex")
    except Exception as e:
        raise OutputFileError(
            extra_info={
                "error": "The PDF file failed to generate",
                "extra_info": str(e),
            })


def generate_word(
    doc,
    filepath: str,
):
    # add the .docx extension
    if not filepath.endswith(".docx"):
        filepath = filepath + ".docx"

    try:
        doc_word = pandoc.write(doc, format="docx")
        with open(filepath, "wb") as f:
            f.write(doc_word)
    except Exception as e:
        raise OutputFileError(
            extra_info={
                "error": "The Word file failed to generate",
                "extra_info": str(e),
            })