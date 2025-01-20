import dotenv
from langchain_openai import ChatOpenAI
import os
import sys
from pathlib import Path
import unittest
sys.path.append(str(Path(__file__).resolve().parents[1]))
from auto_reg.regression.varable_config import *
from auto_reg.regression.panel_data import *
from auto_reg.regression.regression_config import *
from auto_reg.analysis.generate_table import *
from basic_data import setup_basic_data, get_research_topic
import pdb

class TestTableGeneration(unittest.TestCase):
    def setup(self, model_name: str = "gpt-4o"):

        dotenv.load_dotenv()
        if model_name == "gpt-4o":
            print("using gpt-4o")
            self.chat_model = ChatOpenAI(
                model_name=model_name,
                timeout=(45.0), # 45 seconds before timeout
                temperature=0
            )
        elif model_name == "deepseek-chat":
            print("using deepseek-chat")
            os.environ["OPENAI_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")
            os.environ["OPENAI_API_BASE"] = os.getenv("DEEPSEEK_API_BASE")
            self.chat_model = ChatOpenAI(
                model_name="deepseek-chat",
                # timeout=(5.0, 15.0),
                temperature=0
            )

        df, research_config = setup_basic_data()
        research_topic: str = get_research_topic(research_config)
        return df, research_topic, research_config
    
    def test_design_regression_tables(self):
        """
        Test designing regression tables
        """
        df, research_topic, research_config = self.setup()
        regression_results = run_regressions(df, 
                                            research_config.generate_regression_configs())

        table_design = design_regression_tables(
            research_topic, 
            regression_results, 
            self.chat_model)

        # print("\n".join([desc for desc, _, _ in regression_results]))
        print(type(table_design))
        print(table_design)

    def test_validate_design_regression_tables(self):
        """
        Test validating regression table design
        """
        output = TableDesign(
            number_of_tables=2,
            table_index=[[0, 1], [2, 3]],
            table_regression_nums=[],
            table_title=[]
        )
        number_of_results = 4
        self.assertTrue(validate_design_regression_tables(output, number_of_results))

        output = TableDesign(
            number_of_tables=3,
            table_index=[[0, 1], [2, 3], [4, 5]],
            table_regression_nums=[],
            table_title=[]
        )
        number_of_results = 4
        self.assertFalse(validate_design_regression_tables(output, number_of_results))

        output = TableDesign(
            number_of_tables=3,
            table_index=[[0, 1], [2, 3], [4, 5]],
            table_regression_nums=[],
            table_title=[]
        )
        number_of_results = 6
        self.assertTrue(validate_design_regression_tables(output, number_of_results))

        output = TableDesign(
            number_of_tables=3,
            table_index=[[0, 1], [2, 1], [4, 5], [3]],
            table_regression_nums=[],
            table_title=[]
        )
        number_of_results = 6
        self.assertFalse(validate_design_regression_tables(output, number_of_results))

    def test_validate_design_regression_tables_additional(self):
        """
        Additional test cases for validating regression table design
        """
        output = TableDesign(
            number_of_tables=7,
            table_index=[[0], [1, 2], [3, 4], [5], [6], [7], [8]],
            table_regression_nums=[],
            table_title=[]
        )
        number_of_results = 9
        self.assertTrue(validate_design_regression_tables(output, number_of_results))

        output = TableDesign(
            number_of_tables=5,
            table_index=[[0], [5], [7], [8], [1, 2, 3, 4, 6]],
            table_regression_nums=[],
            table_title=[]
        )
        number_of_results = 9
        self.assertFalse(validate_design_regression_tables(output, number_of_results))

        output = TableDesign(
            number_of_tables=6,
            table_index=[[0], [1, 2], [3, 4], [5], [6], [7, 8]],
            table_regression_nums=[],
            table_title=[]
        )
        number_of_results = 9
        self.assertTrue(validate_design_regression_tables(output, number_of_results))

        output = TableDesign(
            number_of_tables=5,
            table_index=[[0], [1, 2], [3, 4], [5], [6], [7, 8]],
            table_regression_nums=[],
            table_title=[]
        )
        number_of_results = 9
        self.assertFalse(validate_design_regression_tables(output, number_of_results))

        output = TableDesign(
            number_of_tables=7,
            table_index=[[0], [1, 2], [3, 4], [5], [6], [7, 8]],
            table_regression_nums=[],
            table_title=[]
        )
        number_of_results = 9
        self.assertFalse(validate_design_regression_tables(output, number_of_results))

    def not_test_draw_all_tables(self):
        """
        Test drawing all tables
        """
        async def run_test(
                write_to_file: bool = False, 
                user_select: bool = False):
            df, research_topic, research_config = self.setup()

            # run regressions
            regression_results = run_regressions(
                df, 
                research_config.generate_regression_configs()
            )

            table_design: TableDesign|None = await design_regression_tables(
                research_topic, 
                regression_results, 
                self.chat_model
            )

            if table_design is None:
                return
            

            # user select tables
            if user_select:
                table_design = select_table_design(table_design)
                print(table_design)
            
            # draw tables
            table_results = ResultTables()
            await draw_tables(
                regression_results,
                table_design,
                self.chat_model,
                table_results
            )

            await analyze_regression_results(
                regression_results,
                table_design,
                table_results,
                self.chat_model
            )

            # combine tables
            table_results: ResultTables = await combine_tables(table_results, table_design, self.chat_model)

            if True:
                with open("temp/tables_with_analysis.txt", "w") as f:
                    tables = table_results.tables
                    descriptions = table_results.description
                    analysis = table_results.analysis
                    for table, description, analysis in zip(tables, descriptions, analysis):
                        if table is not None:
                            f.write(description + "\n")
                            f.write(table.latex_table + "\n\n")
                            f.write(analysis.analysis + "\n\n")
        asyncio.run(run_test(write_to_file=True, user_select=True))

if __name__ == '__main__':
    test = TestTableGeneration()
    # test.test_design_regression_tables()
    # test.test_validate_design_regression_tables()
    # test.test_validate_design_regression_tables_additional()
    test.test_draw_all_tables()