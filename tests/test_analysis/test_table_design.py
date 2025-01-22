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

if __name__ == '__main__':
    test = TestTableGeneration()
    # test.test_design_regression_tables()
    # test.test_validate_design_regression_tables()
    # test.test_validate_design_regression_tables_additional()
    test.test_draw_all_tables()