
import unittest
from auto_reg.analysis.models import TableDesign
from auto_reg.analysis.design import validate_design_regression_tables
from auto_reg.errors import DesignError

class TestTableGeneration(unittest.TestCase):

    def test_valid_table_design_with_exact_indices(self):
        """Test table design validation with exact number of indices matching results"""
        output = TableDesign(
            number_of_tables=2,
            table_index=[[0, 1], [2, 3]],
            table_regression_nums=[],
            table_title=[],
        )
        number_of_results = 4
        self.assertIsNone(validate_design_regression_tables(output, number_of_results))

    def test_invalid_table_design_with_insufficient_results(self):
        """Test table design validation fails when indices exceed available results"""
        output = TableDesign(
            number_of_tables=3,
            table_index=[[0, 1], [2, 3], [4, 5]],
            table_regression_nums=[],
            table_title=[],
        )
        number_of_results = 4
        with self.assertRaises(DesignError):
            validate_design_regression_tables(output, number_of_results)

    def test_valid_table_design_with_sufficient_results(self):
        """Test table design validation with sufficient results for all indices"""
        output = TableDesign(
            number_of_tables=3,
            table_index=[[0, 1], [2, 3], [4, 5]],
            table_regression_nums=[],
            table_title=[],
        )
        number_of_results = 6
        self.assertIsNone(validate_design_regression_tables(output, number_of_results))

    def test_invalid_table_design_with_duplicate_indices(self):
        """Test table design validation fails when indices are duplicated"""
        output = TableDesign(
            number_of_tables=3,
            table_index=[[0, 1], [2, 1], [4, 5], [3]],
            table_regression_nums=[],
            table_title=[],
        )
        number_of_results = 6
        with self.assertRaises(DesignError):
            validate_design_regression_tables(output, number_of_results)

    def test_valid_table_design_with_single_result_tables(self):
        """Test table design validation with mix of single and multiple result tables"""
        output = TableDesign(
            number_of_tables=7,
            table_index=[[0], [1, 2], [3, 4], [5], [6], [7], [8]],
            table_regression_nums=[],
            table_title=[],
        )
        number_of_results = 9
        self.assertIsNone(validate_design_regression_tables(output, number_of_results))

    def test_invalid_table_design_with_out_of_order_indices(self):
        """Test table design validation fails with out of order indices"""
        output = TableDesign(
            number_of_tables=5,
            table_index=[[0], [5], [7], [8], [1, 2, 3, 4, 6]],
            table_regression_nums=[],
            table_title=[],
        )
        number_of_results = 9
        with self.assertRaises(DesignError):
            validate_design_regression_tables(output, number_of_results)

    def test_valid_table_design_with_mixed_table_sizes(self):
        """Test table design validation with varying number of results per table"""
        output = TableDesign(
            number_of_tables=6,
            table_index=[[0], [1, 2], [3, 4], [5], [6], [7, 8]],
            table_regression_nums=[],
            table_title=[],
        )
        number_of_results = 9
        self.assertIsNone(validate_design_regression_tables(output, number_of_results))

    def test_invalid_table_design_mismatched_table_count(self):
        """Test table design validation fails when table count doesn't match indices"""
        output = TableDesign(
            number_of_tables=5,
            table_index=[[0], [1, 2], [3, 4], [5], [6], [7, 8]],
            table_regression_nums=[],
            table_title=[],
        )
        number_of_results = 9
        with self.assertRaises(DesignError):
            validate_design_regression_tables(output, number_of_results)

    def test_invalid_table_design_with_incorrect_table_count(self):
        """Test table design validation fails when declared table count is wrong"""
        output = TableDesign(
            number_of_tables=7,
            table_index=[[0], [1, 2], [3, 4], [5], [6], [7, 8]],
            table_regression_nums=[],
            table_title=[],
        )
        number_of_results = 9
        with self.assertRaises(DesignError):
            validate_design_regression_tables(output, number_of_results)


if __name__ == "__main__":
    unittest.main()
