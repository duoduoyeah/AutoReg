import unittest
from auto_reg.output.tex_convertor import convert_to_latex

class TestTexConvertor(unittest.TestCase):
    def test_convert_normal_string(self):
        test_str = "This is a normal string"
        result = convert_to_latex(test_str)
        self.assertEqual(result, test_str)

    def test_convert_percent_string(self):
        test_str = "Value increased by 20%"
        expected = r"Value increased by 20\%"
        result = convert_to_latex(test_str)
        self.assertEqual(result, expected)

    def test_convert_escaped_percent(self):
        test_str = r"Already escaped \% should not change"
        result = convert_to_latex(test_str)
        self.assertEqual(result, test_str)

if __name__ == '__main__':
    unittest.main()
