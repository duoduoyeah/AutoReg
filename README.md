# Auto Reg

Auto Reg is a Python module for [brief description of what your module does].

## Document
[Document here](https://salty-impatiens-ec1.notion.site/AutoReg-1755b94d731880ea8334e43110610f27)

## Installation

To install Auto Reg, you can clone the repository and install the required packages using pip.
```bash
# Clone the repository
git clone https://github.com/yourusername/auto_reg.git

# Navigate to the project directory
cd auto_reg

# Install the required packages
pip install .
# or
python setup.py install
```

## Uasage(Temporary)
Use the `examples/example.py` file as a template. You can ask a Language Model with the following prompt: "What modifications should I make to run this file as a user?"  

### 1. Add a .env file in the root directory.
```bash
cd auto_reg
touch .env
```
### 2. Add your OpenAI API key and base URL to the .env file.

### 3. modify the `examples/research_config.json` file to follow your research topic.
For each XX_vars entry in the JSON file, ensure that the corresponding variable name matches exactly with the column name in the CSV file. Otherwise, the program will not recognize the variable.


### 4. run the example.py file.

```bash
python examples/example.py
```

## Contributing
If any bug is found, please submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
