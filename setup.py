from setuptools import setup, find_packages

setup(
    name="auto_reg",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "litellm",
        "linearmodels",
        "pandas",
        "langchain",
        "langchain-openai",
        "langsmith",
        "openai",
    ],
    author="Shiyuan Li",
    author_email="lizhicuo2020@gmail.com",
    description="Tools for automatic regression, table generation, and analysis.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
) 