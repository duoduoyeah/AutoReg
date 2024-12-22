from setuptools import setup, find_packages

setup(
    name="auto-reg-tools",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Add dependencies here
    ],
    author="Shiyuan Li",
    author_email="lizhicuo2020@gmail.com",
    description="Tools for automatic register setup and table generation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
) 