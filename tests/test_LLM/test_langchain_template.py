from langchain_deepseek import ChatDeepSeek
from auto_reg.analysis.generate_table import analyze_regression_result
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import asyncio

from auto_reg.regression.regression_config import RegressionConfig

load_dotenv()

model_deepseek = ChatDeepSeek(
    model_name="deepseek-chat",
    temperature=0,
    timeout=(130.0),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_API_BASE"),
)


class AnalysisResult(BaseModel):
    analysis_result: str


regression_result = r"""
\begin{table}[htbp]
\caption{2SLS Endogeneity Test Using Instrumental Variable}
\label{2SLS_endogeneity_test}
\centering
\begin{tabular}{p{5cm}p{3cm}p{3cm}} % The sum of the width of the columns should be no more than 12cm
\toprule
& (1) & (2) \\
Variable Name  & extreme temperature & stock revenue \\
\midrule
Instrumental Variable: company latitude   & 0.0340*** & empty \\
                & (4.4059) &  \\
Predicted Independent Variable: extreme temperature predicted   & empty & -0.8668\\
                &  & (-0.7999) \\
Control Variable: company size     & 0.0041 & 0.1201*** \\
                & (0.5645) & (3.2104) \\
Control Variable: company age     & 0.0119 & 0.1580*** \\
                & (1.5450) & (5.2136) \\
Control Variable: company distance to sea     & -0.0069 & 0.0769** \\
                & (-1.0471) & (2.4707) \\
Control Variable: rain amount     & -0.0039 & 0.0865*** \\
                & (-0.5307) & (2.6404) \\
Control Variable: dry amount     & -0.0020 & 0.1135*** \\
                & (-0.2718) & (3.0760) \\
Constant    & 0.1136*** & empty \\
                & (6.881e+11) &  \\

Number of id       & 100        & 100 \\
Individual FE      & YES          & YES \\
Year FE            & YES          & YES \\
Observations       & 1,000       & 1,000 \\
R-squared          & 0.0294        & 0.0672 \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Note:} t-statistics are in parentheses; *, **, *** denote significance at the 10\%, 5\%, and 1\% levels, respectively.
\end{tablenotes}
\end{table}
"""

regression_config = RegressionConfig(
    dependent_vars=["stock revenue"],
    independent_vars=["extreme temperature"],
    control_vars=[
        "company size",
        "company age",
        "company distance to sea",
        "rain amount",
        "dry amount",
    ],
    instrument_var="company latitude",
    constant=True,
)

regression_description = "2sls regression for endogeneity test"

language_used = "english"


async def test_langchain_template():
    return await analyze_regression_result(
        regression_config,
        regression_description,
        regression_result,
        model_deepseek,
        language_used,
    )


if __name__ == "__main__":
    output = asyncio.run(test_langchain_template())
    print(output)
