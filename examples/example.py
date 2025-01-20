import dotenv
import os
from langchain_openai import ChatOpenAI
import pandas as pd
import json

from auto_reg.regression.regression_config import ResearchConfig
from auto_reg.regression.panel_data import *
from auto_reg.analysis.generate_table import *


#==============================================
# setup langchain model
#==============================================
dotenv.load_dotenv()
model_4o = ChatOpenAI(
    model_name="gpt-4o",
    timeout=(45.0), # 45 seconds before timeout
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

model_deepseek = ChatOpenAI(
    model_name="deepseek-chat",
    temperature=0,
    timeout=(45.0),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_API_BASE")
)

# Map different tasks to specific models
model: dict[str, ChatOpenAI] = {
    'design_model': model_4o,  # For designing table layouts
    'draw_model': model_deepseek,  # For drawing regression tables
    'analysis_model': model_4o,  # For analyzing regression results
}

#==============================================
# setup data
#==============================================
df = pd.read_csv('temp/simulated_data.csv')
df = df.set_index(['company_id', 'year'])
# Remove rows with missing values
df = df.dropna()

def load_research_config(config_path: str) -> ResearchConfig:
    with open(config_path) as f:
        config_data = json.load(f)
    return ResearchConfig(**config_data)

research_config = load_research_config('libs/auto_reg/examples/research_config.json')
research_config.validate_research_config(df)
research_topic: str = research_config.research_topic

#==============================================
# main program
#==============================================
async def main():
    regression_results = run_regressions(
        df, 
        research_config.generate_regression_configs()
    )

    # Design regression tables
    table_design: TableDesign|None = await design_regression_tables(
        research_topic,
        regression_results,
        model['design_model']
    )

    if table_design is None:
        return

    # user select tables
    table_design = select_table_design(table_design)
    print(table_design)
    
    # draw tables
    table_results = ResultTables()
    await draw_tables(
        regression_results,
        table_design,
        model['draw_model'],
        table_results
    )

    await analyze_regression_results(
        regression_results,
        table_design,
        table_results,
        model['analysis_model']
    )

    # combine tables
    combined_table_results: ResultTables = await combine_tables(
        table_results, 
        table_design, 
        model['draw_model'])

    return combined_table_results

if __name__ == "__main__":
    asyncio.run(main())