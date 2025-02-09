import os
from pathlib import Path
import pandas as pd
import numpy as np

CURRENT_FILE = Path(__file__).resolve()
PANEL_DATA_FILE = os.path.join(
    os.path.dirname(os.path.dirname(CURRENT_FILE)),
    "test_data",
    "panel_data_5000_20.csv",
)

DEBUG = False


def add_fix_effect(df: pd.DataFrame, new_effect: str, indexed_effect: str):
    # Get unique entity IDs
    unique_entities = df.index.get_level_values(indexed_effect).unique()

    # Generate random integer assignments for each unique entity
    # Using integers 0-9 as an example
    cities = list(range(10))
    entity_city_map = {entity: np.random.choice(cities) for entity in unique_entities}

    # Create integer assignments for all rows based on entity_id
    df[new_effect] = df.index.get_level_values(indexed_effect).map(entity_city_map)

    return df


if __name__ == "__main__":
    df = pd.read_csv(PANEL_DATA_FILE)
    df = df.set_index(["entity_id", "time"])

    if DEBUG:
        print(df.head())
        input("Press any key to continue...")

    df = add_fix_effect(df, "city", "entity_id")

    output_file = os.path.join(
        os.path.dirname(os.path.dirname(CURRENT_FILE)),
        "test_data",
        "panel_data_5000_20_with_city.csv",
    )
    df.to_csv(output_file)
