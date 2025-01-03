import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pathlib import Path
import os

import unittest
import time
from io import StringIO
import sys

CURRENT_FILE = Path(__file__).resolve()
PANEL_DATA_FILE = os.path.join(
    os.path.dirname(os.path.dirname(CURRENT_FILE)),
    "test_data",
    "panel_data_5000_20_with_city.csv"
)

def read_df(df: pd.DataFrame):
    output = []
    output.append(f"Shape: {df.shape}")
    output.append(f"Columns: {df.columns.tolist()}")
    output.append(f"Data Types:\n{df.dtypes}")
    output.append(f"First 5 rows:\n{df.head()}")
    output.append(f"Basic Statistics:\n{df.describe()}")
    output.append(f"Missing Values:\n{df.isnull().sum()}")


def parallel_read_df(df: pd.DataFrame, num_threads=5, run_times=10):
    # Split dataframe into chunks
    chunks = np.array_split(df, run_times)
    
    # Create thread pool and execute read_df on chunks in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = []
        results.extend(list(executor.map(read_df, chunks)))
    return results

class TestParallelReadDF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(PANEL_DATA_FILE)
        cls.df = cls.df.set_index(["entity_id", "time"])
        
    def test_parallel_execution_faster(self):
        run_times = 100
        # Test single thread execution time
        start_time = time.time()
        for _ in range(run_times):
            read_df(self.df)
        single_thread_time = time.time() - start_time
        
        # Test parallel execution time
        start_time = time.time()
        parallel_read_df(self.df, num_threads=5, run_times=run_times)
        parallel_time = time.time() - start_time
        
        # Verify parallel execution is faster
        print(f"Single thread time: {single_thread_time:.2f} seconds")
        print(f"Parallel time: {parallel_time:.2f} seconds") 
        self.assertLess(parallel_time, single_thread_time)
        

if __name__ == "__main__":
    unittest.main()
