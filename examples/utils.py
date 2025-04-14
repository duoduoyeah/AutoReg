import pandas as pd

def convert_xlsx_to_csv(xlsx_path: str, csv_path: str) -> str:
    """
    Converts an XLSX file to a CSV file.

    Args:
        xlsx_path: The path to the XLSX file.
        csv_path: The path to the output CSV file.

    Returns:
        The path to the generated CSV file.
    """
    try:
        df = pd.read_excel(xlsx_path)
        df.to_csv(csv_path, index=False)
        return csv_path
    except FileNotFoundError:
        print(f"Error: File not found at {xlsx_path}")
        return ""
    except Exception as e:
        print(f"Error converting XLSX to CSV: {e}")
        return ""

def get_csv_headers(csv_path: str) -> list:
    """
    Reads a CSV file and returns its headers as a list.

    Args:
        csv_path: The path to the CSV file.

    Returns:
        A list of strings representing the column headers.
        Returns an empty list if the file is not found or an error occurs.
    """
    try:
        df = pd.read_csv(csv_path)
        return df.columns.tolist()
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        return []
    except Exception as e:
        print(f"Error reading CSV headers: {e}")
        return []

def add_future_ncskew(csv_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Reads a CSV file with financial data, indexes by company_id and year,
    and adds NCSKEW_1 and NCSKEW_2 variables representing NCSKEW values
    for year+1 and year+2.
    
    Args:
        csv_path: Path to the input CSV file
        output_path: Optional path to save the modified data. If None, the data is not saved.
        
    Returns:
        The modified pandas DataFrame with the new variables added
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Ensure year is integer type for proper sorting
        df['year'] = df['year'].astype(int)
        
        # Create a copy of the dataframe indexed by company_id and year
        # This preserves the original order while allowing for the shift operation
        indexed_df = df.set_index(['company_id', 'year'])
        
        # Group by company_id and shift NCSKEW to get future values
        grouped = df.groupby('company_id')
        
        # Create the new variables by shifting NCSKEW within each company group
        df['NCSKEW_1'] = grouped['NCSKEW'].shift(-1)  # next year's NCSKEW
        df['NCSKEW_2'] = grouped['NCSKEW'].shift(-2)  # year after next's NCSKEW
        
        # Save to output file if specified
        if output_path:
            df.to_csv(output_path, index=False)
            
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error processing CSV file: {e}")
        return pd.DataFrame()

def predict_missing_ncskew(csv_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Predicts missing NCSKEW_1 and NCSKEW_2 values based on CPRI changes.
    
    For each company_id and year with missing NCSKEW_1 or NCSKEW_2 values:
    - If CPRI is higher than previous year: last year's NCSKEW + 0.1
    - Otherwise: last year's NCSKEW - 0.1
    
    Args:
        csv_path: Path to the CSV file with panel data
        output_path: Optional path to save the updated data
        
    Returns:
        Updated pandas DataFrame with predicted values
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Ensure proper data types
        df['year'] = df['year'].astype(int)
        if 'company_id' not in df.columns:
            raise ValueError("CSV must contain 'company_id' column")
        
        # Sort by company_id and year for proper sequential processing
        df = df.sort_values(['company_id', 'year'])
        
        # Process each company separately
        for company in df['company_id'].unique():
            company_data = df[df['company_id'] == company].copy()
            
            # Skip if only one year of data
            if len(company_data) <= 1:
                continue
                
            # Process each year for this company
            for i in range(1, len(company_data)):
                prev_year_idx = company_data.index[i-1]
                curr_year_idx = company_data.index[i]
                
                # Check if CPRI is available for comparison
                if 'CPRI' in df.columns and not pd.isna(company_data.loc[curr_year_idx, 'CPRI']) and not pd.isna(company_data.loc[prev_year_idx, 'CPRI']):
                    cpri_increased = company_data.loc[curr_year_idx, 'CPRI'] > company_data.loc[prev_year_idx, 'CPRI']
                    adjustment = 0.1 if cpri_increased else -0.1
                    
                    # Update NCSKEW_1 if missing
                    if 'NCSKEW_1' in df.columns and pd.isna(company_data.loc[prev_year_idx, 'NCSKEW_1']) and not pd.isna(company_data.loc[prev_year_idx, 'NCSKEW']):
                        df.loc[prev_year_idx, 'NCSKEW_1'] = company_data.loc[prev_year_idx, 'NCSKEW'] + adjustment
                    
                    # Update NCSKEW_2 if missing
                    if 'NCSKEW_2' in df.columns and pd.isna(company_data.loc[prev_year_idx, 'NCSKEW_2']):
                        # If NCSKEW_1 is available, base prediction on it
                        if not pd.isna(company_data.loc[prev_year_idx, 'NCSKEW_1']):
                            df.loc[prev_year_idx, 'NCSKEW_2'] = company_data.loc[prev_year_idx, 'NCSKEW_1'] + adjustment
                        # Otherwise base it on NCSKEW
                        elif not pd.isna(company_data.loc[prev_year_idx, 'NCSKEW']):
                            df.loc[prev_year_idx, 'NCSKEW_2'] = company_data.loc[prev_year_idx, 'NCSKEW'] + (2 * adjustment)
        
        # Fill any remaining missing values in the last years
        # These are likely 2022-2023 data points that don't have future values
        for company in df['company_id'].unique():
            company_data = df[df['company_id'] == company].sort_values('year')
            
            # Handle last and second-to-last year
            if len(company_data) >= 2:
                last_idx = company_data.index[-1]
                second_last_idx = company_data.index[-2]
                
                # For the second-to-last year, if NCSKEW_2 is missing, base it on NCSKEW_1
                if 'NCSKEW_2' in df.columns and pd.isna(df.loc[second_last_idx, 'NCSKEW_2']) and not pd.isna(df.loc[second_last_idx, 'NCSKEW_1']):
                    # Use the same CPRI-based adjustment
                    if 'CPRI' in df.columns and not pd.isna(df.loc[second_last_idx, 'CPRI']) and not pd.isna(df.loc[second_last_idx-1, 'CPRI']):
                        cpri_increased = df.loc[second_last_idx, 'CPRI'] > df.loc[second_last_idx-1, 'CPRI']
                        adjustment = 0.1 if cpri_increased else -0.1
                        df.loc[second_last_idx, 'NCSKEW_2'] = df.loc[second_last_idx, 'NCSKEW_1'] + adjustment
                
                # For the last year, predict both NCSKEW_1 and NCSKEW_2 if missing
                if 'NCSKEW_1' in df.columns and pd.isna(df.loc[last_idx, 'NCSKEW_1']) and not pd.isna(df.loc[last_idx, 'NCSKEW']):
                    if 'CPRI' in df.columns and not pd.isna(df.loc[last_idx, 'CPRI']) and not pd.isna(df.loc[last_idx-1, 'CPRI']):
                        cpri_increased = df.loc[last_idx, 'CPRI'] > df.loc[last_idx-1, 'CPRI']
                        adjustment = 0.1 if cpri_increased else -0.1
                        df.loc[last_idx, 'NCSKEW_1'] = df.loc[last_idx, 'NCSKEW'] + adjustment
                
                if 'NCSKEW_2' in df.columns and pd.isna(df.loc[last_idx, 'NCSKEW_2']):
                    if not pd.isna(df.loc[last_idx, 'NCSKEW_1']):
                        if 'CPRI' in df.columns and not pd.isna(df.loc[last_idx, 'CPRI']) and not pd.isna(df.loc[last_idx-1, 'CPRI']):
                            cpri_increased = df.loc[last_idx, 'CPRI'] > df.loc[last_idx-1, 'CPRI']
                            adjustment = 0.1 if cpri_increased else -0.1
                            df.loc[last_idx, 'NCSKEW_2'] = df.loc[last_idx, 'NCSKEW_1'] + adjustment
                    elif not pd.isna(df.loc[last_idx, 'NCSKEW']):
                        if 'CPRI' in df.columns and not pd.isna(df.loc[last_idx, 'CPRI']) and not pd.isna(df.loc[last_idx-1, 'CPRI']):
                            cpri_increased = df.loc[last_idx, 'CPRI'] > df.loc[last_idx-1, 'CPRI']
                            adjustment = 0.1 if cpri_increased else -0.1
                            df.loc[last_idx, 'NCSKEW_2'] = df.loc[last_idx, 'NCSKEW'] + (2 * adjustment)
        
        # Save to output file if specified
        if output_path:
            df.to_csv(output_path, index=False)
            
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error predicting missing NCSKEW values: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    csv_path = "/workspaces/AutoReg/examples/ping/data_with_future_ncskew_predicted.csv"
    print(get_csv_headers(csv_path))

