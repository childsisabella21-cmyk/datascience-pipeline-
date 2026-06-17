import pandas as pd

def load_data(path):
    """
    Load dataset from the specified path.
    """
    try:
        df = pd.read_csv(path)
        if df.empty:
            print(f"Error: The file at '{path}' is empty.")
            return None
        return df
    except FileNotFoundError:
        print(f"Error: The file at '{path}' was not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: The file at '{path}' is empty.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading the data: {e}")
        return None
