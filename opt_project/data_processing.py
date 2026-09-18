import pandas as pd
import numpy as np

def is_more_than_one(x: pd.DataFrame, id: str) -> pd.Series:
    """
    Function to check if the user has more than one review.
    """
    return x[id].map(x[id].value_counts()) > 1
