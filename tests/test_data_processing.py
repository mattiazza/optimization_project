# Testing the function
import pandas as pd
from opt_project.data_processing import is_more_than_one

def test_is_more_than_one():
    # Create a sample dataframe
    sample_data = {
        'review_profilename': ['user1', 'user2', 'user1', 'user3', 'user2', 'user1'],
        'review_overall': [4.0, 3.5, 4.5, 2.0, 3.0, 4.0]
    }
    df_sample = pd.DataFrame(sample_data)
    # Expected output   
    expected_output = pd.Series([True, True, True, False, True, True])
    # Test the function
    assert is_more_than_one(df_sample, 'review_profilename').equals(expected_output)
    def test_is_more_than_one_empty_dataframe():
        # Test with empty dataframe
        df_empty = pd.DataFrame({'review_profilename': []})
        # Empty DataFrame should return empty Series
        assert len(is_more_than_one(df_empty, 'review_profilename')) == 0

    def test_is_more_than_one_single_users():
        # Test when all users have only one review
        sample_data = {
            'review_profilename': ['user1', 'user2', 'user3', 'user4'],
            'review_overall': [4.0, 3.5, 4.5, 2.0]
        }
        df_sample = pd.DataFrame(sample_data)
        # Expected output - all False because no user has more than one review
        expected_output = pd.Series([False, False, False, False])
        assert is_more_than_one(df_sample, 'review_profilename').equals(expected_output)

    def test_is_more_than_one_different_column():
        # Test with a different column as the ID
        sample_data = {
            'user_id': ['A', 'B', 'A', 'C', 'B'],
            'product': ['p1', 'p2', 'p3', 'p4', 'p5']
        }
        df_sample = pd.DataFrame(sample_data)
        expected_output = pd.Series([True, True, True, False, True])
        assert is_more_than_one(df_sample, 'user_id').equals(expected_output)