import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(path):
    """
    Load dataset containing text and sentiment labels.
    """
    df = pd.read_csv(path)
    return df


def split_dataset(texts, labels, test_size=0.2):
    """
    Split dataset into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=42
    )

    return X_train, X_test, y_train, y_test
