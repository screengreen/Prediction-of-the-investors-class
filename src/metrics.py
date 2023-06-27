import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def get_metrics(y_pred, y_test):
    """
    Evaluate model performance metrics.

    Args:
        y_pred (array_like): A numpy array or a list of predicted target values.
        y_test (array_like): A numpy array or a list of true target values.

    Returns:
        dict: A dictionary containing accuracy_score, precision_score, recall_score, and f1_score metrics.

    Raises:
        ValueError: If `y_pred` and `y_test` have different lengths or incompatible data types.
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    return {
        'accuracy_score': accuracy,
        'precision_score': precision,
        'recall_score': recall,
        'f1_score': f1
    }

def make_log_data(model, model_params, metrics, random_state):
    """
    Return a dictionary of model information, its parameters, performance metrics, and the random state.

    Args:
        model (object): An object that implements the Scikit-learn API of a supervised learning algorithm.
        model_params (dict): A dictionary of hyperparameters used by the model.
        metrics (dict): A dictionary of performance metrics returned from fitting the model.
        random_state (int): A seed integer that will be used for initializing the random number generator.

    Returns:
        dict: Model information, its parameters, performance metrics, and the random state.

    Raises:
        TypeError: If any of the input arguments have incorrect types.
    """
    log_data = {'model': model, "model_params": model_params, "metrics": metrics, 'random state': random_state}
    return log_data


def append_to_json(data, file_name):
    """
    Append data to a JSON file.

    Args:
        data (dict): The data to be appended to the file.
        file_name (str): The name of the file where data should be appended.

    Returns:
        None

    Raises:
        TypeError: If data is not of type dict.
        OSError: If unable to open or add data to the provided file.
    """
    if not isinstance(data, dict):
        raise TypeError("Data must be a dictionary.")

    try:
        with open(file_name, "a") as f:
            json.dump(data, f, indent=4)
    except OSError as e:
        raise OSError(f"Unable to open or write to {file_name}. Error: {e}") from e
