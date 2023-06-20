from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json

def get_metrics(y_pred, y_test):
    # Evaluate model performance
    print("Accuracy: ",  accuracy_score( y_test, y_pred))
    print("Precision: ", precision_score(y_test, y_pred, average='weighted'))
    print("Recall: ",    recall_score(   y_test, y_pred, average='weighted'))
    print("F1-Score: ",  f1_score(       y_test, y_pred, average='weighted'))

    return {'accuracy_score': accuracy_score( y_test, y_pred), 'precision_score': precision_score(y_test, y_pred, average='weighted'), 'recall_score': recall_score(y_test, y_pred, average='weighted'), 'f1_score': f1_score(y_test, y_pred, average='weighted')}


def make_log_data(model, model_params, metrics, random_state):
    return {'model': model, "model_params": model_params, "metrics": metrics, 'random state': random_state}


def append_to_json(data, file_name):
    with open(file_name, "a") as f:
        json.dump(data, f, indent=4)



