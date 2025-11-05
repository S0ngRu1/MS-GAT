from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

def collect_metrics(dataset, y_true, y_pred):
    y_pred_labels = y_pred.argmax(1)

    acc = accuracy_score(y_true, y_pred_labels)
    precision = precision_score(y_true, y_pred_labels, average='weighted')
    recall = recall_score(y_true, y_pred_labels, average='weighted')
    f1 = f1_score(y_true, y_pred_labels, average='weighted')

    eval_results = {
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }

    return eval_results
