def construct_Zdataset(path, include_herb_target=True):

    print('Zgraph construct finished')
    return graph
def load_Zdataset(path):
    # load heterogeneous data

    print('Zdataset loading finished')
    return data_set, graph, herb_num, target_num

def evaluate(model, g, test_data, args):
    model.eval()
    with torch.no_grad():
        drug_embedding, disease_embedding, score_mat = model(g, args)
        score_mat = score_mat.cpu().numpy()

    test_data = test_data.cpu().numpy()
    test_score = score_mat[tuple(np.array(test_data[:, 0:2]).T)].tolist()
    test_true = list(test_data[:, 2])

    # Calculate AUROC and AUPR
    auroc = metrics.roc_auc_score(test_true, test_score)
    aupr = metrics.average_precision_score(test_true, test_score)

    # Convert scores to binary predictions using a threshold (default: 0.5)
    threshold = 0.5
    test_pred = [1 if score >= threshold else 0 for score in test_score]

    # Calculate accuracy, precision, recall, and F1-score
    accuracy = metrics.accuracy_score(test_true, test_pred)
    precision = metrics.precision_score(test_true, test_pred)
    recall = metrics.recall_score(test_true, test_pred)
    f1 = metrics.f1_score(test_true, test_pred)

    return auroc, aupr, accuracy, precision, recall, f1, test_score, test_true


def get_ppr_matrix(
        adj_matrix: np.ndarray,
        alpha: float = 0.1) -> np.ndarray:
    num_nodes = adj_matrix.shape[0]
    A_tilde = adj_matrix + np.eye(num_nodes)
    D_tilde = np.diag(1 / np.sqrt(A_tilde.sum(axis=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return alpha * np.linalg.inv(np.eye(num_nodes) - (1 - alpha) * H)


def get_heat_matrix(
        adj_matrix: np.ndarray,
        t: float = 5.0) -> np.ndarray:
    num_nodes = adj_matrix.shape[0]
    A_tilde = adj_matrix + np.eye(num_nodes)
    D_tilde = np.diag(1 / np.sqrt(A_tilde.sum(axis=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return expm(-t * (np.eye(num_nodes) - H))
