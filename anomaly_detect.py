import argparse
import math
import os
import pickle
import random

from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve
import torch

from train_helper import get_config, load_models

parser = argparse.ArgumentParser()
parser.add_argument("--no_corrupt", default=False, action="store_true")
parser.add_argument("--corrupt_rate", default=0.3, type=float)
parser.add_argument("--graph_data_file_path", default="./data/test_dataset.p", type=str)
parser.add_argument("--max_graph_count", default=25000, type=int)
parser.add_argument(
    "--hyperparam_str",
    default="usggen_ordering-random_classweights-none_nodepred-True_edgepred-True_argmaxFalse_MHPFalse_batch-256_samples-256_epochs-300_nlr-0.001_nlrdec-0.95_nstep-1710_elr-0.001_elrdec-0.95_estep-1710",
    type=str,
)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EDGE_INFO = True  # Allow eGRU_from to receive information from eGRU_to


def load_graph_data(path: str) -> list:
    """
    Load graph data from a pickle file provided in the path. This assumes that the graph data is an array of tuples,
    where each tuple contains:
    - nodes: a list of node indices, each index corresponds to a node in the scenegraph
    - adjacent matrix: an array of shape (num_nodes, num_nodes), where each element is the index of the predicate between two nodes.

    Input:
    - path: str, path to the pickle file containing the graph data. The data

    Output:
    - graphs: list of tuples, where each tuple contains:
        - nodes: a list of node indices, each index corresponds to a node in the scenegraph
        - edges_from: a tensor of shape (num_nodes, num_nodes), where each element is the index of the predicate between two nodes.
        Each element in this is an upper triangular matrix that show node `i` has an edge to node `j`.
        - edges_to: a tensor array of shape (num_nodes, num_nodes), where each element is the index of the predicate between two nodes.
        Each element in this is an upper triangular matrix that show node `i` has an edge coming from node `j`.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    # Iterate through the data and extract nodes and adjacency matrixes.
    graphs = []
    for graph in data:
        nodes = torch.tensor(graph[0]).long().to(DEVICE)
        # HACK: with current config, maximum number of nodes is 25. So we only include graphs with less than 25 nodes.
        if len(nodes) > 25:
            continue

        adj_matrix = torch.tensor(graph[1]).long().to(DEVICE)

        # Divide adj_matrix into edges_from and edges_to list
        edges_from = torch.triu(adj_matrix, diagonal=1)
        edges_to = torch.triu(np.transpose(adj_matrix), diagonal=1)

        graphs.append((nodes, edges_from, edges_to))

    return graphs


# NOTE: don't need to count true edges because even non-edge NLL is sumed up in the end.
# def get_num_edges(edges_from, edges_to, indx_to_predicates) -> int:
#     """
#     Get the number of edges in the graph.

#     Input:
#     - edges_from: a tensor of shape (num_nodes, num_nodes), where each element is the index of the predicate between two nodes.
#     Each element in this is an upper triangular matrix that show node `i` has an edge to node `j`.
#     - edges_to: a tensor array of shape (num_nodes, num_nodes), where each element is the index of the predicate between two nodes.
#     Each element in this is an upper triangular matrix that show node `i` has an edge coming from node `j`.
#     - indx_to_predicates: a dictionary mapping index to predicate names.

#     Output:
#     - num_edges: int, the number of edges in the graph.
#     """
#     max_predicate = len(indx_to_predicates.keys())

#     count_edges_from = torch.count_nonzero((edges_from != 0) & (edges_from != max_predicate)).item()
#     count_edges_to = torch.count_nonzero((edges_to != 0) & (edges_to != max_predicate)).item()

#     return count_edges_from + count_edges_to


# BUG: the NLL is way too high, can reach infinity.
def cal_NLL(graph, params, config, models, edge_supervision=True) -> float:
    """
    Return the negative log-likelihood of a graph using trained models.

    Input:
    - graph: a list of tuples (nodes, adjacent matrix), where:
        - nodes: a list of node indices, each index corresponds to a node in the scenegraph
        - adjacent matrix: a numpy array of shape (num_nodes, num_nodes), where each element is the index of the predicate between two nodes.
    - params: parameters for the model.
    - config: configuration for the model.
    - models: the trained model.
    - edge_supervision: boolean, if True, use edge supervision.

    Output:
    - Negative log likelihood (NLL) score: float, the NLL score of the graph. The higher the score, the more anomalous the graph is.
    """
    # Extract true nodes, edges_from, edges_to fron input graph
    true_nodes, edges_from, edges_to = graph
    max_num_nodes = params.max_num_node - 1
    # Unpack config and model.
    num_graphs = 1
    ordering, weighted_loss, node_pred, edge_pred, use_argmax, use_MHP = get_config(
        config
    )
    (
        node_emb,
        edge_emb,
        mlp_node,
        gru_graph1,
        gru_graph2,
        gru_graph3,
        gru_edge1,
        gru_edge2,
    ) = models

    # set models to eval mode
    node_emb.eval()
    edge_emb.eval()
    mlp_node.eval()
    gru_graph3.eval()
    gru_graph1.eval()
    gru_graph2.eval()
    gru_edge1.eval()
    gru_edge2.eval()

    # intantiate placeholder for node and edge vectors. True graph data will be used to fill these vectors at each auto-regression time step.
    X = torch.zeros(num_graphs, max_num_nodes).to(DEVICE).long()
    Fto = torch.zeros(num_graphs, max_num_nodes, max_num_nodes + 1).to(DEVICE).long()
    Ffrom = torch.zeros(num_graphs, max_num_nodes, max_num_nodes + 1).to(DEVICE).long()

    # sample initial object from true nodes
    first_node = true_nodes[0]
    Xsample = torch.Tensor([first_node - 1]).long().to(DEVICE)
    # initial edge vector
    init_edges = torch.zeros(num_graphs, max_num_nodes).to(DEVICE).long()
    edge_SOS_token = torch.Tensor([params.edge_SOS_token]).to(DEVICE).long()
    # init gru_graph hidden state
    gru_graph1.hidden = gru_graph1.init_hidden(num_graphs)
    gru_graph2.hidden = gru_graph2.init_hidden(num_graphs)
    gru_graph3.hidden = gru_graph3.init_hidden(num_graphs)

    # Initialize negative log likelihood (NLL) score
    total_nll, node_nll, fto_nll, ffrom_nll = 0, 0, 0, 0

    softmax = torch.nn.Softmax(dim=0)

    # Go through each node in the graph and compute NLL.
    for i in range(len(true_nodes) - 1):

        # update graph info with generated nodes/edges
        X[:, i] = Xsample + 1
        Fto_vec = Fto[:, :, i]
        Ffrom_vec = Ffrom[:, :, i]

        # embed node and edge vectors for 3 history GRUs.
        Xsample1 = torch.unsqueeze(X[:, i], 1)
        Fto_vec = torch.unsqueeze(Fto_vec, 1)
        Ffrom_vec = torch.unsqueeze(Ffrom_vec, 1)
        Xsample1 = node_emb(Xsample1)
        Fto_vec = edge_emb(Fto_vec)
        Fto_vec = Fto_vec.contiguous().view(Fto_vec.shape[0], Fto_vec.shape[1], -1)
        Ffrom_vec = edge_emb(Ffrom_vec)
        Ffrom_vec = Ffrom_vec.contiguous().view(
            Ffrom_vec.shape[0], Ffrom_vec.shape[1], -1
        )
        gru_graph_in = torch.cat(
            (Xsample1.float(), Fto_vec.float(), Ffrom_vec.float()), 2
        )

        # run one step of gru_graph
        gru_edge_hidden1 = gru_graph1(gru_graph_in, list(np.ones(num_graphs))).data
        gru_edge_hidden2 = gru_graph2(gru_graph_in, list(np.ones(num_graphs))).data
        mlp_input = gru_graph3(Xsample1.float(), list(np.ones(num_graphs))).data

        # run mlp_node and calculate node_nll
        Xscores = mlp_node(mlp_input)
        Xscores = softmax(torch.squeeze(Xscores))
        true_node = true_nodes[i + 1] - 1
        true_node_prob = Xscores[true_node]
        node_nll += -torch.log(true_node_prob)
        Xsample = torch.tensor([true_node])

        # get initial hidden state of gru_edge
        if params.egru_num_layers > 1:
            gru_edge_hidden1 = torch.cat(
                (
                    gru_edge_hidden1,
                    torch.zeros(
                        params.egru_num_layers - 1,
                        gru_edge_hidden1.shape[1],
                        gru_edge_hidden1.shape[2],
                    ).to(DEVICE),
                ),
                0,
            )
            gru_edge_hidden2 = torch.cat(
                (
                    gru_edge_hidden2,
                    torch.zeros(
                        params.egru_num_layers - 1,
                        gru_edge_hidden2.shape[1],
                        gru_edge_hidden2.shape[2],
                    ).to(DEVICE),
                ),
                0,
            )
        gru_edge1.hidden = gru_edge_hidden1
        gru_edge2.hidden = gru_edge_hidden2

        # init edge vectors
        Fto_vec = init_edges.clone()
        Ffrom_vec = init_edges.clone()

        # Go through previous nodes to get edges score.
        for j in range(i + 1):

            # input for gru_in
            x1 = X[:, j]
            x2 = Xsample + 1
            fto = Fto_vec[:, j - 1] if j > 0 else edge_SOS_token
            ffrom = Ffrom_vec[:, j - 1] if j > 0 else edge_SOS_token

            # print('Embedding inputs to egru', x1, x2, fto, ffrom)
            x1 = node_emb(x1.view(x1.shape[0], 1))
            x2 = node_emb(x2.view(x2.shape[0], 1))
            fto = edge_emb(fto.view(fto.shape[0], 1))
            ffrom = edge_emb(ffrom.view(ffrom.shape[0], 1))

            # run gru_edge_to, calculate fto_nll and update Fto_vec
            if edge_supervision:
                gru_edge_in1 = torch.cat((x1, x2, fto, ffrom), 2)
            else:
                gru_edge_in1 = torch.cat((fto, ffrom), 2)
            Fto_scores = gru_edge2(gru_edge_in1)
            Fto_scores = softmax(torch.squeeze(Fto_scores))
            true_edge_to = edges_to[j, i + 1] - 1
            true_edge_to_prob = Fto_scores[true_edge_to]
            fto_nll += -torch.log(true_edge_to_prob)
            Fto_sample = torch.tensor([true_edge_to])
            Fto_vec[:, j] = torch.squeeze(Fto_sample) + 1

            # Prepare the input for Ffrom GRU
            fto_out = Fto_vec[:, j]
            fto_out = edge_emb(fto_out.view(fto_out.shape[0], 1))
            if edge_supervision:
                if EDGE_INFO:
                    gru_edge_in2 = torch.cat((x1, x2, fto, ffrom, fto_out), 2)
                else:
                    gru_edge_in2 = torch.cat((x1, x2, fto, ffrom), 2)
            else:
                gru_edge_in2 = torch.cat((fto, ffrom), 2)

            # run gru_edge_from and calculate ffrom_nll
            Ffrom_scores = gru_edge1(gru_edge_in2)
            Ffrom_scores = softmax(torch.squeeze(Ffrom_scores))
            true_edge_from = edges_from[j, i + 1] - 1
            true_edge_from_prob = Ffrom_scores[true_edge_from]
            ffrom_nll += -torch.log(true_edge_from_prob)
            Ffrom_sample = torch.tensor([true_edge_from])
            Ffrom_vec[:, j] = torch.squeeze(Ffrom_sample) + 1

            # update hidden state of gru_edge
            gru_edge1.hidden = gru_edge1.hidden.data.to(DEVICE)
            gru_edge2.hidden = gru_edge2.hidden.data.to(DEVICE)

        # update hidden state of gru_graph
        gru_graph1.hidden = gru_graph1.hidden.data.to(DEVICE)
        gru_graph2.hidden = gru_graph2.hidden.data.to(DEVICE)
        gru_graph3.hidden = gru_graph3.hidden.data.to(DEVICE)
        Fto[:, :, i + 1] = Fto_vec
        Ffrom[:, :, i + 1] = Ffrom_vec

    # Normalize NLL scores by the number of nodes in the graph.
    total_nll = node_nll + fto_nll + ffrom_nll
    # HACK: the NLL is way too high, can reach infinity.
    return 9999 if math.isinf(total_nll) else total_nll


def load_sample_graph() -> list:
    """
    Load a sample graph for testing purposes. This function is a placeholder and should be replaced with actual graph loading logic.

    Output:
    - A list of tuples (nodes, adjacent matrix), where:
        - nodes: a list of node indices, each index corresponds to a node in the scenegraph
        - adjacent matrix: a numpy array of shape (num_nodes, num_nodes), where each element is the index of the predicate between two nodes.
    """
    # This graph is drawn and stored in ./my_generates_samples/5.png.
    # Number of nodes: 9
    # NOTE: 51 represents no edge, because the predicate index is 0-50 in the pretrained model.
    X = np.array([73, 74, 8, 63, 104, 136, 104, 74, 61])
    Ffrom = np.array(
        [
            [0, 51, 51, 51, 51, 51, 51, 51, 51],
            [0, 0, 20, 51, 51, 51, 51, 51, 51],
            [0, 0, 0, 8, 51, 8, 23, 51, 51],
            [0, 0, 0, 0, 51, 51, 51, 51, 51],
            [0, 0, 0, 0, 0, 51, 51, 51, 51],
            [0, 0, 0, 0, 0, 0, 51, 51, 51],
            [0, 0, 0, 0, 0, 0, 0, 51, 51],
            [0, 0, 0, 0, 0, 0, 0, 0, 51],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    Fto = np.array(
        [
            [0, 51, 51, 51, 51, 51, 51, 51, 51],
            [0, 0, 51, 51, 51, 51, 51, 51, 51],
            [0, 0, 0, 51, 51, 51, 31, 20, 51],
            [0, 0, 0, 0, 51, 51, 51, 51, 51],
            [0, 0, 0, 0, 0, 51, 51, 51, 51],
            [0, 0, 0, 0, 0, 0, 51, 51, 51],
            [0, 0, 0, 0, 0, 0, 0, 51, 51],
            [0, 0, 0, 0, 0, 0, 0, 0, 51],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    return [(X, Ffrom, Fto)]


def normalize_nll(nll, graph) -> float:
    """
    Normalize NLL score by the number of nodes and edges in the graph.

    Input:
    - nll: float, the NLL score of the graph.
    - graph: a tuple (nodes, adjacent matrix).

    Output:
    - normalized_nll: float, the normalized NLL score of the graph.
    """

    num_nodes = graph[0].shape[0]
    num_edges = num_nodes * (num_nodes - 1) / 2  # Complete graph with no self-loops

    # Normalize NLL score by the number of nodes and edges in the graph.
    normalized_nll = nll / (num_nodes + num_edges)

    return normalized_nll


def plot_nll_score(nll) -> None:
    """
    Plot NLL scores as a histogram.

    Input:
    - nll: list of NLL scores for each graph.
    """

    fig, ax = plt.subplots()
    ax.set_title("NLL score histogram")
    ax.set_xlabel("NLL score")
    ax.set_ylabel("Frequency")

    ax.hist(nll, bins=60, color="blue", alpha=0.7, edgecolor="black")

    fig.show()


def plot_ROC_curve(nlls, labels) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    fpr, tpr, thresholds = roc_curve(labels, nlls)

    auroc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC curve (area = {auroc:.2f})")
    ax.plot(
        [0, 1], [0, 1], color="red", linestyle="--"
    )  # This is random guess line (random classifier)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve for Anomaly Detection using NLL")
    ax.legend()

    fig.show()

    return fpr, tpr, thresholds, auroc


def corrupt_edges(edges, num_nodes: int, ind_to_predicates: list) -> torch.Tensor:
    """
    Corrupt the edges in the graph by randomly changing their indices.

    Input:
    - edges: a tensor of shape (num_nodes, num_nodes), where each element is the index of the predicate between two nodes.
    - num_nodes: int, the number of nodes in the graph.
    - ind_to_predicates: list, mapping from index to predicate names.

    Ouput:
    - corrupted_edges: a tensor of shape (num_nodes, num_nodes), where each element is the index of the predicate between two nodes.
    """
    corrupted_edges = edges.clone()
    max_edge_value = len(ind_to_predicates) - 1

    # Since edges_from and edges_to are upper triangular matrices, we only need to iterate through the upper triangular part of the matrix.
    # This is because the graph is undirected, so edges_from[i][j] and edges_to[j][i] are the same.
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < 0.5:
                corrupted_edges[i, j] = (
                    torch.randint(0, max_edge_value, (1,)).long().to(DEVICE)
                )

    return corrupted_edges


def corrupt_nodes(nodes, num_nodes: int, ind_to_classes: list) -> torch.Tensor:
    """
    Corrupt the nodes in the graph by randomly changing their indices.

    Input:
    - nodes: a tensor of shape (num_nodes), where each element is the index of a node in the scenegraph.
    - num_nodes: int, the number of nodes in the graph.

    Output:
    - corrupted_nodes: a tensor of shape (num_nodes), where each element is the index of a node in the scenegraph.
    """
    corrupted_nodes = nodes.clone()
    max_node_value = len(ind_to_classes) - 1
    for i in range(num_nodes):
        if random.random() < 0.5:  # Randomly change the node index with 50% probability
            corrupted_nodes[i] = (
                torch.randint(0, max_node_value, (1,)).long().to(DEVICE)
            )

    return corrupted_nodes


def corrupt_graph_data(
    graphs: list, corrupt_rate: float, ind_to_classes: list, ind_to_predicates: list
) -> tuple[list, np.ndarray]:
    """
    Corrupt the graph data by randomly changing the nodes and edges in the graph.

    Input:
    - graphs: list of tuples (nodes, edges_from, edges_to).
    - corrupt_rate: float, the rate at which to corrupt the graph data.
    - ind_to_classes: list, mapping from index to class names.
    - ind_to_predicates: list, mapping from index to predicate names.

    Output:
    - corrupted_graphs: list of tuples (nodes, edges_from, edges_to).
    - labels: np.ndarray, array of labels indicating whether the graph is corrupted (1) or not (0).
    """
    corrupted_graphs = []
    labels = []

    for graph in graphs:
        nodes, edges_from, edges_to = graph
        num_nodes = len(nodes)

        # Randomly corrupt the nodes and edges in the graph based on the corrupt_rate
        if random.random() < corrupt_rate:
            # Corrupt the nodes
            nodes = corrupt_nodes(nodes, num_nodes, ind_to_classes)

            # Corrupt the edges
            edges_from = corrupt_edges(edges_from, num_nodes, ind_to_predicates)
            edges_to = corrupt_edges(edges_to, num_nodes, ind_to_predicates)

            labels.append(1)  # Corrupted graph
        else:
            labels.append(0)  # Original graph

        corrupted_graphs.append((nodes, edges_from, edges_to))

    return corrupted_graphs, np.array(labels)


if __name__ == "__main__":
    # Get random seed for reproducibility
    print("Loading data..")
    RANDOM_SEED = 121
    os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    print("random seed", RANDOM_SEED)

    # Load the model, params, and config
    print("Loading SceneGraphGen model")
    params, config, models = load_models("./models", args.hyperparam_str)
    ind_to_classes, ind_to_predicates, _ = pickle.load(
        open(os.path.join("./data", "categories.p"), "rb")
    )
    result_file_name = "./nll.txt"

    # Load graph data
    graphs = load_graph_data(args.graph_data_file_path)
    # graphs = load_sample_graph()

    # Corrupt graph data if needed
    print("Corrupting graph data...")
    if args.no_corrupt:
        labels = np.array([0] * len(graphs))
    else:
        graphs, labels = corrupt_graph_data(
            graphs, args.corrupt_rate, ind_to_classes, ind_to_predicates
        )

    # Calculate negative log likelihood for each graph
    print("Calculating NLL for each graph...")
    max_call_count = args.max_graph_count
    count = 1
    with open(result_file_name, "w") as f:
        for graph in graphs:
            if count == max_call_count:
                break

            nll = cal_NLL(graph, params, config, models)
            # nll = normalize_nll(nll, graph)
            print(f"NLL {count}: {nll}")
            count += 1

            f.writelines(f"{nll}\n")

    # HACK: test specific graph
    # graph = graphs[168]
    # nll = cal_NLL(graph, params, config, models)
    # print(f"NLL: {nll}")

    print("Done calculating NLL.")

    # Detach tensors and convert to NumPy arrays if necessary
    with open(result_file_name, "r") as f:
        nll_arr = [float(line.strip()) for line in f.readlines()]

    # Plot NLL scores and ROC curve
    plot_nll_score(nll_arr)
    fpr, tpr, thresholds, auroc = plot_ROC_curve(nll_arr, labels[: max_call_count - 1])

    # Find optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold: {optimal_threshold}")
    print(f"True Positive Rate: {tpr[optimal_idx]}")
    print(f"False Positive Rate: {fpr[optimal_idx]}")
    print(f"AUROC: {auroc}")

    plt.show()
