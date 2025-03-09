import os
import lzma
import pickle
import numpy as np
from nltk.tree import Tree


class DistanceDict(dict):
    """
    Small helper class implementing a symmetrical dictionary to hold distance data.
    """

    def __init__(self, distances):
        self.distances = {tuple(sorted(t)): v for t, v in distances.items()}

    def __getitem__(self, i):
        if i[0] == i[1]:
            return 0
        else:
            return self.distances[(i[0], i[1]) if i[0] < i[1] else (i[1], i[0])]

    def __setitem__(self, i):
        raise NotImplementedError()


def get_label(node):
    if isinstance(node, Tree):
        return node.label()
    else:
        return node


def load_hierarchy(dataset, data_dir):
    """
    Load the hierarchy corresponding to a given dataset.

    Args:
        dataset: The dataset name for which the hierarchy should be loaded.
        data_dir: The directory where the hierarchy files are stored.

    Returns:
        A nltk tree whose labels corresponds to wordnet wnids.
    """
    if dataset == "tiered-imagenet-224":
        fname = os.path.join(data_dir, "tiered_imagenet_tree.pkl")
    elif dataset == "inaturalist19-224":
        fname = os.path.join(data_dir, "inaturalist19_tree.pkl")
    elif dataset == "cifar-100":
        fname = os.path.join(data_dir, "cifar_100_tree.pkl")
    elif dataset == "fgvc-aircraft":
        fname = os.path.join(data_dir, "fgvc_aircraft_tree.pkl")
    elif dataset == "wildlife-species":
        fname = os.path.join(data_dir, "wildlife_species_tree.pkl")
    else:
        raise ValueError("Unknown dataset {}".format(dataset))

    with open(fname, "rb") as f:
        return pickle.load(f)


def load_distances(dataset, dist_type, data_dir):
    """
    Load the distances corresponding to a given hierarchy.

    Args:
        dataset: The dataset name for which the hierarchy should be loaded.
        dist_type: The distance type, one of ['jc', 'ilsvrc'].
        data_dir: The directory where the hierarchy files are stored.

    Returns:
        A dictionary of the form {(wnid1, wnid2): distance} for all precomputed
        distances for the hierarchy.
    """
    assert dist_type in ["ilsvrc", "original"]

    if dataset == "tiered-imagenet-224":
        dataset = "tiered-imagenet"
    elif dataset == "inaturalist19-224":
        dataset = "inaturalist19"
    elif dataset == "cifar-100":
        dataset = "cifar-100"
    elif dataset == "fgvc-aircraft":
        dataset = "fgvc-aircraft"
    elif dataset == "wildlife-species":
        dataset = "wildlife_species"
    else:
        raise ValueError("Unknown dataset {}".format(dataset))

    with lzma.open(os.path.join(data_dir, "{}_{}_distances.pkl.xz".format(dataset, dist_type).replace("-", "_")),
                   "rb") as f:
        return DistanceDict(pickle.load(f))


def get_classes(hierarchy: Tree, output_all_nodes=False):
    """
    Return all classes associated with a hierarchy. The classes are sorted in
    alphabetical order using their label, putting all leaf nodes first and the
    non-leaf nodes afterwards.

    Args:
        hierarhcy: The hierarchy to use.
        all_nodes: Set to true if the non-leaf nodes (excepted the origin) must
            also be included.

    Return:
        A pair (classes, positions) of the array of all classes (sorted) and the
        associated tree positions.
    """

    def get_classes_from_positions(positions):
        classes = [get_label(hierarchy[p]) for p in positions]
        class_order = np.argsort(classes)  # we output classes in alphabetical order
        positions = [positions[i] for i in class_order]
        classes = [classes[i] for i in class_order]
        return classes, positions

    positions = hierarchy.treepositions("leaves")
    classes, positions = get_classes_from_positions(positions)

    if output_all_nodes:
        positions_nl = [p for p in hierarchy.treepositions() if p not in positions]
        classes_nl, positions_nl = get_classes_from_positions(positions_nl)
        classes += classes_nl
        positions += positions_nl

    return classes, positions


def hierarchy_to_lca(hierarchy: np.ndarray) -> np.ndarray:
    """
    Converts a hierarchy to a Least Common Ancestor (LCA) matrix.

    The LCA matrix is a square matrix where each element (i, j) represents
    the level of the least common ancestor for classes i and j.

    Args:
        hierarchy (np.array): A matrix where each row represents the ancestor
            hierarchy of a class.

    Returns:
        A square numpy array containing the LCA matrix.
    """
    # Number of hierarchy levels (L)
    # Number of finer classes (C)
    L, C = hierarchy.shape

    lca = np.full((C, C), L, dtype=int)

    for level in hierarchy:
        for row, coarse in zip(lca, level):
            for index, value in enumerate(level):
                if coarse == value:
                    row[index] -= 1
    return lca


def lca_to_hierarchy(lca: np.ndarray) -> np.ndarray:
    """
    Converts a Least Common Ancestor (LCA) matrix to a hierarchy matrix.

    The hierarchy matrix is a matrix where each row represents the ancestor hierarchy
    of a class.

    Args:
        lca (np.array): A square matrix where each element (i, j) represents
            the level of the least common ancestor for classes i and j.

    Returns:
        A numpy array containing the hierarchy matrix.
    """
    # Make a copy to avoid inplace operations
    lca = np.array(lca, dtype=int)

    # Number of hierarchy levels (L)
    # Number of finer classes (C)
    L, C = lca.max(), len(lca)

    hierarchy = -np.ones((L, C), dtype=int)

    for level in range(L):
        # Find all siblings at `level`,
        # reverse to be consistence at level 0
        siblings = np.unique(lca == level, axis=0)[::-1]

        # Generate labeler
        labeler = np.arange(len(siblings), dtype=int)

        # Apply labels to siblings with labeler
        labels = labeler @ siblings

        # Add labels to hierarchy
        hierarchy[level] = labels

        # Update lca for next iteration
        lca[lca == level] += 1

    return hierarchy


def accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
    hierarchy: np.ndarray,
    level: int = 0,
    k: int = 1,
) -> float:
    top_k_preds = np.argsort(predictions, axis=1)[:, -k:]
    labels = labels.reshape(-1, 1)
    corrects = np.any(
        hierarchy[level][top_k_preds] == hierarchy[level][labels],
        axis=1,
    )
    return np.mean(corrects)


def error_rate(
    predictions: np.ndarray,
    labels: np.ndarray,
    hierarchy: np.ndarray,
    level: int = 0,
    k: int = 1,
) -> float:
    top_k_preds = np.argsort(predictions, axis=1)[:, -k:]
    labels = labels.reshape(-1, 1)
    wrongs = np.all(
        hierarchy[level][top_k_preds] != hierarchy[level][labels],
        axis=1,
    )
    return np.mean(wrongs)


def hier_dist_mistake(
    predictions: np.ndarray,
    labels: np.ndarray,
    hierarchy: np.ndarray,
    level: int = 0,
    k: int = 1,
) -> float:
    top_k_preds = np.argsort(predictions, axis=1)[:, -k:]
    labels = labels.reshape(-1, 1)
    wrongs = np.all(
        hierarchy[level][top_k_preds] != hierarchy[level][labels],
        axis=1,
    )
    lca = hierarchy_to_lca(hierarchy[level:])
    lca_heights = lca[top_k_preds[wrongs], labels[wrongs]]
    return np.mean(lca_heights)


def hier_dist(
    predictions: np.ndarray,
    labels: np.ndarray,
    hierarchy: np.ndarray,
    level: int = 0,
    k: int = 1,
) -> float:
    top_k_preds = np.argsort(predictions, axis=1)[:, -k:]
    labels = labels.reshape(-1, 1)
    lca_heights = hierarchy_to_lca(hierarchy[level:])[top_k_preds, labels]
    return np.mean(lca_heights)