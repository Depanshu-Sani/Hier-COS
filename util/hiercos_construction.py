import os
import numpy as np
import torch
import torch.nn.functional as F
from MBM.better_mistakes.trees import load_hierarchy

# HAFrame
from HAFrame.distance import distance_dict_to_mat
from HAFrame.solve_HAF import map_hdistance_to_cosine_similarity_exponential_decay

# HAFS
from nltk.tree import Tree
from collections import deque


def count_nodes_at_each_level(tree):
    level_counts = []

    def traverse(node, level):
        # Ensure the level_counts list is long enough
        if len(level_counts) <= level:
            level_counts.append(0)

        # Count the current node
        level_counts[level] += 1

        # Recursively count the children
        if isinstance(node, Tree):
            for child in node:
                traverse(child, level + 1)

    traverse(tree, 0)
    return level_counts


def map_tree_to_ids_bfs(tree):
    species_label_to_node_id = {}
    current_id = 0

    # Use a queue to keep track of nodes to process
    queue = deque([tree])

    while queue:
        node = queue.popleft()

        if isinstance(node, Tree):
            node_str = node.label()  # Convert the node to its string representation
        else:
            node_str = node

        if node_str not in species_label_to_node_id and node_str != 'root' and node_str != 'n00001930' and node_str != 'unknown':  # Check to avoid duplicate keys
            species_label_to_node_id[node_str] = current_id
            current_id += 1

        if isinstance(node, Tree):
            for child in node:
                queue.append(child)

    return species_label_to_node_id


def get_descendants(tree):
    # Start with the label of the current node
    descendants = [tree.label()] if isinstance(tree, Tree) else []

    # If the current node is a leaf, just return its label
    if isinstance(tree, str):
        return [tree]

    # Otherwise, recursively collect labels from all children
    for child in tree:
        descendants.extend(get_descendants(child))

    return descendants


def get_distances(logits, projections, n_classes):
    batch_size = logits.shape[0]
    dim = logits.shape[1]

    # Step 1: Apply the projection matrices to each point in logits
    # We avoid expanding logits and use broadcasting directly
    logits = logits.view(batch_size, 1, dim)
    projections = projections.view(1, n_classes, dim, dim)

    # Matrix multiplication of shape (batch_size, 1, dim) @ (1, n_classes, dim, dim) -> (batch_size, n_classes, dim)
    projected_points = torch.einsum('abcd,ead->ebc', projections, logits)

    # Step 2: Compute the Euclidean distance ||logits - projected_points||
    # We use broadcasting directly without expanding
    projection_norm = torch.norm(projected_points, dim=2)
    return projection_norm


def get_orthonormal_vectors(num_classes):
    gaus = np.random.randn(num_classes, num_classes)
    svd = np.linalg.svd(gaus)
    orth = svd[0] @ svd[2]
    return orth


def get_hiercos_parameters(opts, classes, distances, only_fine_labels=True, fa=True, fd=True):
    if opts.feature_space is None:
        return opts, None
    elif only_fine_labels:
        return get_hiercos_parameters_fine_labels(opts, classes, distances)
    else:
        return get_hiercos_parameters_hierarchical_labels(opts, classes, distances, fa, fd)


def get_hiercos_parameters_fine_labels(opts, classes, distances):
    # TODO
    raise NotImplementedError()


def get_hiercos_parameters_hierarchical_labels(opts, classes, distances, fa=True, fd=True):
    # TODO: Construct the level-wise projection matrix for getting hierarchical labels and to compute consistency metric
    hierarchy = load_hierarchy(opts.data, opts.data_dir)
    num_nodes_per_level = count_nodes_at_each_level(hierarchy)[1:]  # ignore the root node
    max_level = len(num_nodes_per_level)

    species_label_to_node_id = map_tree_to_ids_bfs(hierarchy)
    species_node_id_to_label = {v: k for k, v in species_label_to_node_id.items()}
    hiercos_dim = len(species_label_to_node_id)  # dimension of HAFS is equal to the number of nodes in the tree
    opts.num_classes = hiercos_dim

    opts.orthonormal_basis_vectors = get_orthonormal_vectors(hiercos_dim)
    orthonormal_basis_vectors = torch.eye(hiercos_dim, device=opts.gpu, dtype=torch.float32)

    # Projection onto the subspaces corresponding to all the nodes arranged according to their level in the hierarchy tree
    # used to find the projection of a vector onto the subspace corresponding to a node at a specific level
    level_wise_projections = []
    for level in range(max_level):
        level_wise_projections.append(
            torch.zeros((num_nodes_per_level[level], hiercos_dim, hiercos_dim), device=opts.gpu,
                        dtype=torch.float32))

    # Projection onto the subspaces corresponding to leaf classes
    # projections: (num_leaves, hiercos-dim, hiercos-dim)
    # used to find the projection of a vector onto the subspace corresponding to a leaf class
    projections = torch.zeros((len(classes), hiercos_dim, hiercos_dim), device=opts.gpu,
                              dtype=torch.float32)

    level_wise_node_ids = []
    low = 0
    high = num_nodes_per_level[0]
    for level in range(max_level):
        level_wise_node_ids.append(torch.tensor(range(low, high), device=opts.gpu))
        if level < max_level - 1:
            low += num_nodes_per_level[level]
            high += num_nodes_per_level[level + 1]

    leaf_classes = hierarchy.leaves()
    leaf_class_to_hierarchical_labels = []
    level_loss_weights = []
    node_prob_distribution = []

    def get_all_ancestors_and_descendants(tree):
        # Dictionary to store ancestors and all descendants for each node
        node_relations = {}

        # Helper function to recursively get all descendants of a node, including leaf nodes
        def collect_descendants(node):
            descendants = set()
            if isinstance(node, Tree):
                for child in node:
                    if isinstance(child, Tree):
                        descendants.add(species_label_to_node_id[child.label()])
                        descendants.update(collect_descendants(child))
                    elif isinstance(child, str):  # Include leaf nodes
                        descendants.add(species_label_to_node_id[child])
            return descendants

        # Recursive function to traverse the tree and collect relations
        def traverse_tree(node, ancestors=[]):
            # Skip if the node label is 'unknown'
            if isinstance(node, Tree) and (node.label() in ['root', 'unknown', 'n00001930']):
                for child in node:
                    traverse_tree(child, ancestors)
                return

            if isinstance(node, Tree):
                node_label = species_label_to_node_id[node.label()]
                all_descendants = collect_descendants(node)
                node_relations[node_label] = {
                    "ancestors": set(ancestors),
                    "descendants": all_descendants
                }
                # Traverse children and add the current node to ancestors list
                for child in node:
                    traverse_tree(child, ancestors + [node_label])
            elif isinstance(node, str):  # Include leaf nodes as keys in the dictionary
                node_relations[species_label_to_node_id[node]] = {
                    "ancestors": set(ancestors),
                    "descendants": set()  # Leaf nodes have no descendants
                }

        # Start traversal from the root
        traverse_tree(tree)
        return node_relations
    subspace_bases_vectors = get_all_ancestors_and_descendants(hierarchy)

    for level in range(max_level):
        for i, node_id in enumerate(level_wise_node_ids[level]):
            node_id = node_id.item()
            class_idx = i
            if fa:
                for ancestor_node in subspace_bases_vectors[node_id]['ancestors']:
                    level_wise_projections[level][class_idx] += orthonormal_basis_vectors[ancestor_node][:, None] @ orthonormal_basis_vectors[ancestor_node][None, :]
            level_wise_projections[level][class_idx] += orthonormal_basis_vectors[node_id][:, None] @ orthonormal_basis_vectors[node_id][None, :]
            if fd:
                for descendant_node in subspace_bases_vectors[node_id]['descendants']:
                    level_wise_projections[level][class_idx] += orthonormal_basis_vectors[descendant_node][:, None] @ orthonormal_basis_vectors[descendant_node][None, :]

    for i, label in enumerate(leaf_classes):
        class_idx = classes.index(label)
        node_id = species_label_to_node_id[label]
        if fa:
            for ancestor_node in subspace_bases_vectors[node_id]['ancestors']:
                projections[class_idx] += orthonormal_basis_vectors[ancestor_node][:, None] @ \
                                                            orthonormal_basis_vectors[ancestor_node][None, :]
        projections[class_idx] += orthonormal_basis_vectors[node_id][:, None] @ \
                                  orthonormal_basis_vectors[node_id][None, :]
        if fd:
            for descendant_node in subspace_bases_vectors[node_id]['descendants']:
                projections[class_idx] += orthonormal_basis_vectors[descendant_node][:, None] @ \
                                                            orthonormal_basis_vectors[descendant_node][None, :]

    for class_idx, class_ in enumerate(classes):
        class_hierarchy_labels = []
        leaf_index = leaf_classes.index(class_)
        tree_location = hierarchy.leaf_treeposition(leaf_index)
        for i in range(len(tree_location)):
            try:
                label = hierarchy[tree_location[:i + 1]].label()
            except:
                label = hierarchy[tree_location[:i + 1]]
            class_hierarchy_labels.append(species_label_to_node_id[label])

        # Level-wise 0-indexing for level-wise cross-entropy
        leaf_class_to_hierarchical_labels.append(
            [hl - level_wise_node_ids[level].min().item() for level, hl in enumerate(class_hierarchy_labels)])
        level_loss_weights.append(
            [(level + 1) / len(class_hierarchy_labels) for level in range(len(class_hierarchy_labels))])
        arr = np.arange(len(class_hierarchy_labels), 0, -1)  # 5 4 3 2 1
        node_prob_distribution.append(np.square(np.exp(1 / arr) / np.linalg.norm(np.exp(1 / arr))).tolist())
        # Append -100 to hierarchical labels for the nodes that are not at leaf
        if len(class_hierarchy_labels) < max_level:
            for _ in range(len(class_hierarchy_labels), max_level):
                leaf_class_to_hierarchical_labels[-1].append(-100)
                level_loss_weights[-1].append(0)
                node_prob_distribution[-1].append(0)
    leaf_class_to_hierarchical_labels = torch.tensor(leaf_class_to_hierarchical_labels, device=opts.gpu)
    level_loss_weights = torch.tensor(level_loss_weights, device=opts.gpu)
    node_prob_distribution = torch.tensor(node_prob_distribution, device=opts.gpu)

    # Store the similarity between leaf classes based on HAFrame's cosine similarity method.
    # This is for visualizing the hierarchies and will be required for computing HOPS
    # Note: For computing HOPS, we do not use the absolute values from LCA_similarity.npy. Instead, we only require a
    # relative measure via which we can rank the classes. Therefore, the hyperparameter gamma would neitehr impact the
    # Hier-COS' performance nor HOPS'
    distance_matrix = distance_dict_to_mat(distances, classes)
    distance_matrix = torch.tensor(map_hdistance_to_cosine_similarity_exponential_decay(
        hdistance=distance_matrix,
        gamma=opts.haf_gamma,
        min_similarity=0
    ), device=opts.gpu)
    distance_matrix = (distance_matrix - distance_matrix.min(dim=1)[0][:, None]) / (distance_matrix.max(dim=1)[0] - distance_matrix.min(dim=1)[0])[:, None]
    if not os.path.exists(f'{opts.out_folder}/LCA_similarity.npy'):
        with open(f'{opts.out_folder}/LCA_similarity.npy', 'wb') as f:
            np.save(f, distance_matrix.detach().cpu().numpy())

    hiercos_parameters = {
        "gpu": opts.gpu,
        "alpha": opts.alpha,
        "orthonormal_basis_vectors": orthonormal_basis_vectors,
        "projections": projections,
        "level_wise_projections": level_wise_projections,
        "level_wise_node_ids": level_wise_node_ids,
        "leaf_class_to_hierarchical_labels": leaf_class_to_hierarchical_labels,
        "level_loss_weights": level_loss_weights,
        "node_prob_distribution": node_prob_distribution,
        "distance_matrix": distance_matrix,
        "leaf_classes": len(classes),
        "hiercos_dim": hiercos_dim,
        "num_nodes_per_level": num_nodes_per_level
    }

    return opts, hiercos_parameters


class HAFS_Loss(torch.nn.Module):
    def __init__(self, hiercos_params, log_level_wise_performance=False):
        super(HAFS_Loss, self).__init__()
        self.hiercos_params = hiercos_params
        self.log_level_wise_performance = log_level_wise_performance
        self.kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
        self.fpa = []
        self.la = []
        self.predictions = []
        self.labels = []
        self.log = 0

    def forward(self, logits, labels):
        if self.log_level_wise_performance:
            level_wise_accuracy = []
        out = get_distances(logits, self.hiercos_params["projections"], self.hiercos_params["leaf_classes"])

        # Log the cosine similarity between x and Px
        if self.log % 100 == 0:
            cos_sim = (out / logits.norm(dim=1)[:, None])
            angle = torch.rad2deg(torch.acos(cos_sim))
            print("Mean angle with 4 closest negative subspaces", np.round(angle.topk(self.hiercos_params["leaf_classes"])[0][:, -5:-1].mean().item(), 2))
            print("Mean angle with predicted subspace", np.round(angle.topk(self.hiercos_params["leaf_classes"])[0][:, -1].mean().item(), 2))
            print("Mean angle with positive subspace", np.round(angle[range(labels.shape[0]), labels].mean().item(), 2))
        max_level = self.hiercos_params["leaf_class_to_hierarchical_labels"].shape[1]
        fpa = None
        l_gt = torch.zeros(logits.shape[0], logits.shape[1], device=self.hiercos_params['gpu'])
        l_aux = 0

        for level in range(max_level):
            level_labels = self.hiercos_params["leaf_class_to_hierarchical_labels"][labels, level]
            level_projections = torch.matmul(logits, self.hiercos_params["orthonormal_basis_vectors"][self.hiercos_params["level_wise_node_ids"][level]].t()).abs()
            if level_labels[level_labels > 0].shape[0] > 0:
                level_cos_sim = (level_projections / level_projections.norm(dim=1)[:, None])[level_labels >= 0]
                ohe_labels = F.one_hot(level_labels[level_labels >= 0], num_classes=self.hiercos_params["num_nodes_per_level"][level])
                l_aux += (ohe_labels - level_cos_sim).abs().sum(1).mean()
            if (level_labels >= 0).prod() == 0:
                l_aux += level_projections[level_labels < 0].sum(1).mean()

            l_gt[level_labels >= 0, self.hiercos_params["level_wise_node_ids"][level].min().item(): self.hiercos_params["level_wise_node_ids"][level].max().item() + 1] = F.one_hot(level_labels[level_labels >= 0], num_classes=self.hiercos_params["num_nodes_per_level"][level]) * self.hiercos_params["node_prob_distribution"][labels[level_labels >= 0], level, None]
            if self.log % 100 == 0:
                # Log level-wise accuracy
                print(f"[{level}]", (level_projections[level_labels >= 0].argmax(1) == level_labels[level_labels >= 0]).sum() / level_labels[level_labels >= 0].shape[0])

            if self.log_level_wise_performance:
                # if level == max_level - 1:
                #     level_labels = labels
                level_projections = get_distances(logits, self.hiercos_params["level_wise_projections"][level], self.hiercos_params["num_nodes_per_level"][level])
                level_wise_accuracy.append((level_projections.argmax(1)[level_labels >= 0] == level_labels[level_labels >= 0]))
                if len(self.predictions) <= level:
                    self.predictions.append([])
                    self.labels.append([])
                self.labels[level].extend(level_labels[level_labels >= 0].detach().cpu().numpy())
                self.predictions[level].extend(level_projections[level_labels >= 0].detach().cpu().numpy())
                if len(self.la) < max_level:
                    if level_labels[level_labels >= 0].shape[0] == 0:
                        self.la.append(1)
                    else:
                        self.la.append((level_wise_accuracy[-1].sum() / level_labels[level_labels >= 0].shape[0]).tolist())
                else:
                    if level_labels[level_labels >= 0].shape[0] == 0:
                        self.la[level] += 1
                    else:
                        self.la[level] += (level_wise_accuracy[-1].sum() / level_labels[level_labels >= 0].shape[0]).tolist()
                if fpa is None:
                    fpa = level_wise_accuracy[-1]
                else:
                    fpa[level_labels >= 0] = (fpa[level_labels >= 0] & level_wise_accuracy[-1])

        if self.log_level_wise_performance:
            self.fpa.append((fpa.sum() / fpa.shape[0]).item())

        l_kl = self.kl_loss(torch.log_softmax(logits.abs(), dim=1), l_gt)
        loss = l_kl + self.hiercos_params["alpha"] * l_aux
        self.log += 1
        return loss, out