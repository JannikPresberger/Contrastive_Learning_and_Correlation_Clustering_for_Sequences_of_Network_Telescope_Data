import torch
import pandas as pd
import umap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import polars as pl
import os
import csv

import andres_graph

import plotly.express as px

from data import IPAddressAsSequenceDataset, DatasetConfig , collate_ip_batch
from create_anchors import AnchorCreationConfig, get_or_create_anchor_file, AnchorLabelFilter


def print_cuda_memory():
    if torch.cuda.is_available():
        print(f"CUDA Memory - Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB, Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    else:
        print("CUDA not available")

def iterate_ip_chunks(df, chunk_size):
    n = df.height

    for i in range(0, n, chunk_size):
        print("Yielding IP chunk:", i, "from", n)
        yield df.slice(i,chunk_size)

def ipv4_int_to_str(ip: int) -> str:
    return ".".join(str((ip >> (8 * i)) & 0xFF) for i in reversed(range(4)))

def ipv4_to_uint32(ipv4_tensor):
    octets = ipv4_tensor.view(-1, 4)
    uint32 = (octets[:, 0].long() << 24) | (octets[:, 1].long() << 16) | (octets[:, 2].long() << 8) | octets[:, 3].long()
    return uint32

def edge_metrics_to_csv(metrics, filepath):

    file_already_exists = os.path.isfile(filepath)

    with open(filepath, mode='a', newline='') as f:
        writer = csv.writer(f)

        if not file_already_exists:
            writer.writerow(metrics.keys())

        writer.writerow(metrics.values())
        

def compute_edge_metrics_np(edge_pred, edge_gt,offset):
    """
    edge_pred: numpy array [E] with {0=join, 1=cut}
    edge_gt:   numpy array [E] with {0=join, 1=cut}
    """

    edge_pred = edge_pred.astype(np.int64)
    edge_gt = edge_gt.astype(np.int64)

    tp = np.sum((edge_pred == 1) & (edge_gt == 1))  # correct cuts
    tn = np.sum((edge_pred == 0) & (edge_gt == 0))  # correct joins
    fp = np.sum((edge_pred == 1) & (edge_gt == 0))  # false cuts
    fn = np.sum((edge_pred == 0) & (edge_gt == 1))  # missed cuts

    eps = 1e-8

    precision_cut = tp / (tp + fp + eps)
    recall_cut    = tp / (tp + fn + eps)
    f1_cut        = 2 * tp / (2 * tp + fp + fn + eps)

    precision_join = tn / (tn + fn + eps)
    recall_join    = tn / (tn + fp + eps)
    f1_join        = 2 * tn / (2 * tn + fp + fn + eps)

    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    # macro F1
    f1_macro = 0.5 * (f1_cut + f1_join)

    # weighted F1
    num_cut = np.sum(edge_gt == 1)
    num_join = np.sum(edge_gt == 0)
    total = num_cut + num_join + eps

    f1_weighted = (num_cut / total) * f1_cut + (num_join / total) * f1_join

    return {
        "offset": offset,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,

        "precision_cut": precision_cut,
        "recall_cut": recall_cut,
        "f1_cut": f1_cut,

        "precision_join": precision_join,
        "recall_join": recall_join,
        "f1_join": f1_join,

        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }

def run_correlation_clustering(embeddings, labels, experiment_base_name:str):

    print("Running correlation clustering with andres_graph...")
    print_cuda_memory()

    num_total = embeddings.size(0)
    num_to_sample = min(10_000, num_total)

    perm = torch.randperm(num_total)[:num_to_sample]

    embeddings = embeddings[perm]
    labels = labels[perm].cpu().numpy()

    similarity_matrix = torch.matmul(embeddings, embeddings.T)

    print("Computed similarity matrix")
    print_cuda_memory()

    N = embeddings.size(0)

    i_idx, j_idx = torch.triu_indices(N, N, offset=1)

    edge_index = torch.stack([i_idx, j_idx], dim=0).T

    print("Computed indices")
    print_cuda_memory()

    
    edge_weight = similarity_matrix[i_idx, j_idx]
    edge_index = edge_index.cpu().numpy()
    edge_weight = edge_weight.cpu().numpy()

    print("Computed edge weights")
    print_cuda_memory()

    for x in range(100):

        print(f"Running correlation clustering with offset {x/100:.1f}...")

        cur_offset = x / 100

        cur_edge_weights = edge_weight - cur_offset

        edge_u = edge_index[:, 0]
        edge_v = edge_index[:, 1]

        UNKNOWN_EDGE = -1

        known_mask = (labels[edge_u] != 0) & (labels[edge_v] != 0)

        # if lables are the same, edge_gt = 0, else 1
        edge_gt = np.full(edge_u.shape, UNKNOWN_EDGE, dtype=np.int8)
        edge_gt[known_mask] = (labels[edge_u[known_mask]] != labels[edge_v[known_mask]])

        valid_mask = edge_gt != UNKNOWN_EDGE

        edge_pred = andres_graph.greedy_additive_edge_contraction(N, edge_index, cur_edge_weights)

        edge_gt_eval = edge_gt[valid_mask]
        edge_pred_eval = edge_pred[valid_mask]

        metrics = compute_edge_metrics_np(edge_pred_eval, edge_gt_eval,cur_offset)

        edge_metrics_to_csv(metrics, experiment_base_name + "_correlation_clustering_edge_metrics.csv")



def cosine_distributions_chunked(embeddings,labels,labels_name:str,experiment_base_name:str,chunk_size:int=8192,exclusion_labels = None):

    total_num_embeddings = embeddings.size(0)

    total_intra_sum = 0.0
    total_inter_sum = 0.0

    total_intra_count = 0
    total_inter_count = 0

    num_bins = 50

    bin_edges = torch.linspace(-1, 1, num_bins + 1)
    intra_hist = torch.zeros(num_bins)
    inter_hist = torch.zeros(num_bins)

    for i in range (0, total_num_embeddings, chunk_size):
        for j in range (0, total_num_embeddings, chunk_size):
            chunk_similarity_matrix = torch.matmul(embeddings[i:i+chunk_size],embeddings[j:j+chunk_size].T)

            i_idx = torch.arange(i, min(i+chunk_size, total_num_embeddings),device=chunk_similarity_matrix.device)
            j_idx = torch.arange(j, min(j+chunk_size, total_num_embeddings),device=chunk_similarity_matrix.device)

            chunk_same_label_mask = (labels[i:i+chunk_size].unsqueeze(1) == labels[j:j+chunk_size].unsqueeze(0))

            diag_mask = i_idx.unsqueeze(1) == j_idx.unsqueeze(0)

            exclusion_mask = torch.zeros_like(chunk_same_label_mask)

            if exclusion_labels is not None:
                exclusion_mask = exclusion_mask | (exclusion_labels[i:i+chunk_size].unsqueeze(1) == exclusion_labels[j:j+chunk_size].unsqueeze(0))

            exclusion_mask = exclusion_mask | diag_mask

            intra_cluster_similiarities = chunk_similarity_matrix[chunk_same_label_mask & ~exclusion_mask]
            inter_cluster_similiarities = chunk_similarity_matrix[~chunk_same_label_mask & ~exclusion_mask]

            total_intra_sum += intra_cluster_similiarities.sum().item()
            total_inter_sum += inter_cluster_similiarities.sum().item()

            total_intra_count += intra_cluster_similiarities.numel()
            total_inter_count += inter_cluster_similiarities.numel()

            print(f"Processed chunk ({i}, {j}): intra sum={intra_cluster_similiarities.sum().item():.2f}, inter sum={inter_cluster_similiarities.sum().item():.2f}, intra count={intra_cluster_similiarities.numel()}, inter count={inter_cluster_similiarities.numel()}")

            intra_hist += torch.histogram(intra_cluster_similiarities.cpu(), bins=bin_edges)[0]
            inter_hist += torch.histogram(inter_cluster_similiarities.cpu(), bins=bin_edges)[0]

    intra_mean = total_intra_sum / total_intra_count if total_intra_count > 0 else 0.0
    inter_mean = total_inter_sum / total_inter_count if total_inter_count > 0 else 0.0

    append_means_to_csv(experiment_base_name + "_similarity_means.csv", labels_name, intra_mean, inter_mean)

    intra_hist = intra_hist / intra_hist.sum() if intra_hist.sum() > 0 else intra_hist
    inter_hist = inter_hist / inter_hist.sum() if inter_hist.sum() > 0 else inter_hist

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    df = pd.DataFrame({
        "bin_center": bin_centers.detach().cpu().numpy(),
        "intra_density": intra_hist.detach().cpu().numpy(),
        "inter_density": inter_hist.detach().cpu().numpy()
    })

    df.to_csv(experiment_base_name + "_" + labels_name + "_similarity_histogram.csv", index=False)


def compute_histograms(intra, inter, bins=50):
    
    intra_hist = torch.histogram(intra.cpu(), bins=bins, range=(-1,1),density=True)
    inter_hist = torch.histogram(inter.cpu(), bins=bins, range=(-1,1),density=True)

    bin_centers = (intra_hist[1][:-1] + intra_hist[1][1:]) / 2

    return intra_hist, inter_hist, bin_centers

def cosine_distributions(embeddings,labels,labels_name:str,experiment_base_name:str,chunk_size:int=8192):

    similarity_matrix = torch.matmul(embeddings,embeddings.T)

    same_source_mask = (labels.unsqueeze(0) == labels.unsqueeze(1))

    identity_mask = ~torch.eye(similarity_matrix.size(0),dtype=torch.bool).to(similarity_matrix.device)

    intra_cluster_similiarities = similarity_matrix[same_source_mask & identity_mask]
    inter_cluster_similiarities = similarity_matrix[~same_source_mask]

    intra_mean = intra_cluster_similiarities.mean().item()
    inter_mean = inter_cluster_similiarities.mean().item()

    print(f"Intra-cluster similarity mean: {intra_mean:.4f}")
    print(f"Inter-cluster similarity mean: {inter_mean:.4f}")
    print(f"Difference: {intra_mean - inter_mean:.4f}")

    intra_hist,inter_hist,bin_centers = compute_histograms(intra_cluster_similiarities, inter_cluster_similiarities)

    print("Intra-cluster histogram:", intra_hist[0] / intra_hist[0].sum())
    print("Inter-cluster histogram:", inter_hist[0] / inter_hist[0].sum())
    
    df = pd.DataFrame({
    "bin_center": bin_centers.detach().cpu().numpy(),
    "intra_density": intra_hist[0].detach().cpu().numpy(),
    "inter_density": inter_hist[0].detach().cpu().numpy()
    })

    df.to_csv(experiment_base_name + "_" + labels_name + "_similarity_histogram.csv", index=False)



def umap_visualization(embeddings, source_ips,scanner_ids,evaluation_base_name:str, n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine',max_points=100_000):
    
    embeddings_np = embeddings.cpu().numpy()
    source_ips_np = source_ips.cpu().numpy()

    if len(embeddings_np) > max_points:
        idx = np.random.choice(len(embeddings_np), max_points, replace=False)
        embeddings_np = embeddings_np[idx]
        source_ips_np = source_ips_np[idx]

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric)
    embedding_2d = reducer.fit_transform(embeddings_np)

    df = pd.DataFrame({
        "embedding_1": embedding_2d[:, 0],
        "embedding_2": embedding_2d[:, 1],
        "source_ip": source_ips_np
    })

    plt.figure(figsize=(10, 8))
    plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=source_ips_np,
        cmap="tab20",
        s=1,
        alpha=0.5,
        rasterized=True 
    )
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title("UMAP Projection (Rasterized)")
    plt.tight_layout()
    plt.savefig(evaluation_base_name +"_umap.png", dpi=300)

    plt.close()

    plt.figure(figsize=(10, 8))
    plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=scanner_ids,
        cmap="tab20",
        s=1,
        alpha=0.5,
        rasterized=True 
    )
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title("UMAP Projection (Rasterized)")
    plt.tight_layout()
    plt.savefig(evaluation_base_name +"_scanner_umap.png", dpi=300)

    plt.close()


def tsne_visualization(
    embeddings,
    labels,
    evaluation_base_name: str,
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    n_iter=1000,
    metric="cosine",
    max_points=100_000,        # safety limit
    random_state=42
):
    
    embeddings_np = embeddings.cpu().numpy().astype(np.float32)
    labels_np = labels.cpu().numpy()

    if len(embeddings_np) > max_points:
        idx = np.random.choice(len(embeddings_np), max_points, replace=False)
        embeddings_np = embeddings_np[idx]
        labels_np = labels_np[idx]


    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=n_iter,
        metric=metric,
        init="pca",                 # more stable than random
        method="barnes_hut",        # critical for scaling
        random_state=random_state,
        verbose=1
    )

    embedding_2d = tsne.fit_transform(embeddings_np)

    labels_codes = pd.Categorical(labels_np).codes

    df = pd.DataFrame({
        "embedding_1": embedding_2d[:, 0],
        "embedding_2": embedding_2d[:, 1],
        "label": labels_np,
    })

    df.to_csv(evaluation_base_name + "_tsne_visualization.csv", index=False)

    plt.figure(figsize=(10, 8))
    plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=labels_codes,
        cmap="tab20",
        s=1,
        alpha=0.5,
        rasterized=True
    )
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")
    plt.title("t-SNE Projection (Rasterized)")
    plt.tight_layout()
    plt.savefig(evaluation_base_name + "_tsne.png", dpi=300)

    plt.close()

def visualize_tsne_csv(
    file_path: str,
    x_col: str = "embedding_1",
    y_col: str = "embedding_2",
    label_col: str = "scanner_id",
    hover_cols: list = ["mirai","zmap","ip_str"],
    opacity: float = 0.6,
    point_size: int = 4
):
    df = pl.read_csv(file_path)

    required_cols = {x_col, y_col, label_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")


    df = df.with_columns(
        pl.col("source_ip").map_elements(ipv4_int_to_str).alias("ip_str")
    )

    pdf = df.to_pandas()

    fig = px.scatter(
        pdf,
        x=x_col,
        y=y_col,
        color=label_col,
        hover_data=[col for col in hover_cols if col in pdf.columns],
        opacity=opacity
    )

    fig.update_traces(marker=dict(size=point_size))
    fig.update_layout(
        title="Interactive t-SNE Visualization",
        legend_title="Label"
    )

    fig.show()


def append_means_to_csv(filepath, label_name, intra_mean, inter_mean):
    file_exists = os.path.isfile(filepath)

    diff = intra_mean - inter_mean

    with open(filepath, mode='a', newline='') as f:
        writer = csv.writer(f)

        # write header only if file is new
        if not file_exists:
            writer.writerow(['label_name', 'intra_mean', 'inter_mean', 'diff'])

        writer.writerow([label_name, intra_mean, inter_mean, diff])

def embed_anchor_sequences(model, dataset_config:DatasetConfig,anchor_config:AnchorCreationConfig,device,use_backbone_embeddings,experiment_base_path:str,chunk_size:int):

    model.eval()

    all_labeled_embeddings = []
    all_labeled_source_ips = []
    all_labeled_scanner_ids = []
    all_labeled_mirai_flags = []
    all_labeled_zmap_flags = []

    all_embeddings = []
    all_source_ips = []

    all_ips_to_eval = dataset_config.all_ips_in_dataset

    chunk_idx = 0

    append_chunk_data = False

    append_label_data = True

    append_all_data = False

    anchor_positions_raw_df = get_or_create_anchor_file(all_ips_to_eval,anchor_config,dataset_config.path_to_dataset)#pl.read_parquet(dataset_config.path_to_dataset + "/anchors/anchor_positions.parquet")
    anchor_positions = {row["ip.src"]: row["anchors"] for row in anchor_positions_raw_df.iter_rows(named=True)}

    evaluation_name = os.path.join(experiment_base_path, anchor_config.subset_type)

    all_labels_by_name = {col.column_name : [] for col in anchor_config.label_columns if col.column_name != "ip.src"}

    with torch.no_grad():
        for ip_chunk in iterate_ip_chunks(all_ips_to_eval, chunk_size):

            ip_chunk = ip_chunk.with_columns(pl.lit(-1).alias("shard_idx"))

            chunk_config = DatasetConfig(   dataset_config.path_to_dataset,
                                            all_ips_in_dataset=ip_chunk,            
                                            max_sequence_length=dataset_config.max_sequence_length, 
                                            prob_estimation_window_length=dataset_config.prob_estimation_window_length, 
                                            sequence_overlap=dataset_config.sequence_overlap, 
                                            sequence_split_part=dataset_config.sequence_split_part,
                                            sequence_split_mode=dataset_config.sequence_split_mode, 
                                            sequence_split_ratio=dataset_config.sequence_split_ratio, 
                                            sequence_split_seed=dataset_config.sequence_split_seed, 
                                            sampling_config=None,
                                            anchor_positions=anchor_positions)
            
            chunk_dataset = IPAddressAsSequenceDataset(
                source_ips=chunk_config.all_ips_in_dataset,
                path_to_dataset=chunk_config.path_to_dataset,
                max_sequence_length=chunk_config.max_sequence_length,
                prob_estimation_window_length=chunk_config.prob_estimation_window_length,
                sequence_overlap=chunk_config.sequence_overlap,
                sequence_split_mode=chunk_config.sequence_split_mode,
                sequence_split_part=chunk_config.sequence_split_part,
                sequence_split_ratio=chunk_config.sequence_split_ratio,
                sequence_split_seed=chunk_config.sequence_split_seed,
                config=chunk_config
            )

            chunk_loader = torch.utils.data.DataLoader(chunk_dataset, batch_size=512, shuffle=False, collate_fn=collate_ip_batch)

            for batch in chunk_loader:

                for key, val in batch.items():
                    batch[key] = val.to(device)

                embeddings = model(batch,use_backbone_embeddings)
                if torch.isnan(embeddings).any():
                    print("Warning: NaN values found in embeddings before normalization in chunk", chunk_idx)
                
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                if torch.isnan(embeddings).any():
                    print("Warning: NaN values found in embeddings AFTER normalization in chunk", chunk_idx)
                
                source_ips = batch['ip.src'].to(torch.int64)

                all_embeddings.append(embeddings)
                all_source_ips.append(source_ips)

                for col_name, col_label_list in all_labels_by_name.items():
                    col_label_list.append(batch[col_name][:,0])

            chunk_idx += 1


    all_embeddings = torch.cat(all_embeddings,dim=0)
    all_source_ips = torch.cat(all_source_ips,dim=0)

    for col_name, col_label_list in all_labels_by_name.items():
        all_labels_by_name[col_name] = torch.cat(col_label_list,dim=0)

    return all_embeddings, all_source_ips, all_labels_by_name


def evaluate_cosine_distributions_for_source_identities(model,ip_unseen_sequences_config:DatasetConfig, ip_unseen_sources_config:DatasetConfig, device,output_base_path:str,chunk_size=10_000):

    model.eval()

    unseen_sequences_anchor_config = AnchorCreationConfig(
        dataset_name=os.path.basename(ip_unseen_sequences_config.path_to_dataset).replace(".parquet",""),
        subset_type="unseen_sequences",
        max_sequence_length=256,
        num_anchors_to_create=50000,
        sequence_split_part="test",
        sequence_split_ratio=ip_unseen_sequences_config.sequence_split_ratio,
        anchor_sampling_alpha=0.5,
        label_columns=[ AnchorLabelFilter("all_sources","ip.src","!=",[0],1.0)])
    
    anchor_embeddings_unseen_seq,anchor_src_ip_unseen_seq, anchor_labels =  embed_anchor_sequences(model, ip_unseen_sequences_config,unseen_sequences_anchor_config,device,False,output_base_path,chunk_size)
    cosine_distributions_chunked(anchor_embeddings_unseen_seq, anchor_src_ip_unseen_seq,"src_IP_all",os.path.join(output_base_path, unseen_sequences_anchor_config.subset_type),10000)
    
    unseen_sources_anchor_config = AnchorCreationConfig(
        dataset_name=os.path.basename(ip_unseen_sources_config.path_to_dataset).replace(".parquet",""),
        subset_type="unseen_sources",
        max_sequence_length=256,
        num_anchors_to_create=50000,
        sequence_split_part="test",
        sequence_split_ratio=0.0,
        anchor_sampling_alpha=0.5,
        label_columns=[ AnchorLabelFilter("all_sources","ip.src","!=",[0],1.0)])
    
    anchor_embeddings_unseen_sources,anchor_src_ip_unseen_sources, anchor_labels =  embed_anchor_sequences(model, ip_unseen_sources_config,unseen_sources_anchor_config,device,False,output_base_path,chunk_size)
    cosine_distributions_chunked(anchor_embeddings_unseen_sources, anchor_src_ip_unseen_sources,"src_IP_all",os.path.join(output_base_path, unseen_sources_anchor_config.subset_type),10000)

def evaluate_cosine_distributions_for_scanner_ids(model,ip_unseen_sequences_config:DatasetConfig, ip_unseen_sources_config:DatasetConfig, device,output_base_path:str,chunk_size=10_000):

    model.eval()

    unseen_sequences_anchor_config = AnchorCreationConfig(
        dataset_name=os.path.basename(ip_unseen_sequences_config.path_to_dataset).replace(".parquet",""),
        subset_type="unseen_sequences_of_scanners",
        max_sequence_length=256,
        num_anchors_to_create=50000,
        sequence_split_part="test",
        sequence_split_ratio=ip_unseen_sequences_config.sequence_split_ratio,
        anchor_sampling_alpha=0.5,
        label_columns=[ AnchorLabelFilter("scanners","acknowledged_scanner","!=",[0],1.0)])
    
    anchor_embeddings_unseen_seq,anchor_src_ip_unseen_seq, anchor_labels_unseen_sequences =  embed_anchor_sequences(model, ip_unseen_sequences_config,unseen_sequences_anchor_config,device,False,output_base_path,chunk_size)
    cosine_distributions_chunked(anchor_embeddings_unseen_seq, anchor_labels_unseen_sequences['acknowledged_scanner'],"scanner_IDS",os.path.join(output_base_path, unseen_sequences_anchor_config.subset_type),10000,exclusion_labels=anchor_src_ip_unseen_seq)
    
    unseen_sources_anchor_config = AnchorCreationConfig(
        dataset_name=os.path.basename(ip_unseen_sources_config.path_to_dataset).replace(".parquet",""),
        subset_type="unseen_sources_of_scanners",
        max_sequence_length=256,
        num_anchors_to_create=50000,
        sequence_split_part="test",
        sequence_split_ratio=0.0,
        anchor_sampling_alpha=0.5,
        label_columns=[ AnchorLabelFilter("scanners","acknowledged_scanner","!=",[0],1.0)])
    
    anchor_embeddings_unseen_sources,anchor_src_ip_unseen_sources, anchor_labels_unseen_sources =  embed_anchor_sequences(model, ip_unseen_sources_config,unseen_sources_anchor_config,device,False,output_base_path,chunk_size)
    cosine_distributions_chunked(anchor_embeddings_unseen_sources, anchor_labels_unseen_sources['acknowledged_scanner'],"scanner_IDS",os.path.join(output_base_path, unseen_sources_anchor_config.subset_type),10000,exclusion_labels=anchor_src_ip_unseen_sources)

def run_clustering_evaluation(model,ip_seen_sequences_config:DatasetConfig,ip_unseen_sequences_config:DatasetConfig, ip_unseen_sources_config:DatasetConfig, device,output_base_path:str,chunk_size=10_000):

    seen_sequences_of_scanners_anchor_config = AnchorCreationConfig(
        dataset_name=os.path.basename(ip_seen_sequences_config.path_to_dataset).replace(".parquet",""),
        subset_type="seen_sequences_of_scanners",
        max_sequence_length=256,
        num_anchors_to_create=10000,
        sequence_split_part="train",
        sequence_split_ratio=ip_seen_sequences_config.sequence_split_ratio,
        anchor_sampling_alpha=0.5,
        label_columns=[ AnchorLabelFilter("scanners","acknowledged_scanner","!=",[0],0.5),
                        AnchorLabelFilter("no_scanners","acknowledged_scanner","==",[0],0.5)]
    )

    anchor_embeddings_of_seen_scanners,anchor_src_ip_of_seen_scanners, anchor_labels_of_seen_scanners =  embed_anchor_sequences(model, ip_seen_sequences_config,seen_sequences_of_scanners_anchor_config,device,False,output_base_path,chunk_size)
    tsne_visualization(anchor_embeddings_of_seen_scanners, anchor_src_ip_of_seen_scanners,os.path.join(output_base_path, seen_sequences_of_scanners_anchor_config.subset_type) + "_source_ips")
    tsne_visualization(anchor_embeddings_of_seen_scanners, anchor_labels_of_seen_scanners["acknowledged_scanner"],os.path.join(output_base_path, seen_sequences_of_scanners_anchor_config.subset_type) + "_scanner_ids")

    run_correlation_clustering(anchor_embeddings_of_seen_scanners,anchor_labels_of_seen_scanners["acknowledged_scanner"],os.path.join(output_base_path, seen_sequences_of_scanners_anchor_config.subset_type) + "_scanner_ids")


    unseen_sequences_of_scanners_anchor_config = AnchorCreationConfig(
        dataset_name=os.path.basename(ip_unseen_sequences_config.path_to_dataset).replace(".parquet",""),
        subset_type="unseen_sequences_of_scanners",
        max_sequence_length=256,
        num_anchors_to_create=10000,
        sequence_split_part="test",
        sequence_split_ratio=ip_unseen_sequences_config.sequence_split_ratio,
        anchor_sampling_alpha=0.5,
        label_columns=[ AnchorLabelFilter("scanners","acknowledged_scanner","!=",[0],0.5),
                        AnchorLabelFilter("no_scanners","acknowledged_scanner","==",[0],0.5)]
    )

    anchor_embeddings_of_unseen_scanners,anchor_src_ip_of_unseen_scanners, anchor_labels_of_unseen_scanners =  embed_anchor_sequences(model, ip_unseen_sequences_config,unseen_sequences_of_scanners_anchor_config,device,False,output_base_path,chunk_size)
    run_correlation_clustering(anchor_embeddings_of_unseen_scanners,anchor_labels_of_unseen_scanners["acknowledged_scanner"],os.path.join(output_base_path, unseen_sequences_of_scanners_anchor_config.subset_type) + "_scanner_ids")
    tsne_visualization(anchor_embeddings_of_unseen_scanners, anchor_src_ip_of_unseen_scanners,os.path.join(output_base_path, unseen_sequences_of_scanners_anchor_config.subset_type) + "_source_ips")
    tsne_visualization(anchor_embeddings_of_unseen_scanners, anchor_labels_of_unseen_scanners["acknowledged_scanner"],os.path.join(output_base_path, unseen_sequences_of_scanners_anchor_config.subset_type) + "_scanner_ids")


    unseen_sources_of_scanners_anchor_config = AnchorCreationConfig(
        dataset_name=os.path.basename(ip_unseen_sources_config.path_to_dataset).replace(".parquet",""),
        subset_type="unseen_sources_of_scanners",
        max_sequence_length=256,
        num_anchors_to_create=10000,
        sequence_split_part="test",
        sequence_split_ratio=0.0,
        anchor_sampling_alpha=0.5,
        label_columns=[ AnchorLabelFilter("scanners","acknowledged_scanner","!=",[0],0.5),
                        AnchorLabelFilter("no_scanners","acknowledged_scanner","==",[0],0.5)]
    )

    anchor_embeddings_of_unseen_sources,anchor_src_ip_of_unseen_src_scanners, anchor_labels_of_unseen_sources =  embed_anchor_sequences(model, ip_unseen_sources_config,unseen_sources_of_scanners_anchor_config,device,False,output_base_path,chunk_size)
    run_correlation_clustering(anchor_embeddings_of_unseen_sources,anchor_labels_of_unseen_sources["acknowledged_scanner"],os.path.join(output_base_path, unseen_sources_of_scanners_anchor_config.subset_type) + "_scanner_ids")
    tsne_visualization(anchor_embeddings_of_unseen_sources, anchor_src_ip_of_unseen_src_scanners,os.path.join(output_base_path, unseen_sources_of_scanners_anchor_config.subset_type) + "_source_ips")
    tsne_visualization(anchor_embeddings_of_unseen_sources, anchor_labels_of_unseen_sources["acknowledged_scanner"],os.path.join(output_base_path, unseen_sources_of_scanners_anchor_config.subset_type) + "_scanner_ids")


def run_evaluation(model,ip_seen_sequences_config:DatasetConfig,ip_unseen_sequences_config:DatasetConfig, ip_unseen_sources_config:DatasetConfig, device,output_base_path:str,chunk_size=10_000):

    model.eval()

    evaluate_cosine_distributions_for_source_identities(model,ip_unseen_sequences_config, ip_unseen_sources_config, device,output_base_path,chunk_size)

    evaluate_cosine_distributions_for_scanner_ids(model,ip_unseen_sequences_config, ip_unseen_sources_config, device,output_base_path,chunk_size)

    run_clustering_evaluation(model,ip_seen_sequences_config,ip_unseen_sequences_config, ip_unseen_sources_config, device,output_base_path,chunk_size)
