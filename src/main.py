import os

import torch

import multiprocessing as mp

from transformer import TransformerBackbone, ProjectionHead, IPSeqClassificationTransformer

from data import IPSamplingConfig, DatasetConfig, DataloaderConfig, contrastive_collate, read_ip_stats_parquet, shuffle_and_split_source_ips, make_contrastive_train_dataset_factory, make_contrastive_dataloader_factory, DataManager,compact_parquet_dataset
from evaluation import run_evaluation
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from train_constrastive_ip_seq import train_contrastive_sequence_comparison

from create_anchors import get_or_create_anchor_file, AnchorCreationConfig, AnchorLabelFilter
from run_config import RunConfig, config_to_dict, config_to_json, config_hash, save_config, load_config

from pathlib import Path

def main(run_config: RunConfig = None):
    torch.manual_seed(1)

    # You can use the write_csv_as_parquet(...) and compact_parquet_dataset(...) functions to create a compact parquet dataset from raw CSV files.
    # A simple synthetic csv file can be created with create_synthetic_dataset.py. 
    # The current values in the config file are those used for the experiments if you would want to run the code with a small synthetic dataset you need to adjust the parameters in the config such thati i.e. batch_size does not exceed the number of IP addresses generated in the synthetic dataset.


    if run_config is not None:
        dataset_path = run_config.dataset_path

        train_ip_split_ratio =run_config.train_ip_split_ratio
        sequence_split_mode =run_config.sequence_split_mode
        sequence_split_ratio =run_config.sequence_split_ratio
        sequence_split_seed =run_config.sequence_split_seed
        max_sequence_length =run_config.max_sequence_length
        batch_size =run_config.batch_size
        num_elements_per_refresh =run_config.num_elements_per_refresh
        train_dataset_step_refresh_intervall =run_config.train_dataset_step_refresh_intervall
        max_refreshes =run_config.max_refreshes
        n_steps =run_config.n_steps
        lr =run_config.lr
        n_attention_heads =run_config.n_attention_heads
        n_encoder_layers =run_config.n_encoder_layers
        d_octet_emb =run_config.d_octet_emb
        d_port_emb =run_config.d_port_emb
        d_protoc_emb =run_config.d_protoc_emb
        d_time_emb =run_config.d_time_emb
        d_model =run_config.d_model
        d_ff =run_config.d_ff
        prob_dropout =run_config.prob_dropout
        sampling_start_alpha =run_config.sampling_start_alpha
        sampling_end_alpha =run_config.sampling_end_alpha
        sampling_schedule =run_config.sampling_schedule
        probability_estimation_window_size =run_config.probability_estimation_window_size
        sequence_overlap =run_config.sequence_overlap

        #experiments_path = "./experiments/" + config_hash(run_config) + "/"
        experiments_path = "./experiments/"

        if not os.path.exists(experiments_path):
            os.makedirs(experiments_path, exist_ok=True)
            save_config(run_config, experiments_path + "run_config.json")

    else:

        dataset_path = "path_to_parquet_dataset"

        train_ip_split_ratio = 0.8

        sequence_split_mode = "temporal"
        sequence_split_ratio = 0.8
        sequence_split_seed = 42

        max_sequence_length = 256
        batch_size = 256
        num_elements_per_refresh = 256
        train_dataset_step_refresh_intervall = 500
        max_refreshes = 5

        n_steps = max_refreshes * train_dataset_step_refresh_intervall

        lr = 1e-3

        n_attention_heads = 4
        n_encoder_layers = 4
        d_octet_emb = 8
        d_port_emb = 16
        d_protoc_emb = 4
        d_time_emb = 16
        d_model = 128
        d_ff = 256
        prob_dropout = 0.1

        sampling_start_alpha = 0.5
        sampling_end_alpha = 0.5
        sampling_schedule = "constant"

        probability_estimation_window_size = 0
        sequence_overlap = 0.0

        experiments_path = "./experiments/"

    ip_stats = read_ip_stats_parquet(dataset_path)

    ip_train, ip_test, total_num_rows = shuffle_and_split_source_ips(ip_stats,train_ratio=train_ip_split_ratio,seed=42)

    ip_train_config = IPSamplingConfig(
        subset_size=num_elements_per_refresh,
        total_num_refreshes=max_refreshes,
        sampling_probabilities_alpha_start=sampling_start_alpha,
        sampling_probabilities_alpha_end=sampling_end_alpha,
        sampling_schedule=sampling_schedule,
    )

    ip_sequence_train_config = DatasetConfig(   dataset_path,
                                                all_ips_in_dataset=ip_train,            
                                                max_sequence_length=max_sequence_length, 
                                                prob_estimation_window_length=probability_estimation_window_size, 
                                                sequence_overlap=sequence_overlap, 
                                                sequence_split_part="train",
                                                sequence_split_mode=sequence_split_mode, 
                                                sequence_split_ratio=sequence_split_ratio, 
                                                sequence_split_seed=sequence_split_seed, 
                                                sampling_config=ip_train_config)

    ip_sequence_train_dataloader_config = DataloaderConfig(batch_size=batch_size,batches_per_refresh=train_dataset_step_refresh_intervall,num_workers=2,collate_fn=contrastive_collate)

    


    ip_unseen_sequence_test_config = DatasetConfig(   dataset_path,
                                                all_ips_in_dataset=ip_train,            
                                                max_sequence_length=max_sequence_length, 
                                                prob_estimation_window_length=probability_estimation_window_size, 
                                                sequence_overlap=sequence_overlap, 
                                                sequence_split_part="test",
                                                sequence_split_mode=sequence_split_mode, 
                                                sequence_split_ratio=sequence_split_ratio, 
                                                sequence_split_seed=sequence_split_seed, 
                                                sampling_config=None)


    ip_unseen_sources_test_config = DatasetConfig(   dataset_path,
                                                all_ips_in_dataset=ip_test,            
                                                max_sequence_length=max_sequence_length, 
                                                prob_estimation_window_length=probability_estimation_window_size, 
                                                sequence_overlap=sequence_overlap, 
                                                sequence_split_part="none",
                                                sequence_split_mode="none", 
                                                sequence_split_ratio=1.0, 
                                                sequence_split_seed=sequence_split_seed, 
                                                sampling_config=None)


    num_classes = 256

    train_contrastive_baseline = False
    evaluate_contrastive_baseline = True



    torch.manual_seed(1)
    contrastive_backbone = TransformerBackbone(n_encoder_layers,n_attention_heads,d_model,d_octet_emb,d_port_emb,d_protoc_emb,d_time_emb,d_ff,prob_dropout,max_sequence_length,num_classes)

    torch.manual_seed(1)
    contrastive_projection_head = ProjectionHead(d_model,d_model)

    torch.manual_seed(1)
    contrastive_baseline_model = IPSeqClassificationTransformer(contrastive_backbone,contrastive_projection_head)
    contrastive_baseline_model = contrastive_baseline_model.to(device)

    if train_contrastive_baseline:
        ip_sequence_train_dataset_factory = make_contrastive_train_dataset_factory(ip_sequence_train_config)
        ip_sequence_train_dataloader_factory = make_contrastive_dataloader_factory(ip_sequence_train_dataloader_config)

        data_manager = DataManager(ip_sequence_train_dataset_factory,ip_sequence_train_dataloader_factory,refresh_interval=train_dataset_step_refresh_intervall)

        train_contrastive_sequence_comparison(data_manager,contrastive_baseline_model,experiments_path,n_steps,lr)

    if evaluate_contrastive_baseline:
        contrastive_baseline_model.load_state_dict(torch.load(experiments_path + 'transformer_contrast_comp_weights.pth'))
        run_evaluation(contrastive_baseline_model,ip_sequence_train_config,ip_unseen_sequence_test_config,ip_unseen_sources_test_config,device,experiments_path,50000)


def run_all_configs(config_dir="configs"):
    config_paths = sorted(Path(config_dir).glob("*.json"))

    for path in config_paths:
        print(f"\n=== Starting {path.name} ===")

        config = load_config(path)
        print(f"Loaded Config: {config}")
        main(config)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    #main()
    run_all_configs()
