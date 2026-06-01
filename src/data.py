import torch
from torch.utils.data import Dataset
import random

import polars as pl
import pyarrow.parquet as pq

import itertools

from torch.nn.utils.rnn import pad_sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

from collections import defaultdict
import pathlib
import time

import numpy as np

import multiprocessing as mp

from sampling_cache import SamplingCache, IPSamplingConfig



class DatasetConfig:
    def __init__(self, 
                 path_to_dataset:str,
                 all_ips_in_dataset,
                 max_sequence_length:int, 
                 prob_estimation_window_length:int, 
                 sequence_overlap:float,
                 sequence_split_part:str, 
                 sequence_split_mode:str, 
                 sequence_split_ratio:float, 
                 sequence_split_seed:int,
                 sampling_config:IPSamplingConfig,
                 anchor_positions=None
                 ):
        self.all_ips_in_dataset = all_ips_in_dataset
        self.path_to_dataset = path_to_dataset
        self.max_sequence_length = max_sequence_length
        self.prob_estimation_window_length = prob_estimation_window_length
        self.sequence_overlap = int(max_sequence_length * sequence_overlap)
        self.sequence_split_part = sequence_split_part
        self.sequence_split_mode = sequence_split_mode
        self.sequence_split_ratio = sequence_split_ratio
        self.sequence_split_seed = sequence_split_seed
        self.sampling_config = sampling_config
        self.anchor_positions = anchor_positions

class DataloaderConfig:
    def __init__(self, batch_size:int, num_workers:int, batches_per_refresh:int=None,collate_fn=None):
        """
        if batches_per_fresh is None or 0 we will refresh every step
        if batches_per_refresh is -1 we will iterate through the whole dataset before refreshing i.e. when the iterater raises StopIteration
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.batches_per_refresh = batches_per_refresh

        self.collate_fn = collate_fn

def ip_dst_to_octets(dataframe):

    dataframe = dataframe.with_columns(dataframe['ip.dst'].str.split(".").alias("octets"))

    dataframe = dataframe.with_columns([
        pl.col("octets").list.get(0).cast(pl.UInt8).alias("ip.dst.o1"),
        pl.col("octets").list.get(1).cast(pl.UInt8).alias("ip.dst.o2"),
        pl.col("octets").list.get(2).cast(pl.UInt8).alias("ip.dst.o3"),
        pl.col("octets").list.get(3).cast(pl.UInt8).alias("ip.dst.o4")
    ])

    dataframe = dataframe.drop("octets")
    dataframe = dataframe.drop("ip.dst")

    return dataframe

def service_to_port_and_protocol(dataframe):

    dataframe = dataframe.with_columns([
        (pl.col('service') // (1 << 16)).cast(pl.UInt16).alias("protocol"),
        (pl.col('service') % (1 << 16)).cast(pl.UInt16).alias("port")
    ])

    dataframe = dataframe.drop("service")

    return dataframe

def ip_to_uint32(ip: str) -> int:
    a, b, c, d = map(int, ip.split('.'))
    return (a << 24) | (b << 16) | (c << 8) | d

def ipv4_int_to_str(ip: int) -> str:
    return ".".join(str((ip >> (8 * i)) & 0xFF) for i in reversed(range(4)))


def ip_string_column_to_uint32(dataframe, column_name):

    ip_struct = (
        pl.col(column_name)
        .str.split_exact(".", 3)
        .struct.rename_fields(["a", "b", "c", "d"])
    )

    dataframe = dataframe.with_columns(
        (
            ip_struct.struct.field("a").cast(pl.UInt32) * 16777216
            + ip_struct.struct.field("b").cast(pl.UInt32) * 65536
            + ip_struct.struct.field("c").cast(pl.UInt32) * 256
            + ip_struct.struct.field("d").cast(pl.UInt32)
        ).alias(column_name)
    )

    return dataframe


def estimate_num_batches_in_csv(csv_path:str,batch_size:int=75_000_000):

    with open(csv_path, 'r') as f:
        lines = list(itertools.islice(f, batch_size))

    avg_line_size = sum(len(line) for line in lines) / len(lines)

    file_size = os.path.getsize(csv_path)

    batch_mem_size =  avg_line_size * batch_size

    print(f"file size: {file_size} bytes, avg line size: {avg_line_size:.2f} bytes, batch mem size: {batch_mem_size:.2f} bytes")

    estimated_batches = int(file_size / batch_mem_size) + 1

    return estimated_batches

def write_csv_as_parquet(csv_path:str,parquet_path:str,num_batches_per_iteration:int=500,batch_size:int=20_000_000):

    reader = pl.read_csv_batched(csv_path, batch_size=batch_size)

    print(f"Writing {csv_path} to {parquet_path}")
    print(f"Batch size: {batch_size}")
    print(f"Reader batches per iteration: {num_batches_per_iteration}")

    counter = {}
    created_folders = set()

    num_read_batches = 0
    num_rows_read = 0
    approx_num_batches = None

    while True:
    #for _ in range(5):

        start = time.time()

        batches = reader.next_batches(num_batches_per_iteration)

        if not batches:
            break

        batch_df = pl.concat(batches, rechunk=False)

        num_rows_read += batch_df.height

        print(f"Time to read and concatenate batches: {time.time() - start:.2f} seconds")

        if approx_num_batches is None:
            approx_num_batches = estimate_num_batches_in_csv(
                csv_path,
                batch_df.height
            )

        print(
            f"Iteration {num_read_batches}/{approx_num_batches} | "
            f"Rows processed: {num_rows_read:,}"
        )

        num_read_batches += 1

        groups = batch_df.partition_by("ip.src", as_dict=False)

        print(f"Time to partition by ip.src: {time.time() - start:.2f} seconds")

        for group_df in groups:

            src_ip = group_df["ip.src"][0]

            src_folder = os.path.join(parquet_path, f"ip.src={src_ip}")

            if src_ip not in created_folders:
                os.makedirs(src_folder, exist_ok=True)
                created_folders.add(src_ip)

            idx = counter.get(src_ip, 0)

            output_path = os.path.join(src_folder, f"part_{idx}.parquet")

            group_df.write_parquet(
                output_path,
                compression="zstd",
                compression_level=3
            )

            counter[src_ip] = idx + 1
        
        print(f"Time to write batches to parquet: {time.time() - start:.2f} seconds")


def write_csv_as_parquet_with_pyarrow(csv_path: str, parquet_path: str,
                         batch_size: int = 20_000_000,
                         batches_per_iteration: int = 5):

    os.makedirs(parquet_path, exist_ok=True)

    reader = pl.read_csv_batched(csv_path, batch_size=batch_size)

    total_rows = 0
    iteration = 0

    print(f"Reading CSV: {csv_path}")
    print(f"Writing dataset: {parquet_path}")
    print(f"Batch size: {batch_size}")

    while True:
    #for _ in range(5):

        start = time.time()

        batches = reader.next_batches(batches_per_iteration)

        if not batches:
            break

        df = pl.concat(batches, rechunk=False)

        total_rows += df.height
        iteration += 1

        print(f"Iteration {iteration} | rows processed: {total_rows:,}")
        print(f"Time to collect batches: {time.time() - start:.2f} seconds")

        # Convert Polars -> Arrow
        table = df.to_arrow()

        print(f"Time to convert to arrow: {time.time() - start:.2f} seconds")

        # Write partitioned dataset
        pq.write_to_dataset(
            table,
            root_path=parquet_path,
            partition_cols=["ip.src"],
            compression="zstd",
            use_dictionary=True,
            max_partitions=20000,
            max_rows_per_file=5000000
        )

        print(f"Time to write to parquet: {time.time() - start:.2f} seconds")

    print("Finished.")


def compact_parquet_dataset(parquet_path:str,shard_size:int=5_000_000):

    root = pathlib.Path(parquet_path)

    dataset_name = root.name

    parent_path = root.parent

    num_folders = sum(1 for item in root.iterdir() if item.is_dir())

    num_processed_folders = 0

    stats = []

    for src_folder in root.iterdir():

        if src_folder.is_dir() and src_folder.name.startswith("ip.src="):
            num_processed_folders += 1
            print(f"Compacting {num_processed_folders} out of {num_folders} folders")

            src_ip = src_folder.name.split("=")[1]

            parquet_files = list(src_folder.glob("*.parquet"))

            dfs = [pl.read_parquet(pf) for pf in parquet_files]

            combined_df = pl.concat(dfs)

            combined_df = combined_df.sort("timestamp")

            combined_df = combined_df.with_columns(pl.arange(0, pl.len()).alias("id"))

            #check if dataframe has column service and if so split it into port and protocol and drop service column
            if "service" in combined_df.columns:
                combined_df = service_to_port_and_protocol(combined_df)

            #check if ip.src and ip.dst are of type string and if so convert them to uint32
            if combined_df["ip.src"].dtype == pl.Utf8:
                combined_df = ip_string_column_to_uint32(combined_df,"ip.src")


            if combined_df["ip.dst"].dtype == pl.Utf8:
                combined_df = ip_string_column_to_uint32(combined_df,"ip.dst")

            #compute_conditional_octet_probabilities(combined_df)

            combined_df = combined_df.with_columns([
                pl.col("zmap").cast(pl.UInt8),
                pl.col("mirai").cast(pl.UInt8),
                pl.col("ttl").cast(pl.UInt8)
            ])


            num_rows = combined_df.height
            num_shards = 0


            for shard_idx, start in enumerate(range(0, num_rows, shard_size)):
                shard = combined_df.slice(start, shard_size)

                shard.write_parquet(
                    src_folder / f"shard_{shard_idx:05d}.parquet",
                    compression="zstd",
                    compression_level=3,
                    row_group_size=100_000,
                )
                num_shards += 1

            stats.append((combined_df["ip.src"][0],num_rows,num_shards,shard_size,combined_df["zmap"][0],combined_df["mirai"][0],combined_df["acknowledged_scanner"][0]))

            for pf in src_folder.glob("*.parquet"):
                if not pf.name.startswith("shard_"):
                    pf.unlink()


    stats_file_name = dataset_name + "_ip_stats.parquet"


    stats_frame = pl.DataFrame(stats, schema=["ip.src", "num_rows","num_shards","shard_size","zmap","mirai","acknowledged_scanner"],orient="row")
    stats_frame.write_parquet(root / stats_file_name, compression="zstd", compression_level=3)


def read_ip_stats_parquet(root_path:str):

    root = pathlib.Path(root_path)

    dataset_name = root.name
    stats_file_name = dataset_name + "_ip_stats.parquet"

    df = pl.read_parquet(root / stats_file_name)

    return df

def shuffle_and_split_source_ips(df,train_ratio:float=0.8,seed:int=42) :

    df = df.sample(fraction=1.0,shuffle=True,seed=42)

    df = df.with_columns(pl.col("num_rows").cum_sum().alias("cumulative_rows"))

    total_rows = df["num_rows"].sum()

    split_threshold = total_rows * train_ratio


    train_df = df.filter(pl.col("cumulative_rows") <= split_threshold)
    test_df = df.filter(pl.col("cumulative_rows") > split_threshold)

    return train_df, test_df, total_rows

def calculate_sampling_probabilities(df,alpha_start:float=0.5,alpha_end:float=0.5,alpha_schedule:str="constant",refresh_idx:int=0, total_num_refreshes:int=100):

    valid_schedules = ["constant"]

    if alpha_schedule not in valid_schedules:
        print(f"Invalid alpha schedule: {alpha_schedule}. Valid schedules are: {valid_schedules}. Defaulting to constant.")
        alpha_schedule = "constant"


    if alpha_schedule == "constant":
        alpha = alpha_start

    weights = df["num_rows"].to_numpy() ** alpha
    probs = weights / weights.sum()

    return probs

def ip_str_to_zeropadded_i64(ip_str: str) -> int:
    return int("".join(f"{int(p):03d}" for p in ip_str.split(".")))


def ip_str_to_int_array(ip_str: str) -> list[int]:
    return [int(ip_octet) for ip_octet in ip_str.split(".")]

def split_service_into_port_and_protocol(service: int) -> tuple[int,int]:
    port = service & 0xFFFF

    ip_prtcl = service >> 16

    return port, ip_prtcl

def ip_to_octet_tensor(ip_str:str) -> torch.Tensor:
    ip_as_list = [int(octet) for octet in ip_str.split('.')]

    return torch.tensor(ip_as_list,dtype=torch.int64)

def uint32_to_octets(ip_tensor: torch.Tensor):

    x = ip_tensor.to(torch.int64)

    return torch.stack([
        (x >> 24) & 0xFF,
        (x >> 16) & 0xFF,
        (x >> 8)  & 0xFF,
        x & 0xFF
    ], dim=-1).to(torch.uint8)

def list_source_ips_in_parquet(base_path:str):
    found_source_ips = []

    for folder in os.listdir(base_path):
        if folder.startswith("ip.src="):
            found_source_ips.append(folder.split("=")[1])
    
    return found_source_ips

def split_rows_in_train_and_test(config:DatasetConfig,dataframe):

    if config.sequence_split_mode != "none":

        num_rows = config.all_ips_in_dataset.filter(pl.col('ip.src') == dataframe["ip.src"][0])["num_rows"][0]
        dataframe = dataframe.with_columns(pl.lit(num_rows).alias("num_entries_per_src"))

        if config.sequence_split_mode == "temporal":

            dataframe = dataframe.with_columns(
                (pl.col("num_entries_per_src") * config.sequence_split_ratio)
                .floor()
                .cast(pl.Int64)
                .alias("split_index")
            )

            if config.sequence_split_part == "train":
                dataframe = dataframe.filter(
                    pl.col("id") < pl.col("split_index")
                )
            elif config.sequence_split_part == "test":
                dataframe = dataframe.filter(
                    pl.col("id") >= pl.col("split_index")
                )
            else:
                raise ValueError(f"Invalid sequence split part: {config.sequence_split_part}")
            
            
        else:
            raise ValueError(f"Invalid sequence split mode: {config.sequence_split_mode}")
        
    return dataframe

def split_into_sequences(config,dataframe):

    start_expr = (pl.col('local_id') - (config.max_sequence_length + config.prob_estimation_window_length)) // (config.max_sequence_length - config.sequence_overlap) + 1
    end_expr = (pl.col('local_id')) // (config.max_sequence_length - config.sequence_overlap ) + 1 

    safe_start = pl.when(start_expr < 0).then(0).otherwise(start_expr)

    dataframe = dataframe.with_columns((pl.arange(0,pl.len()).over('ip.src')).alias('local_id'))

    dataframe = dataframe.with_columns(
            (pl.int_ranges(safe_start, end_expr))
            .alias("chunk_ids")
    )

    dataframe = dataframe.explode("chunk_ids")

    return dataframe

def load_single_parquet_file(args):


    config,src_ip,shard_idx = args

    src_ip_string = ipv4_int_to_str(src_ip) if not isinstance(src_ip, str) else src_ip

    if shard_idx != -1:
        parquet_file_path = os.path.join(config.path_to_dataset, f"ip.src={src_ip_string}", f"shard_{shard_idx:05d}.parquet")
    else:
        parquet_file_path = os.path.join(config.path_to_dataset, f"ip.src={src_ip_string}", "*.parquet")

    df = pl.read_parquet(parquet_file_path)

    df = df.sort("timestamp")

    df = split_rows_in_train_and_test(config,df)

    cols_to_chunk = ['ip.dst','port','protocol','timestamp','acknowledged_scanner','zmap','mirai']

    data = {}

    if config.anchor_positions is not None and src_ip in config.anchor_positions:

        df = df.with_columns((pl.arange(0,pl.len()).over('ip.src')).alias('local_id'))

        half_seq_len = config.max_sequence_length // 2

        sequences = []

        if src_ip not in config.anchor_positions:
            print(f"Warning: Source IP {src_ip} does not have anchor positions defined. Skipping.")

        if len(config.anchor_positions[src_ip]) == 0:
            print(f"Warning: Source IP {src_ip} has an empty list of anchor positions. Skipping.")

        for anchor_pos in config.anchor_positions[src_ip]:

            start_pos = max(0, anchor_pos - half_seq_len)
            end_pos = min(anchor_pos + half_seq_len + config.prob_estimation_window_length, df.height)

            seq_df = df.filter((pl.col("local_id") >= start_pos) & (pl.col("local_id") < end_pos))

            sequences.append(seq_df)

        for col in cols_to_chunk:
            data[col] = np.concatenate([
                seq[col].to_numpy() for seq in sequences
            ])

        lengths = [len(seq) for seq in sequences]

        data["offsets"] = np.concatenate([[0], np.cumsum(lengths)])

        data["ip.src"] = np.full(
            (len(sequences),),
            src_ip,
            dtype=np.uint32
        )
            

    else:
        df = split_into_sequences(config,df)

        grouped = (
            df.group_by("chunk_ids")
            .agg(*[pl.col(c) for c in cols_to_chunk])
            .filter(pl.col('ip.dst').list.len() >= config.prob_estimation_window_length + 1)
        )



        data["offsets"] = grouped["ip.dst"].to_arrow().offsets.to_numpy()

        for col in cols_to_chunk:
            arr = grouped[col].to_arrow()

            data[f"{col}"] = arr.values.to_numpy()

        # chunk ids can be added if needed 
        #data["chunk_id"] = torch.from_numpy(grouped["chunk_ids"].to_numpy())

        data["ip.src"] = np.full(
            (grouped.height,),
            src_ip,
            dtype=np.uint32
        )

    return data

def concat_offsets(offset_list):
    result = []
    current_offset = 0

    for i, offsets in enumerate(offset_list):
        if len(offsets) == 0:
            continue

        offsets = offsets.astype(np.int64)  # safety

        shifted = offsets + current_offset

        if result:
            shifted = shifted[1:]

        if len(shifted) > 0:
            result.append(shifted)

        # update running total using last offset
        current_offset += offsets[-1]

    # guard against all inputs being empty
    if not result:
        return torch.empty(0, dtype=torch.int64)

    return torch.from_numpy(np.concatenate(result))

class IPAddressAsSequenceDataset:


    def __init__(self,
                 source_ips,
                 path_to_dataset:str,
                 max_sequence_length:int,
                 prob_estimation_window_length:int,
                 sequence_overlap:float,
                 sequence_split_mode:str,
                 sequence_split_part:str,
                 sequence_split_ratio:float,
                 sequence_split_seed:int,
                 config:DatasetConfig):
        super(IPAddressAsSequenceDataset,self).__init__()

        self.path_to_dataset = path_to_dataset

        self.max_sequence_length = max_sequence_length

        self.sequence_overlap = int(max_sequence_length * sequence_overlap)

        self.prob_estimation_window_length = prob_estimation_window_length

        self.sequence_split_mode = sequence_split_mode
        self.sequence_split_part = sequence_split_part
        self.sequence_split_ratio = sequence_split_ratio
        self.sequence_split_seed = sequence_split_seed

        self.source_ips = source_ips

        self.config = config

        self.parallel_materialize_dataset_for_grouped_parquet_files(self.source_ips)


    def set_overlap_and_window(self,overlap:float,window_length:int):

        self.sequence_overlap = overlap

        self.prob_estimation_window_length = window_length

        self.dataset = None

    def split_rows_in_train_and_test(self,dataframe):

        if self.sequence_split_mode != "none":

            dataframe = dataframe.with_columns(pl.len().over("ip.src").alias("num_entries_per_src"))

            if self.sequence_split_mode == "temporal":

                dataframe = dataframe.with_columns(
                    (pl.col("num_entries_per_src") * self.sequence_split_ratio)
                    .floor()
                    .cast(pl.Int64)
                    .alias("split_index")
                )

                if self.sequence_split_part == "train":
                    dataframe = dataframe.filter(
                        pl.col("id") < pl.col("split_index")
                    )
                elif self.sequence_split_part == "test":
                    dataframe = dataframe.filter(
                        pl.col("id") >= pl.col("split_index")
                    )
                else:
                    raise ValueError(f"Invalid sequence split part: {self.sequence_split_part}")
                
                
            else:
                raise ValueError(f"Invalid sequence split mode: {self.sequence_split_mode}")
            
        return dataframe
    
    def split_into_sequences(self,dataframe):

        start_expr = (pl.col('local_id') - (self.max_sequence_length + self.prob_estimation_window_length)) // (self.max_sequence_length - self.sequence_overlap) + 1
        end_expr = (pl.col('local_id')) // (self.max_sequence_length - self.sequence_overlap ) + 1 

        safe_start = pl.when(start_expr < 0).then(0).otherwise(start_expr)

        dataframe = dataframe.with_columns((pl.arange(0,pl.len()).over('ip.src')).alias('local_id'))

        dataframe = dataframe.with_columns(
                (pl.int_ranges(safe_start, end_expr))
                .alias("chunk_ids")
        )

        dataframe = dataframe.explode("chunk_ids")

        return dataframe
    
    def parallel_materialize_dataset_for_grouped_parquet_files(self,scr_ip_df:pl.DataFrame):

        ips = scr_ip_df['ip.src'].to_numpy()
        shard_indices = scr_ip_df['shard_idx'].to_numpy()

        if self.config.anchor_positions is not None:
            args = [(self.config,scr_ip,shard_idx) for scr_ip,shard_idx in zip(ips,shard_indices) if scr_ip in self.config.anchor_positions and len(self.config.anchor_positions[scr_ip]) > 0]
        else:
            args = [(self.config,scr_ip,shard_idx) for scr_ip,shard_idx in zip(ips,shard_indices)]

        with mp.Pool(processes=1) as pool:
            results = pool.map(load_single_parquet_file,args)
       
        self.dataset = {}
        keys_to_concat = ['ip.src','ip.dst','port','protocol','timestamp','acknowledged_scanner','zmap','mirai']

        for key in keys_to_concat:
            tensors = [torch.from_numpy(d[key]) for d in results]

            if tensors:
                self.dataset[key] = torch.cat([torch.from_numpy(d[key]) for d in results], dim=0)
            else:
                print("WARNING: No data found for key:", key)
                self.dataset[key] = torch.tensor([], dtype=torch.int64)

        self.dataset['offsets'] = concat_offsets(d['offsets'] for d in results)

    def __len__(self):
        return len(self.dataset['ip.src'])
    
    def __getitem__(self,idx):

        start = self.dataset['offsets'][idx].item()
        end = self.dataset['offsets'][idx+1].item()

        sample = {
                "ip.src": self.dataset['ip.src'][idx],

                "ip.dst": self.dataset['ip.dst'][start:end],
                "port":self.dataset['port'][start:end],
                "protocol":self.dataset['protocol'][start:end],
                "timestamp":self.dataset['timestamp'][start:end],
                "acknowledged_scanner":self.dataset['acknowledged_scanner'][start:end],
                "zmap":self.dataset['zmap'][start:end],
                "mirai":self.dataset['mirai'][start:end],
            }

        return sample


def create_padding_mask_from_reference_sequence(sequence : torch.Tensor):
    # the sequence needs to be such that 0 is the element which was added by the padding
    return sequence == 0

def collate_ip_batch(batch):

    out = {}

    for key in batch[0].keys():
        # Each element is a list of tensors for this key
        values = [sample[key] for sample in batch]

        # If sequence (dim > 1), pad; otherwise just stack
        if values[0].ndim >= 1:
            out[key] = pad_sequence(values, batch_first=True, padding_value=0)
        else:
            out[key] = torch.stack(values)

    mask_list = [create_padding_mask_from_reference_sequence(protocol_seq) for protocol_seq in out['protocol']]

    out['masks'] = torch.stack(mask_list)

    out['ip.dst'] = uint32_to_octets(out['ip.dst'])
    
    return out

class ContrastiveIPSequenceDataset:

    def __init__(self,base_sequence_datasets:IPAddressAsSequenceDataset):
        super(ContrastiveIPSequenceDataset,self).__init__()

        self.base_dataset = base_sequence_datasets

        begin = time.time()
        self.build_source_to_indices_lookup()

    def build_source_to_indices_lookup(self):

        ip_src = self.base_dataset.dataset["ip.src"].to(torch.int64)

        sorted_vals, sorted_idx = torch.sort(ip_src)

        unique_vals, counts = torch.unique_consecutive(
            sorted_vals, return_counts=True
        )

        splits = torch.split(sorted_idx, counts.tolist())

        self.source_to_indices = {
            int(src.item()): indices
            for src, indices in zip(unique_vals, splits)
            if indices.numel() > 1
        }

    def __len__(self):
        return sum(len(indices) for indices in self.source_to_indices.values())

    def __getitem__(self,index_pair):

        i,j = index_pair
        return self.base_dataset[i], self.base_dataset[j]


class ContrastiveBatchSampler(torch.utils.data.Sampler):
    def __init__(self, sources_to_indices,batch_size,batches_per_refresh):
        self.target_batch_size = batch_size
        self.batches_per_refresh = batches_per_refresh

        self.set_source_to_indices(sources_to_indices)

        #print(self.batch_size)

    def set_source_to_indices(self,sources_to_indices):
        self.sources = [src for src in sources_to_indices.keys()]
        self.sources_to_indices = sources_to_indices
        self.batch_size = min(self.target_batch_size,len(self.sources))

    def __iter__(self):
        for _ in range(self.batches_per_refresh):
            batch_sources = random.sample(self.sources,self.batch_size)

            batch_indices = []

            for src in batch_sources:
                
                indices = self.sources_to_indices[src]

                perm = torch.randperm(indices.size(0))
                i, j = indices[perm[:2]]

                batch_indices.append((i.item(), j.item()))

            yield batch_indices
    
    def __len__(self):
        return self.batches_per_refresh

def contrastive_collate(batch):
    # batch = [(x1_dict, x2_dict), ...]
    x1_list, x2_list = zip(*batch)

    batch_x1 = collate_ip_batch(x1_list)
    batch_x2 = collate_ip_batch(x2_list)

    return batch_x1, batch_x2

def sample_ips_for_training_step(all_ips,sampling_config:IPSamplingConfig,refresh_idx:int):

    ip_train_sample_probs = calculate_sampling_probabilities(all_ips,
                                                            alpha_start=sampling_config.sampling_probabilities_alpha_start,
                                                            alpha_end=sampling_config.sampling_probabilities_alpha_end,
                                                            alpha_schedule=sampling_config.sampling_schedule,
                                                            refresh_idx=refresh_idx,
                                                            total_num_refreshes=sampling_config.total_num_refreshes)
    


    ips = all_ips['ip.src'].to_numpy()

    sampled_indices = np.random.choice(len(ips), size=sampling_config.subset_size, replace=False, p=ip_train_sample_probs)

    sampled_df = all_ips[sampled_indices]

    sampled_shards = np.array([
        np.random.randint(n_shards)
        for n_shards in sampled_df["num_shards"].to_numpy()
    ])

    sampled_df = sampled_df.with_columns(
        pl.Series("shard_idx", sampled_shards)
    )

    return sampled_df


def make_train_dataset_factory(config:DatasetConfig):
    state = {"refresh_idx": 0}

    base_dir_of_all_caches = os.path.join(config.path_to_dataset, "sampling_caches")

    sampling_cache = SamplingCache(base_dir_of_all_caches,config=config.sampling_config)

    def factory():

        if sampling_cache is not None:

            ip_train = sampling_cache.get_or_compute_refresh(state["refresh_idx"],
                                                             compute_fn=lambda r=state["refresh_idx"]: sample_ips_for_training_step(
                                                                    config.all_ips_in_dataset,
                                                                    config.sampling_config,
                                                                    r
                                                                ))
        else:
            ip_train = config.all_ips_in_dataset

            #add shard idx column with -1 to ip_train to indicate that we want to use all shards for the sampled ips
            ip_train = ip_train.with_columns(pl.lit(-1).alias("shard_idx"))

        raw_dataset = IPAddressAsSequenceDataset(
            source_ips=ip_train,
            path_to_dataset=config.path_to_dataset,
            max_sequence_length=config.max_sequence_length,
            prob_estimation_window_length=config.prob_estimation_window_length,
            sequence_overlap=config.sequence_overlap,
            sequence_split_mode=config.sequence_split_mode,
            sequence_split_part=config.sequence_split_part,
            sequence_split_ratio=config.sequence_split_ratio,
            sequence_split_seed=config.sequence_split_seed,
            config=config
        )

        state["refresh_idx"] += 1

        return raw_dataset

    return factory

def make_dataloader_factory(config:DataloaderConfig):
    
    def factory(dataset):

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            collate_fn=config.collate_fn
        )

        return dataloader
    
    return factory

def make_contrastive_train_dataset_factory(config:DatasetConfig):#
    state = {"refresh_idx": 0}

    base_dir_of_all_caches = os.path.join(config.path_to_dataset, "sampling_caches")

    sampling_cache = SamplingCache(base_dir_of_all_caches,config=config.sampling_config)

    def factory():

        if sampling_cache is not None:

            ip_train = sampling_cache.get_or_compute_refresh(state["refresh_idx"],
                                                             compute_fn=lambda r=state["refresh_idx"]: sample_ips_for_training_step(
                                                                    config.all_ips_in_dataset,
                                                                    config.sampling_config,
                                                                    r
                                                                ))
        else:
            ip_train = config.all_ips_in_dataset

            ip_train = ip_train.with_columns(pl.lit(-1).alias("shard_idx"))

        raw_dataset = IPAddressAsSequenceDataset(
            source_ips=ip_train,
            path_to_dataset=config.path_to_dataset,
            max_sequence_length=config.max_sequence_length,
            prob_estimation_window_length=config.prob_estimation_window_length,
            sequence_overlap=config.sequence_overlap,
            sequence_split_mode=config.sequence_split_mode,
            sequence_split_part=config.sequence_split_part,
            sequence_split_ratio=config.sequence_split_ratio,
            sequence_split_seed=config.sequence_split_seed,
            config=config
        )

        state["refresh_idx"] += 1
        contrastive_dataset = ContrastiveIPSequenceDataset(raw_dataset)

        return contrastive_dataset

    return factory

def make_contrastive_dataloader_factory(config:DataloaderConfig):
    
    def factory(dataset):

        contrastive_batch_sampler = ContrastiveBatchSampler(
            sources_to_indices=dataset.source_to_indices,
            batch_size=config.batch_size,
            batches_per_refresh=config.batches_per_refresh
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=contrastive_batch_sampler,
            num_workers=config.num_workers,
            collate_fn=config.collate_fn
        )

        return dataloader
    
    return factory

class DataManager:
    def __init__(
        self,
        dataset_factory,
        dataloader_factory,
        refresh_interval=None,
    ):
        self.dataset_factory = dataset_factory
        self.dataloader_factory = dataloader_factory
        self.refresh_interval = refresh_interval

        self.step = 0
        self.loader = None
        self.loader_iter = None

        self._build()

    def _build(self):
        dataset = self.dataset_factory()
        self.loader = self.dataloader_factory(dataset)
        self.loader_iter = iter(self.loader)

    def _maybe_refresh(self,force=False):

        if force:
            #print("Force refreshing dataset and dataloader")
            self._build()
            return

        if (self.refresh_interval is None or self.refresh_interval == 0) and self.step > 0:
            #print("Refreshing at every step since refresh_interval is None or 0")
            self._build()
        else:
            if self.refresh_interval > 0 and self.step > 0 and self.step % self.refresh_interval == 0:
                #print("Need to refresh!")
                self._build()
            

    def __iter__(self):
        return self

    def __next__(self):
        self._maybe_refresh()

        try:
            batch = next(self.loader_iter)
        except StopIteration:

            if self.refresh_interval == -1:
                self._maybe_refresh(force=True)
            else:
                self.loader_iter = iter(self.loader)
                
            batch = next(self.loader_iter)

        self.step += 1
        return batch


#if __name__ == "__main__":
#    mp.set_start_method("spawn", force=True)
#
#    csv_file_path = "./../code/telescope_data/dataprocessing_verification/test_dataset.csv"
#    dataset_path = "./../code/telescope_data/dataprocessing_verification/test_dataset.parquet"
#
#    #write_csv_as_parquet(csv_file_path,dataset_path)
#    compact_parquet_dataset(dataset_path,10)
