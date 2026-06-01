import polars as pl
import numpy as np
import json
import hashlib
import os

class AnchorLabelFilter:
    def __init__(self,
                 filter_name:str,
                 column_name:str,
                 operation:str,
                 value:list[int],
                 ratio:float):
        self.filer_name = filter_name
        self.column_name = column_name
        self.operation = operation
        self.value = value
        self.ratio = ratio

    def to_dict(self):
        return {
            "filter_name": self.filer_name,
            "column_name": self.column_name,
            "operation": self.operation,
            "value": self.value,
            "ratio": self.ratio
        }
    
    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            filter_name=d["filter_name"],
            column_name=d["column_name"],
            operation=d["operation"],
            value=d["value"],
            ratio=d["ratio"]
        )

class AnchorCreationConfig:
    def __init__(self,
                    dataset_name:str,
                    subset_type:str,
                    max_sequence_length:int,
                    num_anchors_to_create:int,
                    sequence_split_part:str,
                    sequence_split_ratio:float,
                    anchor_sampling_alpha:float,
                    label_columns:list[AnchorLabelFilter] = None,):
        self.dataset_name = dataset_name
        self.subset_type = subset_type
        self.max_sequence_length = max_sequence_length
        self.num_anchors_to_create = num_anchors_to_create
        self.sequence_split_part = sequence_split_part
        self.sequence_split_ratio = sequence_split_ratio
        self.anchor_sampling_alpha = anchor_sampling_alpha
        self.label_columns = label_columns

    def to_dict(self):
        return {
            "version": 1,
            "dataset_name": str(self.dataset_name),
            "subset_type": str(self.subset_type),
            "max_sequence_length": int(self.max_sequence_length),
            "num_anchors_to_create": int(self.num_anchors_to_create),
            "sequence_split_part": str(self.sequence_split_part),
            "sequence_split_ratio": round(float(self.sequence_split_ratio), 10),
            "anchor_sampling_alpha": round(float(self.anchor_sampling_alpha), 10),
            "label_columns": [label.to_dict() for label in self.label_columns] if self.label_columns is not None else None,
        }

    def get_config_hash(self) -> str:
        canonical = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":")  # removes whitespace differences
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    
    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            dataset_name=d["dataset_name"],
            subset_type=d["subset_type"],
            max_sequence_length=d["max_sequence_length"],
            num_anchors_to_create=d["num_anchors_to_create"],
            sequence_split_part=d["sequence_split_part"],
            sequence_split_ratio=d["sequence_split_ratio"],
            anchor_sampling_alpha=d["anchor_sampling_alpha"],
            label_columns=[AnchorLabelFilter.from_dict(filter_data) for filter_data in d.get("label_columns")]
            )

def filter_any_label(df: pl.DataFrame, label_col_filter: AnchorLabelFilter) -> pl.DataFrame:

    if not label_col_filter:
        return df
    
    expr = None

    if label_col_filter.operation == "==":
        expr = pl.col(label_col_filter.column_name).is_in(label_col_filter.value)
    elif label_col_filter.operation == "!=":
        expr = ~pl.col(label_col_filter.column_name).is_in(label_col_filter.value)
    elif label_col_filter.operation is None:
        return df
    else:
        raise ValueError(f"Unsupported operation: {label_col_filter.operation}")

    return df.filter(expr)

def create_anchor_position_file(ips_df,config: AnchorCreationConfig,dataset_base_path: str) -> None:

    max_sequence_length = config.max_sequence_length
    num_anchors_to_create = config.num_anchors_to_create
    sequence_split_ratio = config.sequence_split_ratio
    anchor_sampling_alpha = config.anchor_sampling_alpha
    label_columns = config.label_columns

    if config.sequence_split_part == "train":
        split_ratio = sequence_split_ratio
    elif config.sequence_split_part == "test":
        split_ratio = 1.0 - sequence_split_ratio
    else:
        raise ValueError(f"Invalid sequence_split_part: {config.sequence_split_part}. Must be 'train' or 'test'.")

    all_anchors = []
    total_num_generated_anchors = 0

    for label_col_filter in label_columns or []:

        print(f"Processing filter: {label_col_filter.filer_name} with ratio {label_col_filter.ratio}")

        np.random.seed(42)

        filtered_ips_df = filter_any_label(ips_df,label_col_filter) if label_col_filter is not None else ips_df

        half_window = max_sequence_length // 2

        filtered_ips_df = filtered_ips_df.with_columns(
            (pl.col("num_rows") * split_ratio).cast(pl.Int64).alias("num_rows_in_split")
            ).with_columns(
            (pl.col("num_rows_in_split") - max_sequence_length + 1).clip(lower_bound=1).alias("num_valid_anchor_positions")
            ).with_columns([
            (
                pl.when(pl.col("num_valid_anchor_positions") < max_sequence_length)
                .then(pl.col("num_valid_anchor_positions") / max_sequence_length)
                .otherwise(
                    ((pl.col("num_valid_anchor_positions")) // max_sequence_length)
                )
                .alias("num_anchors")
            )
            ]).with_columns(
            (pl.col("num_anchors") ** anchor_sampling_alpha).alias("score"))

        filtered_ips_df = filtered_ips_df.with_columns(
            (pl.col("score") / pl.col("score").sum()).alias("prob")
        )

        filtered_ips_df = filtered_ips_df.with_columns(pl.col("num_anchors").ceil().alias("num_anchors"))

        print("Possible anchors:",filtered_ips_df.select(pl.col("num_anchors").sum()))

        anchors_to_create_for_filter = int(num_anchors_to_create * label_col_filter.ratio) if label_col_filter is not None else num_anchors_to_create

        if anchors_to_create_for_filter > filtered_ips_df.select(pl.col("num_anchors").sum()).item():
            print(f"Warning: Requested {anchors_to_create_for_filter} anchors but only {filtered_ips_df.select(pl.col('num_anchors').sum()).item()} are available based on the current configuration. Adjusting to maximum available.")
            anchors_to_create_for_filter = filtered_ips_df.select(pl.col("num_anchors").sum()).item()



        indices = np.arange(len(filtered_ips_df))

        sources = filtered_ips_df["ip.src"].to_numpy()
        probs = filtered_ips_df["prob"].to_numpy()
        probs = probs / probs.sum()


        anchors_per_source = {src : [] for src in sources}

        num_generated_anchors = 0

        while num_generated_anchors < anchors_to_create_for_filter:
            
            sampled_index = np.random.choice(indices, p=probs)

            src = sources[sampled_index]

            sampled_row = filtered_ips_df.row(sampled_index,named=True)

            #check if we can still add anchors for this source without exceeding the max sequence length constraint
            if len(anchors_per_source[src]) < sampled_row["num_anchors"]:

                num_rows_for_ip = sampled_row["num_rows_in_split"]

                retry_count = 0

                while retry_count < 10:
                    if num_rows_for_ip <= max_sequence_length:
                        anchor_position = num_rows_for_ip // 2
                    else:
                        anchor_position = np.random.randint(half_window,num_rows_for_ip - half_window)

                    found_valid_anchor_pos = True

                    for existing_anchor in anchors_per_source[src]:
                        if abs(existing_anchor - anchor_position) < max_sequence_length:
                            found_valid_anchor_pos = False
                            break

                    if found_valid_anchor_pos:

                        anchors_per_source[src].append(anchor_position)
                        num_generated_anchors += 1

                        break

                    retry_count += 1

        total_num_generated_anchors += num_generated_anchors

        for src, anchor_positions in anchors_per_source.items():

            if len(anchor_positions) > 0:

                all_anchors.append({
                    "ip.src": src,
                    "anchors": anchor_positions,
                    "filter_name": label_col_filter.filer_name if label_col_filter is not None else "none",
                    "num_anchors": len(anchor_positions)
                })
            


    print(f"Generated {total_num_generated_anchors} anchors.")
    #print(anchors_per_index)

    anchor_df = pl.DataFrame(all_anchors)

    #print(anchor_df.head(20))

    anchor_paths = f"{dataset_base_path}/anchors/{config.get_config_hash()}"

    os.makedirs(anchor_paths, exist_ok=True)

    output_path = f"{anchor_paths}/anchors_{config.get_config_hash()}.parquet"
    anchor_df.write_parquet(output_path)


    metadata_path = f"{anchor_paths}/metadata.json"
    metadata = {
        "config": config.to_dict(),
        "hash": config.get_config_hash(),
    }

    tmp_path = metadata_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(metadata, f, indent=2)

    os.replace(tmp_path, metadata_path)

    return anchor_df

def validate_anchor_file(ips_df, anchor_df):
    ips_in_anchors = set(anchor_df["ip.src"].to_list())
    ips_in_data = set(ips_df["ip.src"].to_list())

    if not ips_in_anchors.issubset(ips_in_data):
        missing_ips = ips_in_anchors - ips_in_data
        raise ValueError(f"Anchor file contains IPs not present in the dataset: {missing_ips}")

def get_or_create_anchor_file(ips_df,config: AnchorCreationConfig, dataset_base_path: str) -> str:
    anchors_base_paths = f"{dataset_base_path}/anchors"
    anchor_paths = f"{dataset_base_path}/anchors/{config.get_config_hash()}"
    anchor_file = f"{anchor_paths}/anchors_{config.get_config_hash()}.parquet"
    metadata_file = f"{anchor_paths}/metadata.json"

    found_existing_anchor_file = False

    if not os.path.exists(anchors_base_paths):
        os.makedirs(anchors_base_paths, exist_ok=True)
        print("Created anchor folder and file since it did not exist.")
        return create_anchor_position_file(ips_df,config,dataset_base_path)

    if os.path.exists(anchor_file) and os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
            if metadata.get("hash") == config.get_config_hash():
                anchor_positions_raw_df = pl.read_parquet(anchor_file)

                validate_anchor_file(ips_df, anchor_positions_raw_df)

                found_existing_anchor_file = True

                return anchor_positions_raw_df
            else:
                print("Config hash mismatch in metadata. Searching other for other anchor files.")
    else:
        print("Anchor file or metadata not found. Generating anchors.")

    for candidate in os.listdir(anchors_base_paths):
        candidate_dir = os.path.join(anchors_base_paths, candidate)
        candidate_meta_path = os.path.join(candidate_dir, "metadata.json")

        if not os.path.exists(candidate_meta_path):
            continue

        try:
            with open(candidate_meta_path, "r") as f:
                metadata = json.load(f)

            stored_config = AnchorCreationConfig.from_dict(metadata["config"])

            if stored_config == config:

                print("Found existing anchor file with matching config. Reusing anchors.")

                anchor_file_path = os.path.join(candidate_dir, f"anchors_{metadata['hash']}.parquet")
                anchor_positions_raw_df = pl.read_parquet(anchor_file_path)



                validate_anchor_file(ips_df, anchor_positions_raw_df)

                found_existing_anchor_file = True

                return anchor_positions_raw_df

        except Exception:
            # Skip corrupted or incompatible metadata
            continue

    if not found_existing_anchor_file:
        print("No compatible anchor file found. Creating new anchor file.")
        return create_anchor_position_file(ips_df,config,dataset_base_path)