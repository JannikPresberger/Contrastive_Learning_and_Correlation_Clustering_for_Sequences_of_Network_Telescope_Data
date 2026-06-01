import os
import json
import polars as pl
import json
import hashlib

from datetime import datetime

class IPSamplingConfig:
    def __init__(self,
                subset_size:int,
                total_num_refreshes:int,
                sampling_probabilities_alpha_start:float,
                sampling_probabilities_alpha_end:float,
                sampling_schedule:str):
        self.subset_size = subset_size
        self.total_num_refreshes = total_num_refreshes
        self.sampling_probabilities_alpha_start = sampling_probabilities_alpha_start
        self.sampling_probabilities_alpha_end = sampling_probabilities_alpha_end
        self.sampling_schedule = sampling_schedule

    def to_dict(self):
        return {
            "version": 2,
            "subset_size": int(self.subset_size),
            "total_num_refreshes": int(self.total_num_refreshes),
            "sampling_probabilities_alpha_start": round(float(self.sampling_probabilities_alpha_start), 10),
            "sampling_probabilities_alpha_end": round(float(self.sampling_probabilities_alpha_end), 10),
            "sampling_schedule": str(self.sampling_schedule),
        }
    
    def get_config_hash(self) -> str:
        canonical = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":")  # removes whitespace differences
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    
    def __eq__(self, other):
        return isinstance(other, IPSamplingConfig) and self.to_dict() == other.to_dict()
    
    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            subset_size=d["subset_size"],
            total_num_refreshes=d["total_num_refreshes"],
            sampling_probabilities_alpha_start=d["sampling_probabilities_alpha_start"],
            sampling_probabilities_alpha_end=d["sampling_probabilities_alpha_end"],
            sampling_schedule=d["sampling_schedule"],
        )

def get_or_create_cache(base_dir: str, config: IPSamplingConfig) -> str:
    """
    Returns a cache directory matching the config.
    Creates a new one if no compatible cache exists.
    """

    os.makedirs(base_dir, exist_ok=True)

    run_id = config.get_config_hash()
    cache_dir = os.path.join(base_dir, run_id)
    metadata_path = os.path.join(cache_dir, "metadata.json")

    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        stored_config = IPSamplingConfig.from_dict(metadata["config"])

        if stored_config != config:
            raise ValueError(
                f"Hash collision or config mismatch for run_id={run_id}"
            )

        return cache_dir


    for candidate in os.listdir(base_dir):
        candidate_dir = os.path.join(base_dir, candidate)
        candidate_meta_path = os.path.join(candidate_dir, "metadata.json")

        if not os.path.exists(candidate_meta_path):
            continue

        try:
            with open(candidate_meta_path, "r") as f:
                metadata = json.load(f)

            stored_config = IPSamplingConfig.from_dict(metadata["config"])

            if stored_config == config:
                return candidate_dir  # reuse existing cache

        except Exception:
            # Skip corrupted or incompatible metadata
            continue


    os.makedirs(cache_dir, exist_ok=True)

    metadata = {
        "config": config.to_dict(),
        "run_id": run_id,
        "created_at": datetime.utcnow().isoformat(),
    }

    # Atomic write (important for crash safety)
    tmp_path = metadata_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(metadata, f, indent=2)

    os.replace(tmp_path, metadata_path)

    return cache_dir

class SamplingCache:
    def __init__(self, base_dir: str, config: IPSamplingConfig):
        self.base_dir = base_dir
        self.config = config
        self.cache_dir = get_or_create_cache(base_dir, config)
        self.metadata_path = os.path.join(self.cache_dir, "metadata.json")

    def _refresh_dir(self, refresh: int) -> str:
        return os.path.join(self.cache_dir, f"refresh={refresh}")

    def _refresh_file(self, refresh: int) -> str:
        return os.path.join(self._refresh_dir(refresh), "data.parquet")

    def has_refresh(self, refresh: int) -> bool:
        return os.path.exists(self._refresh_file(refresh))

    def save_refresh(self, refresh: int, sampled_ips_df: pl.DataFrame):
        refresh_dir = self._refresh_dir(refresh)

        if os.path.exists(refresh_dir):
            raise ValueError(f"Refresh {refresh} already exists in cache")

        os.makedirs(refresh_dir, exist_ok=True)


        file_path = self._refresh_file(refresh)

        # Atomic write
        tmp_path = file_path + ".tmp"
        sampled_ips_df.select(["ip.src","shard_idx"]).write_parquet(tmp_path)
        os.replace(tmp_path, file_path)

    def load_refresh(self, refresh: int):

        file_path = self._refresh_file(refresh)

        if not os.path.exists(file_path):
            return None

        return pl.read_parquet(file_path)

    def get_or_compute_refresh(self, refresh: int, compute_fn):


        if self.has_refresh(refresh):
            return self.load_refresh(refresh)

        sampled_ips_df = compute_fn()
        self.save_refresh(refresh, sampled_ips_df)
        return sampled_ips_df

    def list_refreshes(self):
        if not os.path.exists(self.cache_dir):
            return []

        refreshes = []
        for name in os.listdir(self.cache_dir):
            if name.startswith("refresh="):
                try:
                    refreshes.append(int(name.split("=")[1]))
                except ValueError:
                    continue

        return sorted(refreshes)

    def get_metadata(self):
        with open(self.metadata_path, "r") as f:
            return json.load(f)

    def __repr__(self):
        return f"SamplingCache(dir={self.cache_dir}, config={self.config})"