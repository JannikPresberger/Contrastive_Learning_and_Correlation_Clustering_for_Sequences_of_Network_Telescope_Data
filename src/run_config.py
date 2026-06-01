from dataclasses import dataclass,asdict
import json
import hashlib
from pathlib import Path

@dataclass
class RunConfig:
    
    dataset_path:str
    train_ip_split_ratio:float# = 0.8
    sequence_split_mode:str# = "temporal"
    sequence_split_ratio:int# = 0.8
    sequence_split_seed:int# = 42
    max_sequence_length:int# = 256
    batch_size:int# = 256
    num_elements_per_refresh:int# = 256
    train_dataset_step_refresh_intervall:int# = 500
    max_refreshes:int# = 5
    #n_steps:int# = self.max_refreshes * self.train_dataset_step_refresh_intervall
    lr:float# = 1e-3
    n_attention_heads:int# = 4
    n_encoder_layers:int# = 4
    d_octet_emb:int# = 8
    d_port_emb:int# = 16
    d_protoc_emb:int# = 4
    d_time_emb:int# = 16
    d_model:int# = 128
    d_ff:int# = 256
    prob_dropout:float# = 0.1
    sampling_start_alpha:float# = 0.5
    sampling_end_alpha:float# = 0.5
    sampling_schedule:str# = "constant"
    probability_estimation_window_size:int# = 0
    sequence_overlap:float# = 0.0

    @property
    def n_steps(self):
        return self.max_refreshes * self.train_dataset_step_refresh_intervall

def config_to_dict(config: RunConfig) -> dict:
    return asdict(config)


def config_to_json(config: RunConfig) -> str:
    return json.dumps(
        config_to_dict(config),
        indent=2,
        sort_keys=True
    )


def config_hash(config: RunConfig, length: int = 12) -> str:
    json_str = json.dumps(
        config_to_dict(config),
        sort_keys=True,
        separators=(",", ":")  # canonical form
    )

    h = hashlib.sha256(json_str.encode("utf-8")).hexdigest()
    return h[:length]


def save_config(config: RunConfig, path: str | Path):
    path = Path(path)
    path.write_text(config_to_json(config))


def load_config(path: str | Path) -> RunConfig:
    path = Path(path)

    data = json.loads(path.read_text())

    return RunConfig(**data)