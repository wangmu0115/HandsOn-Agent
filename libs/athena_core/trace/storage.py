from pathlib import Path

from athena_core.metas import PostInitMeta


class TrajectoryStore(metaclass=PostInitMeta):
    def __init__(self, data_dir: str, **metadata):
        self.data_dir = Path(data_dir)
        self.metadata = metadata

    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def save(self): ...


if __name__ == "__main__":
    print(TrajectoryStore("/tmp", model="qwen3:0.6b", device="mps").metadata)
