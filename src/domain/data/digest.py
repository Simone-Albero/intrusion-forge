import hashlib

import pandas as pd


def digest_frames(frames: dict[str, pd.DataFrame], columns: list[str]) -> str:
    """Digest of `columns` in each frame, row by row, framed by frame name and size."""
    digest = hashlib.blake2b(digest_size=16)
    for name, df in frames.items():
        rows = pd.util.hash_pandas_object(df[columns], index=False)
        digest.update(f"{name}:{len(df)}".encode())
        digest.update(rows.to_numpy().tobytes())
    return digest.hexdigest()


def digest_regions(train_df: pd.DataFrame, noise_cluster_ids: list[int]) -> str:
    """Digest of the regions prepare drew on the train split, its noise ids included."""
    noise = pd.DataFrame({"cluster": sorted(noise_cluster_ids)}, dtype="int64")
    return digest_frames({"train": train_df, "noise": noise}, ["cluster"])
