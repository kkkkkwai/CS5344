from dask.distributed import Client
from pathlib import Path
from preprocess import TRENDING_VIDEO_DATA
import dask.dataframe as dd

def split(file_path: Path, dest_dir: Path):
    prev = file_path.rename(file_path.parent.joinpath(file_path.name + ".old"))
    df = dd.read_csv(prev.as_posix(), lineterminator='\n')
    sampled_channel_ids = df["channelId"].unique().sample(frac=0.1).to_frame()

    merged = df.merge(sampled_channel_ids, on="channelId", how="left", indicator=True)
    df_train = df[merged["_merge"] == "left_only"]
    df_dev = df[merged["_merge"] == "both"]

    # df_train.dask.visualize(filename=f"output/train_split.svg")
    # df_dev.dask.visualize(filename=f"output/dev_split.svg")

    df_train.to_csv(file_path, single_file=True, index=False)
    df_dev.to_csv(dest_dir.joinpath(file_path.stem + "_test.csv"), single_file=True, index=False)

if __name__ == "__main__":
    test_path = Path(TRENDING_VIDEO_DATA).parent.joinpath("test_dataset")
    if not test_path.is_dir():
        test_path.mkdir()

    client = Client(n_workers=4)
    file_list = Path(TRENDING_VIDEO_DATA).glob("*processed.csv")
    for f in file_list:
        if f.parent.joinpath(f.name + ".old").exists():
            continue
        print(f.name)
        split(f, test_path)