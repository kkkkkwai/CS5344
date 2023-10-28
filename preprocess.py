from pathlib import Path
from translate import Translator
from dask.distributed import Client
import dask.dataframe as dd
import json
import logging

TRENDING_VIDEO_DATA = "data/youtube-trending-video-dataset"
FOREIGN_LANGUAGES = ["BR", "DE","FR", "IN", "JP", "KR", "MX", "RU"]


def translate_and_preprocess(data_path: Path, category_file: Path, translate=False):
    print(f"start processing {data_path.name}, translate {translate}")
    if(translate):
        sample_and_translate(data_path)
    preprocess(data_path, category_file)
    if(translate):
        data_path.unlink()
        Path(data_path.as_posix() + ".tmp").rename(data_path)
        
    
def sample_and_translate(data_path: Path):
    temp_path = data_path.rename(data_path.as_posix() + ".tmp")
    ddf = dd.read_csv(temp_path.as_posix())

    if(Path("output/title.json").exists()):
        with open("output/title.json") as f:
            title = json.load(f)
            sampled_video_id = [k for k, v in title.items()]
    else:
        video_id_frequency = ddf["video_id"].value_counts(sort=False)
        sampled_video_id = video_id_frequency[video_id_frequency > 5].sample(frac=0.018).index
        # sampling, must yield results
        sampled_video_id.dask.visualize(filename=f"output/{data_path.stem}_cond_sampling.svg")
        sampled_video_id = sampled_video_id.compute()
    ddf = ddf[ddf["video_id"].isin(sampled_video_id)]
    # to dict, must yield results
    #‘index’ : dict like {index -> {column -> value}}
    records = ddf.groupby(["video_id"]).first().compute().to_dict("index")
    
    title_dict, tags_dict, description_dict = translate_all(records)

    ddf = ddf.drop(columns=["title", "tags", "description"])
    ddf["title"] = ddf["video_id"].map(title_dict)
    ddf["tags"] = ddf["video_id"].map(tags_dict)
    ddf["description"] = ddf["video_id"].map(description_dict)

    ddf.dask.visualize(filename=f"output/{data_path.stem}_translation.svg")
    ddf.to_csv(data_path, single_file=True, index=False)



def preprocess(data_path: Path, category_file: Path):
    with category_file.open() as f:
        cat_items = json.load(f)["items"]
    cat_dic = {int(c["id"]): c["snippet"]["title"] for c in cat_items}

    ddf = dd.read_csv(data_path.as_posix())
    ddf["category_name"] = ddf["categoryId"].map(cat_dic)
    ddf["description"] = ddf["description"].fillna("")

    ddf.dask.visualize(filename=f"output/{data_path.stem}_no_translation.svg")
    processed_file_name =  f"{data_path.stem}_processed.csv"
    ddf.to_csv(data_path.parent.joinpath(processed_file_name).as_posix(), single_file=True, index=False)


def skip(data_path: Path):
    processed_file_name =  f"{data_path.stem}_processed.csv"
    return data_path.parent.joinpath(processed_file_name).exists()

def translate_all(records):
    translator = Translator()
    print("translate!")

    if Path("output/title.json").exists():
        title_dict = json.load(open("output/title.json"))
    else:
        title_dict = {vid: translator.translate(r["title"]) for vid, r in records.items()}
        json.dump(title_dict, open("output/title.json", "w"))

    if Path("output/tags.json").exists():
        tags_dict = json.load(open("output/tags.json"))
    else: 
        tags_dict = {}
        for vid, r in records.items():
            if r["tags"] and r["tags"].lower() != "[none]":
                tags_dict[vid] = translator.translate(r["tags"])
            else:
                tags_dict[vid] = r["tags"]
        json.dump(tags_dict, open("output/tags.json", "w"))

    if Path("output/description.json").exists():
        description_dict = json.load(open("output/description.json"))
    else:
        print("description!")
        description_dict = {}
        for vid, r in records.items():
            if r["description"]:
                description_dict[vid] = translator.translate(r["description"])
        json.dump(description_dict, open("output/description.json", "w"))

    return title_dict, tags_dict, description_dict

    
if __name__ == "__main__":
    #start local cluster
    client = Client(n_workers=4)

    # Preprocess YouTube Trending Video Dataset (updated daily)    
    file_list = Path(TRENDING_VIDEO_DATA).glob("*trending_data.csv")

    for data_path in file_list:
        if(skip(data_path)):
            continue

        country_code = data_path.name[:2]
        category_path = data_path.parent.joinpath(f"{country_code}_category_id.json")
        if not category_path.exists():
            logging.warning(f"{category_path.name} not exist! skip {data_path.name}")
            continue
        
        translate_and_preprocess(data_path, category_path, country_code in FOREIGN_LANGUAGES)
