import os
import sys
import json
import pandas as pd
from pathlib import Path
from prefect import task


# TODO: Add support for 'all' flag.
@task
def load_clinc_data(all: bool=False) -> Dict:
    """
    Load OOS data from DATA_DIR into a Dict of pd.DataFrames.

    :param all: Whether to load 'all_wiki_sents.txt' as well.
    :return data: Dictionary containing CLINC data as dataframes.
    """
    data_dir = Path(os.getenv("DATA_DIR"))
    oos_data_dir = data_dir/"oos-eval/data/"
    files = next(os.walk(oos_data_dir))[2]
    data = {}
    print(f"Loading files: {files!r}")

    paths = {f[:f.find('.')]:oos_data_dir/f for f in files}

    print(paths)

    for filen, path in paths.items():
        with open(path, "r") as data_f:
            ext = str(path)[-5:]

            if ext == ".json":
                data_f_json = json.load(data_f)
            else:
                continue
                # TODO Add the bit for all here.

        data[filen] = clinc_json_to_df(data_f_json)

    return data


def clinc_json_to_df(loaded_json):
    """
    Convert a particular CLINC JSON file to a dict of DFs.
    :param loaded_json: JSON object - must be loaded using json.load.
    :return dfs: A dict of DataFrames, containing data in the provided
                    JSON.
    """
    dfs = {
        key:pd.DataFrame(value, columns=['text_cols', 'label_cols'])
        for key, value
        in loaded_json.items()
    }

    return dfs
