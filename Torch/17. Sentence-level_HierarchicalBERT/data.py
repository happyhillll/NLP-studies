# data.py

from pathlib import Path
from typing import Iterable, Any, Dict, Union, List, Callable, Tuple
import json

def readjson(path: Union[str, Path], *, encoding: str = "UTF8"):
    """
    JSON파일을 읽어온다

    :param path: json파일
    :param encoding: 파일의 인코딩(Default: UTF8)
    :return: Dict(json)
    """
    with open(path, encoding=encoding) as f:
        j = json.load(f)
    return j

def get_data():
    """
    :param data_type:  "train" or "test"
    :return: List of (id, text, rating)
    """
    path = Path(f"../../../Corpus/메신저 말뭉치_v2.0/NIKL_MESSENGER_v2.0/국립국어원 메신저 말뭉치(버전 2.0)")
    data = []

    for p in list(path.rglob("*.json"))[:1000]:
        j = readjson(p)
        j = j["document"][0]
        data.append(
            {
                "id": j["id"],
                "label": j["metadata"]["topic"],
                "text": " [SEP] ".join([u["form"] for u in j["utterance"]])
            }
        )

    return data


if __name__ == "__main__":
    get_data()