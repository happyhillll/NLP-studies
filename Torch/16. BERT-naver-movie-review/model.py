# data.py

from pathlib import Path
import csv
import random

random.seed(42)

def readcsv(
    path: Path,
    encoding: str = "UTF8",
    delimiter: str = ",",
    ignore_first_row: bool = False,
):
    """
    CSV파일을 읽어 각각의 행의 element들을 list형태로 반환

     for a, b, c in readcsv("PATH"):
         pass

    :param path: csv파일
    :param encoding: 파일의 인코딩(Default: UTF8)
    :param delimiter: 열 구분자(Default: ',')
    :param ignore_first_row: 첫 행 무시할지 여부 - 첫 행이 제목일 경우 True로 설정
    :return: 각 행의 element들
    """
    with open(path, encoding=encoding, newline="") as f:
        reader = iter(csv.reader(f, delimiter=delimiter))
        try:
            first_row = next(reader)
            if not ignore_first_row:
                yield first_row
        except StopIteration:
            return
        except Exception as e:
            print(f"Error in readcsv: {path}")
            raise e
        yield from reader



def get_data(data_type):
    """
    :param data_type:  "train" or "test"
    :return: List of (id, text, rating)
    """
    path = Path(f"nsmc/ratings_{data_type}.txt")
    data = list(readcsv(path, delimiter="\t", ignore_first_row=True))
    random.shuffle(data)
    return data

if __name__ == "__main__":
    get_data()