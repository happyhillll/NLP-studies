import requests
import zipfile
import os
import datetime
from tqdm import tqdm
import pandas as pd


def get_token(username, password):
    url = "http://braincrew2.iptime.org:8001/api/token-auth/"

    data = dict()
    data["username"] = username
    data["password"] = password

    # POST 요청을 보내어 인증 토큰 받기
    response = requests.post(url, data=data)

    # 응답 출력
    if response.status_code == 200:
        return response.json()["token"]
    else:
        print("오류 메시지:", response.json()["error"])
        return None


def submit(competition_url, username, password, dataframe):
    if not os.path.exists("submissions"):
        os.makedirs("submissions")

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_path = os.path.join("submissions", f"{timestamp}-submission.csv")
    dataframe.to_csv(file_path, index=False)

    # API 엔드포인트
    url = (
        f"http://braincrew2.iptime.org:8001/api/competitions/submit/{competition_url}/"
    )

    token = get_token(username, password)

    print("아이디: ", username)
    print("파일명: ", file_path)

    # 제출할 파일
    files = {"file": open(file_path, "rb")}  # 실제 파일 경로로 대체하세요

    # 요청 헤더에 인증 토큰 포함
    headers = {"Authorization": f"Token {token}"}

    # 파일과 함께 POST 요청 보내기
    response = requests.post(url, files=files, headers=headers)

    # 응답 출력
    if response.status_code == 200:
        print("===" * 20)
        print("[제출에 성공하였습니다]")
        print("제출 결과:", response.json()["score"])
    else:
        print("오류 메시지:", response.json())


def download_files(url, filename="data.zip"):
    # .zip 파일을 스트림 방식으로 다운로드하며 진행율 표시
    with requests.get(url, stream=True) as response:
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
        with open(filename, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

    # 'data' 폴더 생성 (이미 존재하지 않는 경우)
    if not os.path.exists("data"):
        os.makedirs("data")

    # 압축 해제 위치를 'data' 폴더로 지정
    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall("./data")

    # 다운로드 받은 .zip 파일 삭제
    os.remove(filename)


def download_competition_files(
    url="MNIST",
    use_competition_url=True,
):
    if use_competition_url:
        target_url = f"http://braincrew2.iptime.org:8001/competitions/{url}/files/"
    else:
        target_url = url
    download_files(target_url)
