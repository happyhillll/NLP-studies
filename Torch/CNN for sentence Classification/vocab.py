from constant import START_TOKEN, END_TOKEN

def get_vocab():
    doc = '''
    [서울=뉴시스] 양소리 기자 = 윤석열 대통령은 28일(현지시간) 미국 보스턴의 하버드대학교에서 연설을 끝으로 공식 일정을 마무리했다.
    지난 27일 저녁 보스턴에 도착한 윤 대통령은 28일 하루동안 매사추세츠공과대학(MIT) 석학과의 간담회, 한미 클러스터 라운드 테이블, 메사추세츠 주지사와의 오찬, 하버드대 연설 등 일정을 소화했다.
    윤 대통령은 29일 오전 귀국길에 오른다. 나는 밥이 좋아 나는 니가 실허 싫어
    '''
    words = [START_TOKEN, END_TOKEN] + list(set(doc.split()))
    vocab = {
        word: idx for idx, word in enumerate(words)
    }
    # id_to_word = {
    #     v: k for k, v in word_to_id.items()
    # }

    return vocab

if __name__ == "__main__":
    get_vocab()