from keybert import KeyBERT

# 한국어용 의미 임베딩 모델 (BERT 기반)
model = KeyBERT('jhgan/ko-sroberta-multitask')

def extract_keywords(text, top_n=5):
    """
    주어진 텍스트에서 의미 기반으로 중요한 키워드를 추출합니다.
    """
    keywords = model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 1),  # 단어 단위만 추출 (1~2면 구 단위까지 가능)
        stop_words=None,
        top_n=top_n
    )
    return keywords


# 테스트 실행 코드
if __name__ == "__main__":
    test_text = "과거 스토킹으로 신고된 인물이 접근금지 조치에도 불구하고 다시 피해자 주거지 주변을 배회한 정황이 확인되었다."
    result = extract_keywords(test_text)

    print(f"\n입력 문장: {test_text}")
    print("추출된 키워드:")
    for keyword, score in result:
        print(f" - {keyword} (유사도: {score:.4f})")