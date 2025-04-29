from sentence_transformers import SentenceTransformer, util

# 모델 불러오기
model = SentenceTransformer("jhgan/ko-sroberta-multitask")

def compute_similarity(user_input, news_titles):
    """
    사용자 문장과 뉴스 제목 리스트 간 유사도 계산
    :param user_input: str
    :param news_titles: List[str]
    :return: List of similarity scores (float)
    """
    # 임베딩 생성
    user_emb = model.encode(user_input, convert_to_tensor=True)
    news_embs = model.encode(news_titles, convert_to_tensor=True)

    # 코사인 유사도 계산
    similarities = util.cos_sim(user_emb, news_embs)[0]

    # float로 변환된 유사도 점수만 리스트로 반환
    return [float(score) for score in similarities]


# 테스트
if __name__ == "__main__":
    user_text = "과거 스토킹으로 신고된 인물이 접근금지 조치에도 불구하고 다시 피해자 주거지 주변을 배회한 정황이 확인되었다."
    news_samples = [
        "스토킹 접근금지 명령을 어긴 30대 구속",
        "경제 정책 발표, 주식시장 요동",
        "헤어진 남성이 계속 피해자 집 앞을 배회하다 경찰에 체포"
    ]

    similarities = compute_similarity(user_text, news_samples)

    print("\n사용자 문장:", user_text)
    print("유사한 뉴스 순위:")
    for i, (title, score) in enumerate(zip(news_samples, similarities), 1):
        print(f"{i}. ({score:.4f}) {title}")
