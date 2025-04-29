import os
import json
from keyword_extractor import extract_keywords
from similarity_checker import compute_similarity
from news_scraper import search_news_naver, extract_naver_article
from mistral_inference import generate_warning_message

# 사용자 입력 문장
user_input = "경찰에 스토킹으로 신고한 적 있는 사람이 또 저희 집 주변을 배회하는 걸 봤어요. 분명 접근금지 조치가 있었는데 계속 나타납니다."

# 1. 키워드 추출 (의미 파악용)
keywords = extract_keywords(user_input, top_n=3)
print(f"\n추출된 키워드: {[kw for kw, _ in keywords]}")

# 2. 뉴스 데이터 로딩
base_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(base_dir, "data/news_result.json")

if not os.path.exists(json_path):
    print("\n뉴스 파일이 존재하지 않습니다. 먼저 news_scraper.py를 실행해주세요.")
    exit()

with open(json_path, "r", encoding="utf-8") as f:
    news_items = json.load(f)

if not news_items:
    print("\n뉴스 데이터가 비어 있습니다.")
    exit()

# 3. 제목 + 본문 결합한 문장으로 유사도 비교
titles_and_contents = [
    item["title"] + " " + item["content"]
    for item in news_items
]

similarities = compute_similarity(user_input, titles_and_contents)

# 4. 뉴스 + 유사도 점수 묶기
scored_items = [
    {"title": item["title"], "url": item["url"], "score": score}
    for item, score in zip(news_items, similarities)
]

# 5. 유사도 순 정렬
sorted_items = sorted(scored_items, key=lambda x: x["score"], reverse=True)

# 6. 결과 출력
print("\n사용자 문장과 유사한 뉴스:")
for i, item in enumerate(sorted_items, 1):
    status = "유사함" if item["score"] >= 0.31 else "관련 낮음"
    print(f"{i}. ({item['score']:.4f}) {item['title']} → {status}")
    print(f"   ↪ {item['url']}")


# 7. 유사한 뉴스를 기반으로 LLM 판단 요청
top3_news = sorted_items[:3]
llm_output = generate_warning_message(user_input, top3_news)

print("\n📢 LLM 기반 판단 및 경고 메시지:")
print(llm_output)
