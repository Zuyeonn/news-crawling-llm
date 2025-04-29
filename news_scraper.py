import os
import requests
from bs4 import BeautifulSoup
import urllib.parse
import json

# 1. 네이버 뉴스 링크 수집 
def search_news_naver(query, max_articles=5):
    headers = {'User-Agent': 'Mozilla/5.0'}
    encoded_query = urllib.parse.quote(query)
    url = f"https://search.naver.com/search.naver?where=news&query={encoded_query}"

    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, 'html.parser')

    links = []
    for a in soup.select("a.info"):
        href = a.get("href")
        if "news.naver.com" in href and href not in links:
            links.append(href)
        if len(links) >= max_articles:
            break

    print(f"{len(links)}개 네이버 뉴스 링크 수집 완료\n")
    return links

# 2. 네이버 뉴스 제목 + 본문 추출
def extract_naver_article(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')

        content = soup.select_one("#dic_area")
        title = soup.select_one("h2#title_area span") or soup.select_one("h3#articleTitle")

        if not content or not title:
            print(f"본문 또는 제목 추출 실패 → {url}")
            return None

        article_text = content.get_text(strip=True)
        title_text = title.get_text(strip=True)

        return {
            "title": title_text,
            "url": url,
            "content": article_text
        }

    except Exception as e:
        print(f"크롤링 실패: {url}\n   이유: {e}")
        return None

# 3. 전체 수집 → JSON 저장
def collect_and_save_news(query, max_articles=5, save_path="news_crawling/data/news_result.json"):
    print(f"\n뉴스 수집 키워드: '{query}'")
    news_urls = search_news_naver(query, max_articles=max_articles)
    articles = []

    for url in news_urls:
        result = extract_naver_article(url)
        if result:
            articles.append(result)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    print(f"\n{len(articles)}개 뉴스 저장 완료 → {save_path}")

# 4. 실행 예시
if __name__ == "__main__":
    query = "스토킹 배회 주거지 주변"
    collect_and_save_news(query, max_articles=5)
