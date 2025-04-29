from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = "hajeong67/mistral-7b-merged"

# 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
)

# 텍스트 생성 파이프라인 구성 
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    do_sample=True,           
    temperature=0.7,         
    pad_token_id=model.config.eos_token_id  
)

# 경고 메시지 생성 함수
def generate_warning_message(user_input: str, related_news: list) -> str:
    news_str = "\n".join(
        [f"{i+1}. {news['title']} (출처: {news['url']})" for i, news in enumerate(related_news)]
    )

    prompt = f"""
당신은 대한민국 경찰의 상황 판단을 도와주는 AI입니다.

다음 사용자의 신고 내용을 기반으로 다음 3가지를 출력하세요:

1. [판단 요약] ← 사용자 상황에 대한 분석
2. [관련 사례] ← 아래 뉴스 제목+출처를 반드시 포함
3. [경고 및 조치] ← 경찰 입장에서 취해야 할 조치

---

신고 내용:
"{user_input}"

최근 유사 사례 뉴스 (아래 내용을 반드시 포함하세요):
{news_str}

---

※ 아래 형식을 반드시 지키고, 누락 없이 출력하세요.

[판단 요약]
- 

[관련 사례]
1. 뉴스 제목 (출처: URL)
2. ...
3. ...

[경고 및 조치]
- 

"""



    response = generator(
        prompt,
        max_new_tokens=1000,
        return_full_text=False
    )[0]["generated_text"]

    return response
