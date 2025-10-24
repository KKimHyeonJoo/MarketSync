import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import time

# --- 설정 ---
# ⚠️ 원본 파일 경로와 저장될 파일 이름을 확인하세요.
INPUT_CSV_PATH = 'festival_df.csv'
OUTPUT_CSV_PATH = 'festival_df_updated.csv'
# ----------------

def generate_keywords_from_description(llm, description: str) -> str:
    """
    축제 소개글을 바탕으로 Gemini AI를 사용하여 키워드를 생성합니다.
    """
    if not isinstance(description, str) or not description.strip():
        return ""

    # AI에게 역할을 부여하고, 원하는 결과물의 형식과 내용을 구체적으로 지시하는 프롬프트
    prompt = f"""
    당신은 지역 축제 전문 마케팅 분석가입니다.
    아래 제공된 축제 소개글을 읽고, 부스 참가를 고려하는 가게 사장님에게 도움이 될 만한 핵심 키워드를 추출해주세요.

    [추출 가이드라인]
    1. 다음 5가지 카테고리로 키워드를 분류해주세요:
       - **타겟 고객**: (예: 20대, 가족 단위, 친구, 연인, 외국인 관광객)
       - **계절**: (예: 봄, 여름, 가을, 겨울)
       - **축제 분위기**: (예: 활기찬, 전통적인, 힙한, 자연 친화적)
       - **주요 콘텐츠**: (예: 먹거리, 푸드트럭, 체험 활동, 공연, 전통문화, 불꽃놀이, 특산물)
       - **핵심 테마**: (예: 역사, 문화, 음악, 예술, 계절)
    2. 모든 키워드를 쉼표(,)로 구분된 하나의 문자열로 만들어 반환해주세요.
       (예시: 가족 단위, 연인, 활기찬, 전통적인, 먹거리, 체험, 역사, 문화)
    3. 소개글에서 근거를 찾을 수 없는 내용은 추측하여 만들지 마세요.

    [축제 소개글]
    {description}

    [추출된 키워드 (쉼표로 구분)]
    """
    
    try:
        message = HumanMessage(content=prompt)
        response = llm.invoke([message])
        return response.content.strip()
    except Exception as e:
        print(f"  [오류] API 호출 중 문제 발생: {e}")
        return ""

def main():
    """
    메인 실행 함수
    """
    print("--- 🤖 '소개' 기반 AI 키워드 자동 생성 작업을 시작합니다. ---")

    # 1. Google API 키 및 LLM 초기화
    try:
        # 'GOOGLE_API_KEY' 라는 이름의 환경 변수를 찾습니다.
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다. API 키를 설정해주세요.")
        
        # 정확한 판단을 위해 temperature를 낮게 설정
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            # 환경 변수에서 불러온 키를 사용합니다.
            google_api_key=google_api_key, 
            temperature=0.1
        )
        print("✅ Gemini 모델 초기화 완료.")
    except Exception as e:
        print(f"❌ [치명적 오류] Gemini 모델 초기화 실패: {e}")
        return

    # 2. CSV 파일 로드
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
        print(f"✅ '{INPUT_CSV_PATH}' 파일 로딩 완료. (총 {len(df)}개 축제)")
    except FileNotFoundError:
        print(f"❌ [치명적 오류] 파일을 찾을 수 없습니다: '{INPUT_CSV_PATH}'")
        print("    프로젝트 폴더 내에 'festival_df.csv' 파일이 있는지 확인해주세요.")
        return

    # 3. 각 축제별로 키워드 생성 및 추가
    new_keywords_list = []
    total_rows = len(df)

    for index, row in df.iterrows():
        print(f"\n--- ({index + 1}/{total_rows}) '{row['축제명']}' 작업 중 ---")
        
        description = row['소개']
        
        print("  - AI를 호출하여 키워드를 생성합니다...")
        new_keywords = generate_keywords_from_description(llm, description)
        
        original_keywords = str(row.get('키워드', ''))
        
        all_keywords = original_keywords.split(',') + new_keywords.split(',')
        unique_keywords = sorted(list(set([k.strip() for k in all_keywords if k.strip()])))
        
        final_keywords_str = ', '.join(unique_keywords)
        new_keywords_list.append(final_keywords_str)
        
        print(f"  - [기존 키워드]: {original_keywords if original_keywords else '없음'}")
        print(f"  - [AI 생성 키워드]: {new_keywords}")
        print(f"  - [최종 키워드]: {final_keywords_str}")

        time.sleep(0.5) 

    # 4. DataFrame에 새로운 키워드 열 추가 및 저장
    df['키워드'] = new_keywords_list
    
    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
    print(f"\n--- 🎉 작업 완료! ---")
    print(f"✅ 새로운 키워드가 추가된 파일이 '{OUTPUT_CSV_PATH}' 경로에 저장되었습니다.")

if __name__ == "__main__":
    main()
