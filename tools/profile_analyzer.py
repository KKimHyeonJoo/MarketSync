# tools/profile_analyzer.py

import json
import traceback
import pandas as pd
import math
import streamlit as st
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

import config
from modules.llm_provider import get_llm
# filtering 모듈에서 날짜 예측 함수 가져오기
from modules.filtering import FestivalRecommender

logger = config.get_logger(__name__)

# nan 값 처리기
def replace_nan_with_none(data):
    if isinstance(data, dict):
        return {k: replace_nan_with_none(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_nan_with_none(i) for i in data]
    elif isinstance(data, float) and math.isnan(data):
        return None
    return data

# 축제 데이터 로더
@st.cache_data
def _load_festival_data():
    try:
        file_path = config.PATH_FESTIVAL_DF
        if not file_path.exists():
            logger.error(f"--- [Tool Definition ERROR] '{config.PATH_FESTIVAL_DF}' 파일을 찾을 수 없습니다.")
            return None
        df = pd.read_csv(file_path)
        if '축제명' not in df.columns:
            logger.error("--- [Tool Definition ERROR] '축제명' 컬럼이 df에 없습니다.")
            return None
        df_dict = df.set_index('축제명').to_dict(orient='index')
        logger.info(f"--- [Cache] 축제 원본 CSV 로드 및 딕셔너리 변환 완료 (총 {len(df_dict)}개) ---")
        return df_dict
    except Exception as e:
        logger.critical(f"--- [Tool Definition CRITICAL ERROR] 축제 데이터 로드 실패: {e} ---", exc_info=True)
        return None

# ----------------------------
# Tool 1: 특정 축제 정보 조회
@tool
def get_festival_profile_by_name(festival_name: str) -> str:
    """
    축제 이름을 입력받아, 해당 축제의 상세 프로필(소개, 지역, 키워드, 기간, 고객층 등)을 
    JSON 문자열로 반환합니다. 데이터베이스에서 정확한 이름을 찾아야 합니다.
    (예: "보령머드축제 상세 정보 알려줘")
    """
    logger.info(f"--- [Tool] '특정 축제 정보 조회' 도구 호출 (대상: {festival_name}) ---")
    try:
        festival_db = _load_festival_data()
        if festival_db is None:
            return json.dumps({"error": "축제 데이터베이스를 로드하지 못했습니다."})
        profile_dict = festival_db.get(festival_name)
        if profile_dict:
            profile_dict = replace_nan_with_none(profile_dict)
            profile_dict['축제명'] = festival_name
            return json.dumps(profile_dict, ensure_ascii=False)
        else:
            return json.dumps({"error": f"'{festival_name}' 축제를 찾을 수 없습니다. 철자를 확인해주세요."})
    except Exception as e:
        logger.critical(f"--- [Tool CRITICAL] '특정 축제 정보 조회' 중 오류: {e} ---", exc_info=True)
        return json.dumps({"error": f"'{festival_name}' 축제 검색 중 오류 발생: {e}"})

# ----------------------------
# Tool 2: 가맹점 프로필 분석 (LLM)
@tool
def analyze_merchant_profile(store_profile: str) -> str:
    """
    가맹점(가게)의 프로필 데이터(JSON 문자열)를 입력받아, LLM을 사용하여 
    [강점, 약점, 기회 요인]을 분석하는 컨설팅 리포트를 생성합니다.
    이 도구는 가게의 현재 상태를 진단하고 마케팅 전략을 제안하는 데 사용됩니다.
    """
    logger.info("--- [Tool] '가맹점 프로필 분석' 도구 호출 ---")
    try:
        llm = get_llm(temperature=0.3)
        prompt = f"""
        당신은 최고의 상권 분석 전문가입니다.
        아래 [가게 프로필] 데이터를 바탕으로, 이 가게의 [강점], [약점], [기회 요인]을
        사장님이 이해하기 쉽게 컨설팅 리포트 형식으로 요약해주세요.

        [가게 프로필]
        {store_profile}

        [분석 가이드라인]
        1.  **강점 (Strengths)**: '동일 상권/업종 대비' 높은 수치(매출, 방문객, 객단가 등)나 '재방문율' 등을 찾아 **경쟁 우위**가 되는 핵심 요소 강조하세요.
        2.  **약점 (Weaknesses)**: '동일 상권/업종 대비' 낮은 수치나 '신규 고객 비율' 등을 찾아 **개선이 시급한 영역**을 언급하세요.
        3.  **기회 (Opportunities)**: 가게의 현재 강점과 '주요 고객층'이나 '상권' 특성을 바탕으로, **가게가 활용할 수 있는 마케팅(예: 특정 연령대 타겟, 신규 고객 유치)이 효과적일지 제안하고 이를 달성하기 위한 방향성을 제시하세요.
        4.  **형식**: 마크다운을 사용하여 명확하고 가독성 좋게 작성하세요.
        5.  **전문성/친절함**: 전문적인 분석 용어를 사용하되, 사장님이 쉽게 이해할 수 있도록 친절하고 명확하게 설명하세요.
        6.  **(요청 4) 취소선 금지**: 절대로 `~~text~~`와 같은 취소선 마크다운을 사용하지 마세요.

        [답변 형식]
        ### 🏪 사장님 가게 프로필 분석 리포트

        **1. 강점 (Strengths)**
        * [분석된 강점 1] (분석 근거 명시)
        * [분석된 강점 2] (분석 근거 명시)
        * [필요시 추가 강점]

        **2. 약점 (Weaknesses)**
        * [분석된 약점 1] (개선 필요성 명시)
        * [분석된 약점 2] (개선 필요성 명시)
        * [필요시 추가 약점]

        **3. 기회 (Opportunities)**
        * [분석된 기회 요인 1] (활용 방안 제시)
        * [분석된 기회 요인 2] (활용 방안 제시)
        * [필요시 추가 기회 요인]
        """
        response = llm.invoke([HumanMessage(content=prompt)])
        analysis_report = response.content.strip()
        return analysis_report
    except Exception as e:
        logger.critical(f"--- [Tool CRITICAL] '가맹점 프로필 분석' 중 오류: {e} ---", exc_info=True)
        return f"가게 프로필을 분석하는 중 오류가 발생했습니다: {e}"

# ----------------------------
# Tool 3: 축제 프로필 분석 (LLM)
@tool
def analyze_festival_profile(festival_name: str) -> str:
    """
    축제 이름을 입력받아, 해당 축제의 상세 프로필을 조회하고,
    LLM을 사용하여 [핵심 특징]과 [주요 방문객 특성]을 요약 리포트로 반환합니다.
    (예: "보령머드축제는 어떤 축제야?")
    """
    logger.info(f"--- [Tool] '축제 프로필 분석' 도구 호출 (대상: {festival_name}) ---")
    try:
        # 1. Tool 1 호출
        profile_json = get_festival_profile_by_name.invoke(festival_name)
        
        profile_dict = json.loads(profile_json)

        if "error" in profile_dict:
            return profile_json

        # 2. LLM 요약을 위한 정보 추출
        summary = {
            "축제명": profile_dict.get('축제명'),
            "소개": profile_dict.get('소개'),
            "지역": profile_dict.get('지역'),
            "키워드": profile_dict.get('키워드'),
            "2025_기간": profile_dict.get('2025_기간'),
            "주요_고객층": profile_dict.get('주요고객층', 'N/A'),
            "주요_방문자": profile_dict.get('주요방문자', 'N/A'),
            "축제_인기도": profile_dict.get('축제인기', 'N/A'),
            "인기도_점수": profile_dict.get('인기도_점수', 'N/A'),
            "홈페이지": profile_dict.get('홈페이지')
        }

        # 2026년 날짜 예측 추가
        temp_recommender = FestivalRecommender("", "") 
        predicted_2026_timing = temp_recommender._predict_next_year_date(summary["2025_기간"])

        summary_str = json.dumps(summary, ensure_ascii=False, indent=2)

        llm = get_llm(temperature=0.1)

        # --- 프롬프트 수정 ---
        prompt = f"""
        당신은 축제 전문 분석가입니다. 아래 [축제 프로필 요약]을 바탕으로,
        이 축제의 **핵심 특징**과 **주요 방문객(타겟 고객) 특성**을
        이해하기 쉽게 요약해주세요.

        [축제 프로필 요약]
        {summary_str}

        [분석 가이드라인]
        1.  **핵심 특징**: 입력된 **'소개'** 내용을 바탕으로 축제의 주제와 주요 내용을 **2~3문장으로 상세히 요약**하고, '키워드'와 '축제_인기도', '인기도_점수'를 언급하여 부연 설명합니다. (예: "'{summary.get("소개", "소개 정보 없음")[:50]}...'을(를) 주제로 하는 축제입니다. 주요 키워드는 '{summary.get("키워드", "N/A")}'이며, 인기도는 '{summary.get("축제_인기도", "N/A")}' 수준입니다.")
        2.  **주요 방문객**: '주요_고객층'과 '주요_방문자' 컬럼을 직접 인용하여 설명합니다.
            (예: {summary.get("주요_고객층", "N/A")}이 주로 방문하며, {summary.get("주요_방문자", "N/A")} 비율이 높습니다.)
        3.  **형식**: 아래와 같은 마크다운 형식으로 답변을 작성하세요.
        4.  **취소선 금지**: 절대로 `~~text~~`와 같은 취소선 마크다운을 사용하지 마세요.

        [답변 형식]
        ### 🎈 축제 프로필 분석 리포트: {summary.get("축제명")}

        **1. 축제 핵심 특징**
        * [축제 소개 내용을 바탕으로 2~3문장 요약. 키워드와 인기도 포함]

        **2. 주요 방문객 특성**
        * **주요 고객층:** {summary.get("주요_고객층")}
        * **주요 방문자:** {summary.get("주요_방문자")}

        **3. 2026년 개최 기간 (예상)**
        * {predicted_2026_timing}

        **4. 홈페이지**
        * {summary.get("홈페이지", "정보 없음")}
        """

        response = llm.invoke([HumanMessage(content=prompt)])
        analysis_report = response.content.strip()
        return analysis_report

    except Exception as e:
        logger.critical(f"--- [Tool CRITICAL] '축제 프로필 분석' 중 오류: {e} ---", exc_info=True)
        return f"'{festival_name}' 축제 프로필을 분석하는 중 오류가 발생했습니다: {e}"
