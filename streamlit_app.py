# streamlit_app.py (FastAPI 통합 버전)

import streamlit as st
import os
import pandas as pd
import numpy as np  # api/server.py에서 필요
import math       # api/server.py에서 필요
import json
import traceback
# import requests # 더 이상 API 호출에 필요하지 않음
from PIL import Image
from pathlib import Path

from langchain_core.messages import HumanMessage, AIMessage

import config 
from orchestrator import AgentOrchestrator
from modules.visualization import display_merchant_profile
from modules.knowledge_base import load_marketing_vectorstore, load_festival_vectorstore

logger = config.get_logger(__name__)

# --- 페이지 설정 ---
st.set_page_config(
    page_title="MarketSync(마켓싱크)",
    page_icon="🎉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- (1) api/data_loader.py에서 가져온 함수 ---
# config.py를 직접 임포트하므로 sys.path 조작 필요 없음
def load_and_preprocess_data():
    """
    미리 가공된 final_df.csv 파일을 안전하게 찾아 로드하고,
    데이터를 처리하는 과정에서 발생할 수 있는 모든 오류를 방어합니다.
    (api/data_loader.py의 원본 함수)
    """
    try:
        file_path = config.PATH_FINAL_DF

        if not file_path.exists():
            logger.critical(f"--- [CRITICAL DATA ERROR] 데이터 파일을 찾을 수 없습니다. 예상 경로: {file_path}")
            logger.critical(f"--- 현재 작업 경로: {Path.cwd()} ---")
            return None
            
        df = pd.read_csv(file_path)

    except Exception as e:
        logger.critical(f"--- [CRITICAL DATA ERROR] 데이터 파일 로딩 중 예측하지 못한 오류 발생: {e} ---", exc_info=True)
        return None
        
    logger.info("--- [Preprocess] Streamlit Arrow 변환 오류 방지용 데이터 클리닝 시작 ---")
    for col in df.select_dtypes(include='object').columns:
        temp_series = (
            df[col]
            .astype(str)
            .str.replace('%', '', regex=False)
            .str.replace(',', '', regex=False)
            .str.strip()
        )
        numeric_series = pd.to_numeric(temp_series, errors='coerce') 
        df[col] = numeric_series.fillna(temp_series)
        
    logger.info("--- [Preprocess] 데이터 클리닝 완료 ---")

    cols_to_process = ['월매출금액_구간', '월매출건수_구간', '월유니크고객수_구간', '월객단가_구간']
    
    for col in cols_to_process:
        if col in df.columns:
            try:
                series_str = df[col].astype(str).fillna('')
                series_split = series_str.str.split('_').str[0]
                series_numeric = pd.to_numeric(series_split, errors='coerce')
                df[col] = series_numeric.fillna(0).astype(int)
            except Exception as e:
                logger.warning(f"--- [DATA WARNING] '{col}' 컬럼 처리 중 오류 발생: {e}. 해당 컬럼을 건너뜁니다. ---", exc_info=True)
                continue
                
    logger.info(f"--- [Preprocess] 데이터 로드 및 전처리 최종 완료. (Shape: {df.shape}) ---")
    return df

# --- (2) api/server.py에서 가져온 헬퍼 함수 ---
def replace_nan_with_none(data):
    """
    딕셔셔너리나 리스트 내의 모든 NaN 값을 None으로 재귀적으로 변환합니다.
    (api/server.py의 원본 함수)
    """
    if isinstance(data, dict):
        return {k: replace_nan_with_none(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_nan_with_none(i) for i in data]
    elif isinstance(data, float) and math.isnan(data):
        return None
    return data

# --- (3) api/server.py의 POST /profile 로직을 변환한 함수 ---
def get_merchant_profile_logic(merchant_id: str, df_merchant: pd.DataFrame):
    """
    가맹점 ID와 마스터 데이터프레임을 받아 프로파일링된 데이터를 반환합니다.
    (api/server.py의 POST /profile 엔드포인트 로직)
    """
    logger.info(f"✅ [Local Logic] 가맹점 ID '{merchant_id}' 프로파일링 요청 수신")
    try:
        store_df_multiple = df_merchant[df_merchant['가맹점ID'] == merchant_id]

        if store_df_multiple.empty:
            logger.warning(f"⚠️ [Local Logic] 404 - '{merchant_id}' 가맹점 ID를 찾을 수 없습니다.")
            raise ValueError(f"'{merchant_id}' 가맹점 ID를 찾을 수 없습니다.")
        
        if len(store_df_multiple) > 1:
            logger.info(f"   [INFO] '{merchant_id}'에 대해 {len(store_df_multiple)}개의 데이터 발견. 최신 데이터로 필터링합니다.")
            temp_df = store_df_multiple.copy()
            temp_df['기준년월_dt'] = pd.to_datetime(temp_df['기준년월'])
            latest_store_df = temp_df.sort_values(by='기준년월_dt', ascending=False).iloc[[0]]
        else:
            latest_store_df = store_df_multiple

        store_data = latest_store_df.iloc[0].to_dict()

        # (고객 비율 및 자동추출특징 계산 로직은 원본과 동일)
        # 4-1. 고객 성별 비율 계산 및 저장
        store_data['남성고객비율'] = (
            store_data.get('남성20대이하비율', 0) + store_data.get('남성30대비율', 0) + 
            store_data.get('남성40대비율', 0) + store_data.get('남성50대비율', 0) + 
            store_data.get('남성60대이상비율', 0)
        )
        store_data['여성고객비율'] = (
            store_data.get('여성20대이하비율', 0) + store_data.get('여성30대비율', 0) + 
            store_data.get('여성40대비율', 0) + store_data.get('여성50대비율', 0) + 
            store_data.get('여성60대이상비율', 0)
        )
        
        # 4-2. 연령대별 비율 계산 (20대이하, 30대, 40대, 50대이상)
        store_data['연령대20대이하고객비율'] = store_data.get('남성20대이하비율', 0) + store_data.get('여성20대이하비율', 0)
        store_data['연령대30대고객비율'] = store_data.get('남성30대비율', 0) + store_data.get('여성30대비율', 0)
        store_data['연령대40대고객비율'] = store_data.get('남성40대비율', 0) + store_data.get('여성40대비율', 0)
        store_data['연령대50대고객비율'] = (
            store_data.get('남성50대비율', 0) + store_data.get('여성50대비율', 0) + 
            store_data.get('남성60대이상비율', 0) + store_data.get('여성60대이상비율', 0)
        )

        male_ratio = store_data.get('남성고객비율', 0)
        female_ratio = store_data.get('여성고객비율', 0)
        핵심고객_성별 = '남성 중심' if male_ratio > female_ratio else '여성 중심' 

        age_ratios = {
            '20대이하': store_data.get('연령대20대이하고객비율', 0),
            '30대': store_data.get('연령대30대고객비율', 0),
            '40대': store_data.get('연령대40대고객비율', 0),
            '50대이상': store_data.get('연령대50대고객비율', 0),
        }
        핵심연령대_결과 = max(age_ratios, key=age_ratios.get)
        
        store_data['자동추출특징'] = {
            "핵심고객": 핵심고객_성별,
            "핵심연령대": 핵심연령대_결과,
            "매출순위": f"상권 내 상위 {store_data.get('동일상권내매출순위비율', 0):.1f}%, 업종 내 상위 {store_data.get('동일업종내매출순위비율', 0):.1f}%"
        }

        area = store_data.get('상권')
        category = store_data.get('업종')
        
        average_df = df_merchant[(df_merchant['상권'] == area) & (df_merchant['업종'] == category)]

        if average_df.empty:
            average_data = {}
        else:
            numeric_cols = average_df.select_dtypes(include=np.number).columns
            average_data = average_df[numeric_cols].mean().to_dict()

        average_data['가맹점명'] = f"{area} {category} 업종 평균"
        
        final_result = {
            "store_profile": store_data,
            "average_profile": average_data
        }
        
        clean_result = replace_nan_with_none(final_result)
        
        logger.info(f"✅ [Local Logic] '{store_data.get('가맹점명')}({merchant_id})' 프로파일링 성공 (기준년월: {store_data.get('기준년월')})")
        return clean_result

    except ValueError as e: # HTTPException을 ValueError로 변경
        logger.error(f"❌ [Local Logic ERROR] 처리 중 오류: {e}", exc_info=True)
        raise e
    except Exception as e:
        logger.critical(f"❌ [Local Logic CRITICAL] 예측하지 못한 오류: {e}\n{traceback.format_exc()}", exc_info=True)
        raise Exception(f"서버 내부 오류 발생: {e}")

# --- (끝) API 로직 통합 ---


# --- 이미지 로드 함수 ---
@st.cache_data
def load_image(image_name: str) -> Image.Image | None:
    """assets 폴더에서 이미지를 로드하고 캐시합니다."""
    try:
        image_path = config.ASSETS / image_name
        if not image_path.is_file():
            logger.error(f"이미지 파일을 찾을 수 없습니다: {image_path}")
            # ... (오류 로깅) ...
            return None
        return Image.open(image_path)
    except Exception as e:
        logger.error(f"이미지 로딩 중 오류 발생 ({image_name}): {e}", exc_info=True)
        return None

# --- (4) 데이터 로드 함수 수정 ---

@st.cache_data
def load_master_dataframe():
    """
    (수정) FastAPI 서버의 역할을 대신하여,
    앱 시작 시 'final_df.csv' 마스터 데이터 전체를 로드하고 전처리합니다.
    """
    logger.info("마스터 데이터프레임 로드 시도...")
    df = load_and_preprocess_data() # (1)에서 복사한 함수 호출
    if df is None:
        logger.critical("--- [Streamlit Error] 마스터 데이터 로딩 실패! ---")
        return None
    logger.info("--- [Streamlit] 마스터 데이터프레임 로드 및 캐시 완료 ---")
    return df

@st.cache_data
def load_merchant_list_for_ui(_df_master: pd.DataFrame):
    """
    (수정) 마스터 데이터프레임에서 UI 검색용 (ID, 이름) 목록만 추출합니다.
    (api/server.py의 GET /merchants 엔드포인트 로직)
    """
    try:
        if _df_master is None:
            return None
        logger.info(f"✅ [Local Logic] '/merchants' 가맹점 목록 요청 수신")
        merchant_list = _df_master[['가맹점ID', '가맹점명']].drop_duplicates().to_dict('records')
        logger.info(f"✅ [Local Logic] 가맹점 목록 {len(merchant_list)}개 반환 완료")
        return pd.DataFrame(merchant_list)
    except Exception as e:
        st.error(f"가게 목록을 불러오는 데 실패했습니다: {e}")
        logger.critical(f"가게 목록 로딩 실패: {e}", exc_info=True)
        return None

# --- 데이터 로드 실행 (수정) ---
# 마스터 데이터프레임을 먼저 로드합니다.
MASTER_DF = load_master_dataframe() 
if MASTER_DF is None:
    st.error("🚨 데이터 로딩 실패! data/final_df.csv 파일을 확인해주세요.")
    st.stop()

# UI용 가맹점 목록을 마스터에서 추출합니다.
merchant_df = load_merchant_list_for_ui(MASTER_DF)
if merchant_df is None:
    st.error("🚨 가맹점 목록 추출 실패!")
    st.stop()

# --- 세션 초기화 함수 ---
def initialize_session():
    """ 세션 초기화 및 AI 모듈 로드 """
    if "orchestrator" not in st.session_state:
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        if not google_api_key:
            st.error("🔑 GOOGLE_API_KEY 환경변수가 설정되지 않았습니다!")
            st.stop()
        with st.spinner("🧠 AI 모델과 빅데이터를 로딩하고 있어요... 잠시만 기다려주세요!"):
            try:
                # LLM 캐시 설정
                try:
                    from langchain.cache import InMemoryCache
                    from langchain.globals import set_llm_cache
                    set_llm_cache(InMemoryCache())
                    logger.info("--- [Streamlit] 전역 LLM 캐시(InMemoryCache) 활성화 ---")
                except ImportError:
                     logger.warning("--- [Streamlit] langchain.cache 임포트 실패. LLM 캐시 비활성화 ---")


                load_marketing_vectorstore()
                db = load_festival_vectorstore()
                if db is None:
                    st.error("💾 축제 벡터 DB 로딩 실패! 'build_vector_store.py' 실행 여부를 확인하세요.")
                    st.stop()
                logger.info("--- [Streamlit] 모든 AI 모듈 로딩 완료 ---")
            except Exception as e:
                st.error(f"🤯 AI 모듈 초기화 중 오류 발생: {e}")
                logger.critical(f"AI 모듈 초기화 실패: {e}", exc_info=True)
                st.stop()
        st.session_state.orchestrator = AgentOrchestrator(google_api_key)

    # 세션 상태 변수 초기화
    if "step" not in st.session_state:
        st.session_state.step = "get_merchant_name"
        st.session_state.messages = []
        st.session_state.merchant_id = None
        st.session_state.merchant_name = None
        st.session_state.profile_data = None
        st.session_state.consultation_result = None
        if "last_recommended_festivals" not in st.session_state:
            st.session_state.last_recommended_festivals = []

# --- 처음으로 돌아가기 함수 ---
def restart_consultation():
    """ 세션 상태 초기화 """
    keys_to_reset = ["step", "merchant_name", "merchant_id", "profile_data", "messages", "consultation_result", "last_recommended_festivals"]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]

# --- 사이드바 렌더링 함수 ---
def render_sidebar():
    """ 사이드바 렌더링 (Synapse 로고 강조 및 간격 조정) """
    with st.sidebar:
        # 로고 이미지 로드
        synapse_logo = load_image("Synapse.png")
        shinhancard_logo = load_image("ShinhanCard_Logo.png")

        col1, col2, col3 = st.columns([1, 5, 1]) # 가운데 컬럼 너비 조정
        with col2:
            if synapse_logo:
                st.image(synapse_logo, use_container_width=True)

        st.write("")
        st.markdown(" ")
        col_sh1, col_sh2, col_sh3 = st.columns([1, 5, 1]) 
        with col_sh2:
            if shinhancard_logo:
                st.image(shinhancard_logo, use_container_width=True) # 컬럼 너비에 맞춤

        st.markdown("<p style='text-align: center; color: grey; margin-top: 20px;'>2025 Big Contest</p>", unsafe_allow_html=True) # 위쪽 마진 살짝 늘림
        st.markdown("<p style='text-align: center; color: grey;'>AI DATA 활용분야</p>", unsafe_allow_html=True)
        st.markdown("---")

        if st.button('처음으로 돌아가기', key='restart_button_styled', use_container_width=True): # 버튼 아이콘 추가
            restart_consultation()
            st.rerun()

# --- 가게 검색 UI 함수 (수정) ---
def render_get_merchant_name_step():
    """ UI 1단계: 가맹점 검색 및 선택 (API 호출 로직 수정) """
    st.subheader("🔍 컨설팅 받을 가게를 검색해주세요")
    st.caption("가게 이름 또는 가맹점 ID의 일부를 입력하여 검색할 수 있습니다.")

    search_query = st.text_input(
        "가게 이름 또는 가맹점 ID 검색",
        placeholder="예: 메가커피, 스타벅스, 003AC99735 등",
        label_visibility="collapsed"
    )

    if search_query:
        mask = (
            merchant_df['가맹점명'].str.contains(search_query, case=False, na=False, regex=False) |
            merchant_df['가맹점ID'].str.contains(search_query, case=False, na=False, regex=False)
        )
        search_results = merchant_df[mask].copy()

        if not search_results.empty:
            search_results['display'] = search_results['가맹점명'] + " (" + search_results['가맹점ID'] + ")"
            options = ["⬇ 아래 목록에서 가게를 선택해주세요..."] + search_results['display'].tolist()
            selected_display_name = st.selectbox(
                "가게 선택:",
                options,
                label_visibility="collapsed"
            )

            if selected_display_name != "⬇️ 아래 목록에서 가게를 선택해주세요...":
                try:
                    selected_row = search_results[search_results['display'] == selected_display_name].iloc[0]
                    selected_merchant_id = selected_row['가맹점ID']
                    selected_merchant_name = selected_row['가맹점명']
                    button_label = f"🚀 '{selected_merchant_name}' 분석 시작하기"
                    is_selection_valid = True
                except (IndexError, KeyError):
                    button_label = "분석 시작하기"
                    is_selection_valid = False

                if st.button(button_label, disabled=not is_selection_valid, type="primary", use_container_width=True):
                    with st.spinner(f"📈 '{selected_merchant_name}' 가게 정보를 분석 중입니다... 잠시만 기다려주세요!"):
                        profile_data = None
                        try:
                            # --- (수정) API POST 요청 대신 (3)에서 만든 로컬 함수 호출 ---
                            profile_data = get_merchant_profile_logic(selected_merchant_id, MASTER_DF)
                            # --------------------------------------------------------
                            
                            if "store_profile" not in profile_data or "average_profile" not in profile_data:
                                st.error("프로필 생성 형식이 올바르지 않습니다.")
                                profile_data = None
                        except ValueError as e: # 404 오류
                            st.error(f"가게 프로필 로딩 실패: {e}")
                        except Exception as e:
                            st.error(f"가게 프로필 로딩 중 예상치 못한 오류 발생: {e}")
                            logger.critical(f"가게 프로필 로컬 로직 실패: {e}", exc_info=True)

                        if profile_data:
                            st.session_state.merchant_name = selected_merchant_name
                            st.session_state.merchant_id = selected_merchant_id
                            st.session_state.profile_data = profile_data
                            st.session_state.step = "show_profile_and_chat"
                            st.success(f"✅ '{selected_merchant_name}' 분석 완료!")
                            st.rerun()
        else:
            st.info("💡 검색 결과가 없습니다. 다른 검색어를 시도해보세요.")

# --- 프로필 및 채팅 UI 함수 ---
def render_show_profile_and_chat_step():
    """UI 2단계: 프로필 확인 및 AI 채팅"""
    st.subheader(f"✨ '{st.session_state.merchant_name}' 가게 분석 완료")
    with st.expander("📊 상세 데이터 분석 리포트 보기", expanded=True):
        try:
            display_merchant_profile(st.session_state.profile_data)
        except Exception as e:
            st.error(f"프로필 시각화 중 오류 발생: {e}")
            logger.error(f"--- [Visualize ERROR]: {e}\n{traceback.format_exc()}", exc_info=True)

    st.divider()
    st.subheader("💬 AI 컨설턴트와 상담을 시작하세요.")
    st.info("가게 분석 정보를 바탕으로 궁금한 점을 질문해보세요. (예: '20대 여성 고객을 늘리고 싶어요')")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("요청사항을 입력하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("AI 컨설턴트가 답변을 생성 중입니다...(최대 1~2분)"):
                orchestrator = st.session_state.orchestrator
                
                if "store_profile" not in st.session_state.profile_data:
                    st.error("세션에 'store_profile' 데이터가 없습니다. 다시 시작해주세요.")
                    st.stop()
                    
                agent_history = []
                history_to_convert = st.session_state.messages[:-1][-10:]
                
                for msg in history_to_convert:
                    if msg["role"] == "user":
                        agent_history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        agent_history.append(AIMessage(content=msg["content"]))
                
                result = orchestrator.invoke_agent(
                    user_query=prompt,
                    store_profile_dict=st.session_state.profile_data["store_profile"],
                    chat_history=agent_history,
                    last_recommended_festivals=st.session_state.last_recommended_festivals,
                )

                response_text = ""
                st.session_state.last_recommended_festivals = []

                if "error" in result:
                    response_text = f"오류 발생: {result['error']}"

                elif "final_response" in result:
                    response_text = result.get("final_response", "응답을 생성하지 못했습니다.")
                    intermediate_steps = result.get("intermediate_steps", [])
                    
                    try:
                        for step in intermediate_steps:
                            action = step[0]
                            tool_output = step[1]
                            
                            if hasattr(action, 'tool') and action.tool == "recommend_festivals":
                                if tool_output and isinstance(tool_output, list) and isinstance(tool_output[0], dict):
                                    recommended_list = [
                                        f.get("축제명") for f in tool_output if f.get("축제명")
                                    ]
                                    
                                    st.session_state.last_recommended_festivals = recommended_list
                                    logger.info(f"--- [Streamlit] 추천 축제 저장됨 (Intermediate Steps): {recommended_list} ---")
                                    break 
                                    
                    except Exception as e:
                        logger.critical(f"--- [Streamlit CRITICAL] Intermediate steps 처리 중 예외 발생: {e} ---", exc_info=True)

                else:
                    response_text = "알 수 없는 오류가 발생했습니다."

                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})

# --- 메인 실행 함수 ---
def main():
    st.title("🎉 MarketSync (마켓싱크)")
    st.subheader("소상공인 맞춤형 축제 추천 & 마케팅 AI 컨설턴트")
    st.caption("신한카드 빅데이터와 AI 에이전트를 활용하여, 사장님 가게에 꼭 맞는 지역 축제와 마케팅 전략을 찾아드립니다.")
    st.divider()

    initialize_session()
    render_sidebar()

    if st.session_state.step == "get_merchant_name":
        render_get_merchant_name_step()
    elif st.session_state.step == "show_profile_and_chat":
        render_show_profile_and_chat_step()

# --- 앱 실행 ---
if __name__ == "__main__":
    main()
