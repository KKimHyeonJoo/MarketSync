# tools/marketing_strategy.py

import traceback
import json
from typing import List

from langchain_core.tools import tool

import config
from modules.llm_provider import get_llm
from modules.knowledge_base import load_marketing_vectorstore

from tools.profile_analyzer import get_festival_profile_by_name

logger = config.get_logger(__name__)


@tool
def search_contextual_marketing_strategy(user_query: str, store_profile: str) -> str:
    """
    (RAG Tool) 사용자의 질문과 가게 프로필(JSON 문자열)을 바탕으로 '마케팅 전략' Vector DB에서
    관련성이 높은 컨텍스트(전략)를 검색하고, LLM을 통해 최종 답변을 생성하여 반환합니다.
    """
    logger.info("--- [Tool] RAG 마케팅 전략 검색 호출됨 ---")
    
    try:
        retriever = load_marketing_vectorstore()
        if retriever is None:
            raise RuntimeError("마케팅 Retriever가 로드되지 않았습니다.")

        # 1. 컨텍스트를 고려한 검색 쿼리 생성
        try:
            profile_dict = json.loads(store_profile)
            profile_for_query = (
                f"가게 위치: {profile_dict.get('주소', '알 수 없음')}\n"
                f"가게 업종: {profile_dict.get('업종', '알 수 없음')}\n"
                f"핵심 고객: {profile_dict.get('자동추출특징', {}).get('핵심고객', '알 수 없음')}"
            )
        except Exception:
            profile_for_query = store_profile 

        contextual_query = f"[가게 정보:\n{profile_for_query}\n]에 대한 [질문: {user_query}]"
        logger.info(f"--- [Tool] RAG 검색 쿼리: {contextual_query} ---")
        
        # 2. Vector DB 검색
        docs = retriever.invoke(contextual_query)

        if not docs:
            logger.warning("--- [Tool] RAG 검색 결과 없음 ---")
            return "죄송합니다. 사장님의 가게 프로필과 질문에 맞는 마케팅 전략을 찾지 못했습니다. 가게의 특징을 조금 더 알려주시거나, 다른 질문을 시도해보시겠어요?"

        # 3. LLM에 전달할 컨텍스트 포맷팅
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        logger.info("--- [Tool] RAG 컨텍스트 생성 완료 ---")

        # 4. LLM을 통한 답변 재구성
        llm = get_llm(temperature=0.3) 

        # --- (사용자 요청) 프롬프트 원본 유지 ---
        prompt = f"""
        당신은 소상공인 전문 마케팅 컨설턴트입니다.
        아래 [가게 프로필]과 [참고 마케팅 전략]을 바탕으로, 사용자의 [질문]에 대한 맞춤형 마케팅 전략 3가지를 제안해주세요.

        [가게 프로필]
        {store_profile}

        [질문]
        {user_query}

        [참고 마케팅 전략]
        {context}

        [작성 가이드라인]
        1.  [참고 마케팅 전략]을 그대로 복사하지 말고, [가게 프로필]의 특징(예: 업종, 핵심 고객, 상권)과 [질문]의 의도를 조합하여 **가게에 특화된 새로운 아이디어**로 재구성해주세요.
        2.  각 전략은 구체적인 실행 방안을 포함해야 합니다.
        3.  친절하고 전문적인 말투를 사용하세요.
        4.  아래 [출력 형식]을 정확히 지켜주세요.
        5.  **취소선 금지**: 절대로 `~~text~~`와 같은 취소선 마크다운을 사용하지 마세요.

        [출력 형식]
        사장님 가게의 특성을 고려한 3가지 마케팅 아이디어를 제안해 드립니다.

        **1. [전략 제목 1]**
        * **전략 내용:** (가게의 어떤 특징을 활용하여 어떻게 실행하는지 구체적으로 서술)
        * **기대 효과:** (이 전략을 통해 얻을 수 있는 구체적인 효과)

        **2. [전략 제목 2]**
        * **전략 내용:** (가게의 어떤 특징을 활용하여 어떻게 실행하는지 구체적으로 서술)
        * **기대 효과:** (이 전략을 통해 얻을 수 있는 구체적인 효과)

        **3. [전략 제목 3]**
        * **전략 내용:** (가게의 어떤 특징을 활용하여 어떻게 실행하는지 구체적으로 서술)
        * **기대 효과:** (이 전략을 통해 얻을 수 있는 구체적인 효과)
        """


        try:
            response = llm.invoke(prompt)
            logger.info("--- [Tool] RAG + LLM 답변 생성 완료 ---")
            return response.content
        except Exception as llm_e:
            logger.critical(f"--- [Tool CRITICAL] RAG LLM 호출 중 오류: {llm_e} ---", exc_info=True)
            return f"오류: 검색된 전략을 처리하는 중 오류가 발생했습니다. (LLM 오류: {llm_e})"

    except Exception as e:
        logger.critical(f"--- [Tool CRITICAL] RAG 마케팅 전략 검색 중 오류: {e} ---", exc_info=True)
        return f"죄송합니다. 마케팅 전략을 생성하는 중 오류가 발생했습니다: {e}"


@tool
def create_festival_specific_marketing_strategy(festival_name: str, store_profile: str) -> str:
    """
    (RAG x2 Tool) 특정 축제 이름(예: '관악강감찬축제')과 가게 프로필(JSON 문자열)을 입력받아,
    '축제 DB'와 '마케팅 DB'를 *동시에* RAG로 참조하여,
    해당 축제 기간 동안 실행할 수 있는 맞춤형 마케팅 전략 *1개*를 생성합니다.
    """
    logger.info(f"--- [Tool] '*단일* 축제 맞춤형 전략 생성 (RAGx2)' 도구 호출 (대상: {festival_name}) ---")
    
    try:
        # 1. (RAG 1) 축제 정보 가져오기 (기존 도구 재사용)
        festival_profile_str = get_festival_profile_by_name.invoke({"festival_name": festival_name})
        
        if "오류" in festival_profile_str or "찾을 수 없음" in festival_profile_str:
            logger.warning(f"--- [Tool WARNING] 축제 프로필을 찾지 못함: {festival_name} ---")
            festival_profile_str = f"{{\"축제명\": \"{festival_name}\", \"정보\": \"상세 정보를 찾을 수 없습니다.\"}}"
        else:
            logger.info(f"--- [Tool] (RAG 1) 축제 프로필 로드 성공: {festival_name} ---")

        # 2. (RAG 2) 관련 마케팅 전략 검색
        marketing_retriever = load_marketing_vectorstore()
        if marketing_retriever is None:
            raise RuntimeError("마케팅 Retriever가 로드되지 않았습니다.")
        
        combined_query = f"""
        축제 정보: {festival_profile_str}
        가게 프로필: {store_profile}
        질문: 위 가게가 위 축제 기간 동안 할 수 있는 최고의 마케팅 전략은?
        """
        marketing_docs = marketing_retriever.invoke(combined_query)
        
        if not marketing_docs:
            marketing_context = "참고할 만한 마케팅 전략을 찾지 못했습니다."
            logger.warning("--- [Tool] (RAG 2) 마케팅 전략 검색 결과 없음 ---")
        else:
            marketing_context = "\n\n---\n\n".join([doc.page_content for doc in marketing_docs])
            logger.info(f"--- [Tool] (RAG 2) 마케팅 전략 컨텍스트 {len(marketing_docs)}개 확보 ---")

        # 3. LLM을 통한 최종 전략 생성
        llm = get_llm(temperature=0.5)
        
        # --- (사용자 요청) 프롬프트 원본 유지 ---
        prompt = f"""
        당신은 축제 연계 마케팅 전문 컨설턴트입니다.
        아래 [가게 프로필], [축제 프로필], [참고 마케팅 전략]을 모두 고려하여,
        [가게 프로필]의 사장님이 [축제 프로필] 기간 동안 실행할 수 있는
        **창의적이고 구체적인 맞춤형 마케팅 전략 1가지**를 제안해주세요.

        [가게 프로필]
        {store_profile}

        [축제 프로필]
        {festival_profile_str}

        [참고 마케팅 전략]
        {marketing_context}

        [작성 가이드라인]
        1.  **매우 중요:** [가게 프로필]의 특징(업종, 위치, 핵심 고객)과 [축제 프로필]의 특징(주제, 주요 방문객)을 
            **반드시 연관지어** 구체적인 전략을 만드세요.
        2.  [참고 마케팅 전략]은 아이디어 발상에만 활용하고, 복사하지 마세요.
        3.  전략은 1가지만 깊이 있게 제안합니다.
        4.  친절하고 전문적인 말투를 사용하세요.
        5.  아래 [출력 형식]을 정확히 지켜주세요.
        6.  **취소선 금지**: 절대로 `~~text~~`와 같은 취소선 마크다운을 사용하지 마세요.

        [출력 형식]
        ### 🎈 {json.loads(festival_profile_str).get('축제명', festival_name)} 맞춤형 마케팅 전략

        **1. (전략 아이디어 제목)**
        * **전략 개요:** (가게의 어떤 특징과 축제의 어떤 특징을 연관지었는지 설명)
        * **구체적 실행 방안:** (사장님이 '무엇을', '어떻게' 해야 하는지 단계별로 설명. 예: 메뉴 개발, 홍보 문구, SNS 이벤트 등)
        * **타겟 고객:** (이 전략이 축제 방문객 중 누구에게 매력적일지)
        * **기대 효과:** (예상되는 결과, 예: 신규 고객 유입, 객단가 상승 등)
        """

        try:
            response = llm.invoke(prompt)
            logger.info("--- [Tool] (RAGx2) 최종 전략 생성 완료 ---")
            return response.content
        except Exception as llm_e:
            logger.critical(f"--- [Tool CRITICAL] '축제 맞춤형 전략 생성 (RAGx2)' LLM 호출 중 오류: {llm_e} ---", exc_info=True)
            return f"오류: 검색된 전략을 처리하는 중 오류가 발생했습니다. (LLM 오류: {llm_e})"

    except Exception as e:
        logger.critical(f"--- [Tool CRITICAL] '축제 맞춤형 전략 생성 (RAG)' 중 오류: {e} ---", exc_info=True)
        return f"죄송합니다. '{festival_name}' 축제 전략을 생성하는 중 오류가 발생했습니다: {e}"
    

@tool
def create_marketing_strategies_for_multiple_festivals(festival_names: List[str], store_profile: str) -> str:
    """
    여러 개의 축제 이름 리스트와 가게 프로필(JSON 문자열)을 입력받아,
    각 축제에 특화된 맞춤형 마케팅 전략을 *모두* 생성하고 하나의 문자열로 취합하여 반환합니다.
    (예: ["청송사과축제", "부천국제만화축제"])
    """
    logger.info(f"--- [Tool] '*다수* 축제 맞춤형 전략 생성' 도구 호출 (대상: {festival_names}) ---")
    
    final_report = []
    
    if not festival_names:
        logger.warning("--- [Tool] 축제 이름 목록이 비어있음 ---")
        return "오류: 축제 이름 목록이 비어있습니다. 전략을 생성할 수 없습니다."

    # 개별 전략 생성 도구를 재사용
    for festival_name in festival_names:
        try:
            strategy = create_festival_specific_marketing_strategy.invoke({
                "festival_name": festival_name,
                "store_profile": store_profile
            })
            
            final_report.append(strategy)
            
        except Exception as e:
            error_message = f"--- [오류] '{festival_name}'의 전략 생성 중 문제가 발생했습니다: {e} ---"
            logger.critical(f"--- [Tool CRITICAL] '{festival_name}' 전략 생성 중 오류: {e} ---", exc_info=True)
            final_report.append(error_message)

    logger.info("--- [Tool] '다수 축제 맞춤형 전략 생성' 완료 ---")
    return "\n\n---\n\n".join(final_report)
