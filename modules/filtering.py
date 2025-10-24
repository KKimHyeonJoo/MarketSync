# modules/filtering.py

import json
import traceback
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

from langchain_core.messages import HumanMessage 
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import config
from modules.knowledge_base import load_festival_vectorstore
from modules.llm_provider import get_llm
from utils.parser_utils import extract_json_from_llm_response 

logger = config.get_logger(__name__)

# 4번 제안: 파이프라인 로직을 클래스로 캡슐화
class FestivalRecommender:
    """
    하이브리드 축제 추천 파이프라인을 캡슐화한 클래스.
    """
    def __init__(self, store_profile: str, user_query: str, specific_intent: Optional[str] = None):
        self.store_profile = store_profile
        self.user_query = user_query
        self.specific_intent = specific_intent
        
        # LLM 인스턴스를 미리 생성
        self.llm_temp_01 = get_llm(0.1)
        self.llm_temp_03 = get_llm(0.3)
        
        # VectorStore 로드
        self.vectorstore = load_festival_vectorstore()
        
        # 가중치 (config에서 로드)
        self.embedding_weight = config.FESTIVAL_EMBEDDING_WEIGHT
        self.dynamic_weight = config.FESTIVAL_DYNAMIC_WEIGHT

    def _rewrite_query(self) -> str:
        """
        (1단계) 가게 프로필과 사용자 질문을 바탕으로 Vector Store 검색용 쿼리를 LLM이 재작성합니다.
        """
        logger.info("--- [Filter 1/5] 쿼리 재작성 시작 ---")
        
        intent_prompt = f"사용자의 구체적인 요청: {self.specific_intent}" if self.specific_intent else ""

        # --- (사용자 요청) 프롬프트 원본 유지 ---
        prompt = f"""
        당신은 소상공인 마케팅을 위한 AI 컨설턴트입니다.
        당신의 임무는 [가게 프로필]과 [사용자 질문]의 의도를 완벽하게 이해하고, 
        이 가게에 가장 적합한 축제를 찾기 위한 '최적의 검색 키워드'를 생성하는 것입니다.
        검색 엔진은 '축제 소개 내용'을 기반으로 유사도를 측정하여 축제를 찾아냅니다.
    
        [가게 프로필]
        {self.store_profile}

        [사용자 질문]
        {self.user_query}
        {intent_prompt}

        [검색 키워드 생성 가이드]
        1. 가게의 '업종', '상권', '주요 고객층(성별/연령)'을 핵심 키워드로 사용하세요.
        2. 가게의 '강점'이나 '약점'을 보완할 수 있는 방향을 고려하세요. 
        (예: '신규 고객 확보'가 필요하면 '유동 인구', '관광객', '대규모' 등)
        (예: '객단가'가 낮으면 '구매력 높은', '3040대 직장인' 등)
        3. 사용자 질문의 의도를 반영하세요. (예: '여름' 축제, '특정 지역' 축제)
        4. 5~8개의 핵심 키워드를 조합하여 자연스러운 문장이나 구문으로 만드세요.

        [검색 키워드 생성 단계 및 가이드]
        1. **분석:** 가게 프로필(업종, 상권, 주요 고객)을 바탕으로 현재 가게가 마케팅적으로 가장 필요로 하는 것(예: 신규 고객 유입, 객단가 상승, 특정 연령대 확보)이 무엇인지 내부적으로 분석합니다.
        2. **목표 설정:** 분석 결과와 사용자 질문의 의도를 결합하여, 축제에 기대하는 최종적인 목표를 명확히 합니다. (예: "20대 여성의 유입을 증가시킬 축제", "가족 단위 관광객이 많은 축제")
        3. **키워드 추출:** 설정된 목표에 부합하는 **핵심 키워드 7개를 명사 형태로 추출**합니다.
        - '업종', '주요 고객층(성별/연령)', '필요한 고객 유입 형태(예: 관광객, 가족단위, 직장인)', '시즌/테마'를 포함하여 구체적으로 만듭니다.


        [출력 형식]
        (오직 재작성된 쿼리만 출력)
        """
        
        try:
            response = self.llm_temp_01.invoke([HumanMessage(content=prompt)])
            rewritten_query = response.content.strip().replace('"', '').replace("'", "")
            
            if not rewritten_query:
                logger.warning("--- [Filter 1/5 ERROR] 쿼리 재작성 실패, 원본 쿼리 사용 ---")
                return self.user_query
                
            return rewritten_query
            
        except Exception as e:
            logger.critical(f"--- [Filter 1/5 CRITICAL ERROR] {e} ---", exc_info=True)
            return self.user_query # 실패 시 원본 쿼리 반환

    def _search_candidates(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """
        (2단계) 재작성된 쿼리를 사용하여 Vector Store에서 K개의 후보를 검색합니다.
        """
        logger.info(f"--- [Filter 2/5] 후보 검색 (임베딩 점수) 시작 (Query: {query}) ---")
        try:
            if self.vectorstore is None:
                raise RuntimeError("축제 벡터스토어가 로드되지 않았습니다.")
                
            candidates_with_scores = self.vectorstore.similarity_search_with_relevance_scores(query, k=k)
            return candidates_with_scores

        except Exception as e:
            logger.critical(f"--- [Filter 2/5 CRITICAL ERROR] {e} ---", exc_info=True)
            return []

    def _evaluate_candidates_dynamically(self, candidates: List[Document]) -> Dict[str, Dict[str, Any]]:
        """
        (3단계) LLM을 사용하여 후보들의 '동적 속성'을 평가합니다.
        """
        logger.info(f"--- [Filter 3/5] 동적 속성 평가 (LLM) 시작 (후보 {len(candidates)}개) ---")
        
        candidates_data = []
        for doc in candidates:
            meta = doc.metadata
            candidates_data.append({
                "축제명": meta.get('축제명'),
                "주요성별": meta.get('주요성별'),
                "주요연령대": meta.get('주요연령대'),
                "주요고객층": meta.get('주요고객층'),
                "주요방문자": meta.get('주요방문자'),
                "축제인기": meta.get('축제인기'),
                "축제인기도": meta.get('축제인기도'),
                "인기도_점수": meta.get('인기도_점수')
            })
        
        candidates_json_str = json.dumps(candidates_data, ensure_ascii=False, indent=2)

        # --- (사용자 요청) 프롬프트 원본 유지 ---
        prompt = f"""
        당신은 냉철한 축제 데이터 분석가입니다. [가게 프로필]과 [사용자 요청]을 바탕으로,
        각 [축제 후보]가 이 가게의 '타겟 고객' 및 '마케팅 목표'와 얼마나 잘 맞는지
        **오직 제공된 '동적 속성' (주요성별, 주요연령대, 주요고객층, 주요방문자, 인기도)만을
        기준으로** 평가하고 '동적_점수' (0~100점)를 매기세요.

        [가게 프로필]
        {self.store_profile}

        [사용자 요청]
        {self.user_query}

        [평가 대상 축제 후보 목록 (JSON)]
        {candidates_json_str}

        [동적 점수 평가 가이드]
        1.  **타겟 일치 (성별/연령)**: 가게의 '핵심고객'(예: 30대 여성)과 축제의 '주요성별', '주요연령대'가 일치할수록 높은 점수를 주세요.
        2.  **고객층 일치**: 가게의 '업종'(예: 카페)과 축제의 '주요고객층'(예: 2030 여성, 연인)이 시너지가 날수록 높은 점수를 주세요.
        3.  **방문자 특성**: 가게가 '신규 고객 확보'가 필요하고 축제의 '주요방문자'가 '외지인'이라면 높은 점수를 주세요. 반대로 '단골 확보'가 목표인데 '현지인' 방문자가 많다면 높은 점수를 주세요.
        4.  **인기도**: '축제인기', '축제인기도', '인기도_점수'가 높을수록 방문객 수가 보장되므로 높은 점수를 주세요.
        5.  **복합 평가**: 이 모든 요소를 종합하여 0점에서 100점 사이의 '동적_점수'를 부여하세요.
        6.  **이유 작성**: 왜 그런 점수를 주었는지 '평가_이유'에 간략히 요약하세요.

        [출력 형식 (JSON 리스트)]
        [
          {{
            "축제명": "[축제 이름]",
            "동적_점수": 85,
            "평가_이유": "가게의 핵심 고객인 30대 여성과 축제의 주요연령대/주요성별이 일치하며, '외지인' 방문자 특성이 신규 고객 확보 목표에 부합함."
          }},
          ...
        ]
        """
        
        try:
            response = self.llm_temp_01.invoke([HumanMessage(content=prompt)])
            response_text = response.content.strip()
            
            # 5번 제안: 공통 파서 사용
            scores_list = extract_json_from_llm_response(response_text)
            
            scores_dict = {
                item['축제명']: {
                    "dynamic_score": item.get('동적_점수', 0),
                    "dynamic_reason": item.get('평가_이유', 'N/A')
                } 
                for item in scores_list if isinstance(item, dict) and '축제명' in item
            }
            return scores_dict
            
        except (ValueError, json.JSONDecodeError) as e:
            logger.error(f"--- [Filter 3/5 CRITICAL ERROR] 동적 점수 JSON 파싱 실패: {e} ---")
            logger.debug(f"LLM 원본 응답 (앞 500자): {response_text[:500]} ...")
            return {} # 오류 발생 시 빈 딕셔너리 반환 (Fallback)
        except Exception as e:
            logger.critical(f"--- [Filter 3/5 CRITICAL ERROR] (Outer Catch) {e} ---", exc_info=True)
            return {}

    def _calculate_hybrid_scores(
        self,
        embedding_candidates: List[Tuple[Document, float]], 
        dynamic_scores: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        (4단계) Score 1(임베딩)과 Score 2(동적)를 가중 합산하여 최종 '하이브리드 점수'를 계산합니다.
        """
        logger.info("--- [Filter 4/5] 하이브리드 점수 계산 시작 ---")
        hybrid_results = []

        for doc, embedding_score in embedding_candidates:
            festival_name = doc.metadata.get('축제명')
            if not festival_name:
                continue
                
            normalized_embedding_score = embedding_score * 100
            dynamic_eval = dynamic_scores.get(festival_name, {"dynamic_score": 0, "dynamic_reason": "N/A"})
            dynamic_score = dynamic_eval["dynamic_score"]
            
            hybrid_score = (normalized_embedding_score * self.embedding_weight) + \
                           (dynamic_score * self.dynamic_weight)
                           
            hybrid_results.append({
                "document": doc,
                "metadata": doc.metadata,
                "score_embedding": normalized_embedding_score,
                "score_dynamic": dynamic_score,
                "score_dynamic_reason": dynamic_eval["dynamic_reason"],
                "score_hybrid": hybrid_score
            })
            
        hybrid_results.sort(key=lambda x: x.get("score_hybrid", 0), reverse=True)
        return hybrid_results


    # 2026년 날짜 예측 헬퍼 함수
    def _predict_next_year_date(self, date_str_2025: Optional[str]) -> str:
        """2025년 날짜 문자열(YYYY.MM.DD~...)을 받아 2026년 예상 시기를 텍스트로 반환합니다."""
        if not date_str_2025 or not isinstance(date_str_2025, str):
            return "2026년 정보 없음" # 날짜 정보 없으면 명시적 반환

        try:
            # "~" 앞부분만 사용하여 시작 날짜 파싱 (YYYY.MM.DD 형식 가정)
            start_date_str = date_str_2025.split('~')[0].strip()
            date_2025 = pd.to_datetime(start_date_str, format='%Y.%m.%d', errors='coerce')

            if pd.isna(date_2025): # YYYY.MM.DD 파싱 실패 시 다른 형식 시도 (예: YYYY-MM-DD)
                date_2025 = pd.to_datetime(start_date_str, errors='coerce')

            if pd.isna(date_2025): # 최종 파싱 실패 시
                 logger.warning(f"날짜 예측 실패: '{start_date_str}' (원본: '{date_str_2025}') 형식을 인식할 수 없습니다.")
                 return f"2026년 정보 없음 (2025년: {date_str_2025})"

            month = date_2025.month
            day = date_2025.day

            if day <= 10:
                timing = f"{month}월 초"
            elif day <= 20:
                timing = f"{month}월 중순"
            else:
                timing = f"{month}월 말"

            return f"2026년 {timing}경 예상 (2025년: {date_str_2025})"
        except Exception as e:
            logger.error(f"날짜 예측 중 오류 ({date_str_2025}): {e}")
            return f"2026년 정보 없음 (오류: {e})"

    def _format_recommendation_results(
        self,
        ranked_list: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        
        """ (5단계) 최종 답변 포맷팅 (LLM) """
        logger.info(f"--- [Filter 5/5] 최종 답변 포맷팅 (LLM) 시작 (Top {top_k}) ---")
        top_candidates = ranked_list[:top_k]
        candidates_data = []
        for candidate in top_candidates:
            meta = candidate["metadata"]
            date_2025 = meta.get('2025_기간')
            predicted_2026_timing = self._predict_next_year_date(date_2025)
            candidates_data.append({
                "축제명": meta.get('축제명'),
                "소개": meta.get('소개'),
                "predicted_2026_timing": predicted_2026_timing,
                "주요고객층": meta.get('주요고객층'),
                "주요방문자": meta.get('주요방문자'),
                "축제인기": meta.get('축제인기'),
                "홈페이지": meta.get('홈페이지'),
                "추천_점수": round(candidate["score_hybrid"], 1),
                "추천_근거_키워드": f"키워드/소개 일치도 ({round(candidate['score_embedding'], 0)}점)",
                "추천_근거_동적": f"가게 맞춤성({round(candidate['score_dynamic'], 0)}점): {candidate['score_dynamic_reason']}"
            })
        candidates_json_str = json.dumps(candidates_data, ensure_ascii=False, indent=2)

        prompt = f"""
        당신은 소상공인 컨설턴트입니다. [가게 프로필]과 AI가 분석한 [최종 추천 축제 목록]을 바탕으로,
        사장님께 제안할 최종 추천 답변을 생성하세요.

        [가게 프로필]
        {self.store_profile}

        [최종 추천 축제 목록 (JSON) - 소개, 2026년 예상 시기 포함]
        {candidates_json_str}

        [최종 답변 생성 가이드라인]
        1.  **[최종 추천 축제 목록]의 모든 정보**를 사용하여 최종 답변을 JSON 형식으로 생성합니다.
        2.  '추천_이유'는 '추천_근거_키워드'와 '추천_근거_동적'을 조합하여 **자연스러운 서술형 문장**으로 작성하세요.
        3.  **(수정) '축제_기본정보'**: 입력 JSON의 **'소개', '주요고객층', '주요방문자', '축제인기'** 정보를 조합하여 축제를 설명하는 자연스러운 문장으로 작성하세요. '소개' 내용을 바탕으로 축제의 핵심 내용을 요약하고, 고객층/방문자/인기도 정보를 덧붙입니다. (예: "**'{{소개 요약}}'**을(를) 주제로 하는 축제이며, 주로 **{{주요고객층}}**이 방문하고 **{{주요방문자}}** 특성을 보입니다. (인기도: **{{축제인기}}**)")
        4.  **(중요) '2026년 예상 시기'**: 입력 JSON에 있는 **`predicted_2026_timing` 값을 그대로** 가져와서 출력 JSON의 `'2026년 예상 시기'` 필드 값으로 사용하세요. **절대 직접 계산하거나 수정하지 마세요.**
        5.  **(중요) 단점 제외**: '단점'이나 '부적합한 이유'는 절대 출력하지 마세요.
        6.  **(중요) 취소선 금지**: 절대로 `~~text~~`와 같은 취소선 마크다운을 사용하지 마세요.
        7.  **출력 형식 (JSON)**: 반드시 아래의 JSON 리스트 형식으로만 응답하세요. 다른 설명 없이 JSON만 출력해야 합니다.

        [응답 형식 (JSON 리스트)]
        [
          {{
            "축제명": "[축제 이름]",
            "추천_점수": 95.2,
            "축제_기본정보": "[축제 소개 요약, 주요 고객층, 주요 방문자, 인기도를 조합한 서술형 문장]",
            "추천_이유": "[가게 프로필과 추천 근거를 바탕으로 이 축제를 추천하는 이유를 서술형으로 작성.]",
            "홈페이지": "[축제 홈페이지 URL]",
            "2026년 예상 시기": "[입력 JSON의 predicted_2026_timing 값을 그대로 사용]"
          }},
          ...
        ]
        """
        response_text = ""
        try:
            response = self.llm_temp_03.invoke([HumanMessage(content=prompt)])
            response_text = response.content.strip()
            final_list = extract_json_from_llm_response(response_text)
            return final_list
        except (ValueError, json.JSONDecodeError) as e:
            logger.error(f"--- [Filter 5/5 CRITICAL ERROR] 최종 답변 JSON 파싱 실패: {e} ---")
            logger.debug(f"LLM 원본 응답 (앞 500자): {response_text[:500]} ...")
            return [{"error": f"최종 답변 생성 중 JSON 파싱 오류 발생: {e}", "details": response_text}]
        except Exception as e:
            logger.critical(f"--- [Filter 5/5 CRITICAL ERROR] (Outer Catch) {e} ---", exc_info=True)
            return [{"error": f"최종 답변 생성 중 알 수 없는 오류 발생: {e}"}]


    def run(self, search_k: int = 10, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        파이프라인 1~5단계를 순차적으로 실행합니다.
        """
        try:
            # 1단계: 쿼리 재작성
            rewritten_query = self._rewrite_query()
            logger.info(f"--- [Filter 1/5] 쿼리 재작성 완료: {rewritten_query} ---")

            # 2단계: 후보 검색
            embedding_candidates = self._search_candidates(query=rewritten_query, k=search_k)
            if not embedding_candidates:
                logger.warning("--- [Filter 2/5] 후보 검색 결과 없음 ---")
                return [{"error": "추천할 만한 축제를 찾지 못했습니다."}]
            
            logger.info(f"--- [Filter 2/5] 후보 검색 완료 (후보 {len(embedding_candidates)}개) ---")

            # 3단계: 동적 속성 평가
            candidate_docs = [doc for doc, score in embedding_candidates]
            dynamic_scores_dict = self._evaluate_candidates_dynamically(candidates=candidate_docs)
            
            if not dynamic_scores_dict:
                logger.warning("--- [Filter 3/5 WARNING] 동적 속성 평가 실패. 임베딩 점수만으로 추천을 진행합니다. ---")
                # dynamic_scores_dict = {} (빈 딕셔너리로 계속 진행)

            logger.info(f"--- [Filter 3/5] 동적 속성 평가 완료 ({len(dynamic_scores_dict)}개) ---")
            
            # 4단계: 하이브리드 점수 계산
            hybrid_results = self._calculate_hybrid_scores(
                embedding_candidates=embedding_candidates,
                dynamic_scores=dynamic_scores_dict
            )
            logger.info(f"--- [Filter 4/5] 하이브리드 점수 계산 및 정렬 완료 ---")

            # 5단계: 최종 답변 포맷팅
            final_recommendations = self._format_recommendation_results(
                ranked_list=hybrid_results,
                top_k=top_k
            )
            logger.info(f"--- [Filter 5/5] 최종 답변 포맷팅 완료 ---")
            
            # 5단계(LLM 포맷팅) 실패 시 Fallback
            if final_recommendations and isinstance(final_recommendations, list) and "error" in final_recommendations[0]:
                 logger.warning(f"--- [Tool WARNING] 최종 답변 포맷팅 실패. 4단계 원본 데이터로 Fallback. ({final_recommendations[0]['error']}) ---")
                 
                 fallback_results = []
                 for item in hybrid_results[:top_k]:
                     meta = item.get("metadata", {})
                     fallback_results.append({
                         "축제명": meta.get("축제명", "N/A"),
                         "추천_점수": round(item.get("score_hybrid", 0), 1),
                         "추천_이유": f"임베딩({round(item.get('score_embedding',0),0)}점), 맞춤성({round(item.get('score_dynamic',0),0)}점): {item.get('score_dynamic_reason', 'N/A')}",
                         "축제_기본정보": meta.get("소개", "N/A")[:100] + "...",
                         "홈페이지": meta.get("홈페이지", "N/A")
                     })
                 return fallback_results
                 
            return final_recommendations

        except Exception as e:
            logger.critical(f"--- [Tool CRITICAL] 축제 추천 파이프라인 전체 오류: {e} ---", exc_info=True)
            return [{"error": f"축제를 추천하는 과정에서 예기치 못한 오류가 발생했습니다: {e}"}]
