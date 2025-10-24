# tools/festival_recommender.py

from langchain_core.tools import tool
from typing import List, Dict, Any

import config
from modules.filtering import FestivalRecommender

logger = config.get_logger(__name__)

@tool
def recommend_festivals(user_query: str, store_profile: str) -> List[Dict[str, Any]]:
    """
    (도구) 사용자의 질문과 가게 프로필을 바탕으로 맞춤형 축제를 추천하는
    [하이브리드 5단계 파이프라인]을 실행합니다.
    1. 쿼리 재작성 (프로필 기반)
    2. 후보 검색 (임베딩 점수 - Score 1)
    3. 동적 속성 평가 (LLM 기반 - Score 2)
    4. 하이브리드 점수 계산 (Score 1 + Score 2)
    5. 최종 답변 포맷팅 (LLM 기반)
    
    이 도구는 '축제 추천해줘'와 같은 요청 시 단독으로 사용되어야 합니다.
    """
    logger.info(f"--- [Tool] (신규) 하이브리드 축제 추천 파이프라인 시작 (Query: {user_query[:30]}...) ---")
    
    # 4번 제안: 파이프라인 클래스를 인스턴스화하고 실행
    pipeline = FestivalRecommender(store_profile, user_query)
    
    # .run() 메서드가 모든 예외처리를 포함
    return pipeline.run()