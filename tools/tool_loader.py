# tools/tool_loader.py

from .festival_recommender import recommend_festivals
from .marketing_strategy import (
    search_contextual_marketing_strategy,
    create_festival_specific_marketing_strategy,
    create_marketing_strategies_for_multiple_festivals
)
from .profile_analyzer import (
    get_festival_profile_by_name,
    analyze_merchant_profile,
    analyze_festival_profile,
)

# 오케스트레이터가 사용할 최종 도구 리스트
ALL_TOOLS = [
    recommend_festivals,                      # (통합) 가게 맞춤형 축제 추천 (쿼리 재작성 ~ 최종 랭킹) 
    get_festival_profile_by_name,             # (DB조회) 축제 이름으로 상세 프로필(JSON) 검색
    search_contextual_marketing_strategy,     # (RAG) 일반적인 마케팅/홍보 전략을 Vector DB에서 검색
    create_festival_specific_marketing_strategy,        # (LLM) *단일* 축제에 대한 맞춤형 마케팅 전략 생성 
    create_marketing_strategies_for_multiple_festivals, # (LLM) *여러* 축제에 대한 맞춤형 마케팅 전략 동시 생성 
    analyze_merchant_profile,                 # (LLM) 가게 프로필(JSON)을 받아 SWOT/고객 특성 분석 
    analyze_festival_profile,                 # (LLM) 축제 프로필(JSON)을 받아 핵심 특징/방문객 분석 
]