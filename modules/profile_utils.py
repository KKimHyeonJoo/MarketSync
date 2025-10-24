# modules/profile_utils.py

from typing import Dict, Any
import config

logger = config.get_logger(__name__)

def get_chat_profile_dict(store_profile_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    FastAPI (server.py)에서 받은 'store_profile' 딕셔너리를 기반으로,
    visualization.py와 orchestrator.py에서 공통으로 사용할
    '채팅용 프로필 딕셔너리' (사장님이 요청하신 항목)를 생성합니다.
    
    이 함수가 '채팅용 프로필'의 단일 정의(Source of Truth) 역할을 합니다.
    """
    try:
        # 1. 기본 정보
        chat_profile_data = {
            "가맹점명": store_profile_dict.get('가맹점명', 'N/A'),
            "가맹점ID": store_profile_dict.get('가맹점ID', 'N/A'),
            "상권": store_profile_dict.get('상권', 'N/A'),
            "업종": store_profile_dict.get('업종', 'N/A'),
            "주소": store_profile_dict.get('가맹점주소', 'N/A'), 
            "운영 기간 수준": store_profile_dict.get('운영개월수_수준', 'N/A'),
            "매출 수준": store_profile_dict.get('매출구간_수준', 'N/A'),
            "매출 건수 수준": store_profile_dict.get('월매출건수_수준', 'N/A'),
            "방문 고객수 수준": store_profile_dict.get('월유니크고객수_수준', 'N/A'),
            "객단가 수준": store_profile_dict.get('월객단가_수준', 'N/A'),
            "신규/재방문율": f"신규 {(store_profile_dict.get('신규고객비율') or 0):.1f}% / 재방문 {(store_profile_dict.get('재이용고객비율') or 0):.1f}%",
            "동일 상권 대비 매출 순위": f"상위 {(store_profile_dict.get('동일상권내매출순위비율') or 0):.1f}%",
            "동일 업종 대비 매출 순위": f"상위 {(store_profile_dict.get('동일업종내매출순위비율') or 0):.1f}%"
        }
        
        # 2. '자동추출특징' 추가
        chat_profile_data["자동추출특징"] = store_profile_dict.get('자동추출특징', {})
        
        return chat_profile_data
        
    except Exception as e:
        logger.critical(f"--- [Profile Utils CRITICAL] 채팅 프로필 딕셔너리 생성 실패: {e} ---", exc_info=True)
        return {
            "업종": store_profile_dict.get('업종', '알 수 없음'),
            "자동추출특징": store_profile_dict.get('자동추출특징', {}),
            "주소": store_profile_dict.get('가맹점주소', '알 수 없음'),
            "error": "프로필 요약 중 오류 발생"
        }