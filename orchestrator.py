# orchestrator.py

import json
import traceback
from typing import List, Optional, Dict, Any 
from pydantic import ValidationError

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools.render import render_text_description

import config
from modules.llm_provider import set_llm
from modules.profile_utils import get_chat_profile_dict

# tools/tool_loader.py 에서 모든 도구를 가져옴
from tools.tool_loader import ALL_TOOLS

logger = config.get_logger(__name__)

# --- 헬퍼 함수를 공통 유틸리티 호출로 변경 ---
def _get_chat_profile_json_string(store_profile_dict: Dict[str, Any]) -> str:
    """
    공통 유틸리티(profile_utils.py)를 호출하여 '채팅용 프로필 딕셔너리'를 생성하고,
    이를 JSON 문자열로 변환하여 반환합니다.
    """
    try:
        summary_dict = get_chat_profile_dict(store_profile_dict)
        return json.dumps(summary_dict, ensure_ascii=False)
        
    except Exception as e:
        logger.critical(f"--- [Orchestrator CRITICAL] 채팅용 JSON 생성 실패: {e} ---", exc_info=True)
        fallback_data = {
            "업종": store_profile_dict.get('업종', '알 수 없음'),
            "자동추출특징": store_profile_dict.get('자동추출특징', {}),
            "주소": store_profile_dict.get('가맹점주소', '알 수 없음'),
            "error": "프로필 요약 중 오류 발생"
        }
        return json.dumps(fallback_data, ensure_ascii=False)


class AgentOrchestrator:
    def __init__(self, google_api_key):
        """Gemini Flash 기반 Agent Orchestrator 초기화"""
        self.llm = ChatGoogleGenerativeAI(
            model=config.LLM_MODEL_NAME,
            google_api_key=google_api_key,
            temperature=0.1  
        )
        set_llm(self.llm)

        # tool_loader 에서 도구 목록을 가져옴
        self.tools = ALL_TOOLS

        self.rendered_tools = render_text_description(self.tools)

        self.system_prompt_template = """
        {base_system_prompt}

        ---
        📦 [현재 가게 프로필 (JSON)]
        {store_profile_context}

        📜 [이전 추천 축제 리스트]
        {last_recommended_festivals}

        ---
        💡 반드시 위 정보를 기반으로 판단하되,
        도구 라우팅 규칙(1~4순위)에 따라 *적절한 단 하나의 도구를 호출*해야 합니다.
        """

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt_template),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)

        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )
        logger.info(f"--- [Streamlit] AgentOrchestrator 초기화 완료 (Model: {config.LLM_MODEL_NAME}) ---")


    def setup_system_prompt(self):
        """Gemini Flash 전용 강화 프롬프트"""
        
        logger.info("--- [Orchestrator] 시스템 프롬프트 설정 완료 ---")

        # --- (사용자 요청) 프롬프트 원본 유지 ---
        return f"""
        당신은 **신한카드 데이터 기반 지역축제 전문 AI 컨설턴트**입니다.
        당신의 임무는 사장님의 가게 정보를 기반으로
        **가게 분석 → 축제 추천 → 축제 분석 → 마케팅 전략 제안**을 수행하는 것입니다.

        ---
        🔧 [사용 가능한 도구 목록]
        (도구 목록은 에이전트에 내장되어 있으며, 아래 [도구 라우팅 규칙]에 따라 호출됩니다.)

        ---
        🎯 **[핵심 임무 요약]**
        1️⃣ 사용자의 요청을 완수하기 위해 **필요한 모든 도구를 자율적으로 호출**해야 합니다. 때로는 **여러 도구를 순차적으로 호출**해야 할 수도 있습니다. (예: 축제 추천 → 마케팅 전략 생성)
        2️⃣ **도구 호출 없이** "죄송합니다" 또는 "잘 모르겠습니다" 같은 답변을 생성하는 것은 절대 금지입니다.
        3️⃣ 모든 요청은 반드시 적합한 도구 호출로 이어져야 합니다.
        4️⃣ 모든 도구 실행 결과를 바탕으로, 사장님에게 제공할 [최종 답변]을
           **자연스러운 한국어(마크다운 형식)**로 생성합니다.

        ---
        🧭 **[도구 라우팅 규칙 (우선순위 적용)]**

        **[1순위] 축제 추천 요청**
        - 키워드: "축제 추천", "참여할 만한 축제", "어떤 축제", "행사 찾아줘", "어디가 좋아"
        - → `recommend_festivals`

        **[2순위] 특정 축제 분석/전략 요청**
       - **2-1. 마케팅 전략 요청 (축제 1개)**: 축제 이름이 1개 포함되어 있고 '마케팅', '전략' 등의 키워드가 있는 경우
            - → `create_festival_specific_marketing_strategy`
        - **2-2. 마케팅 전략 요청 (축제 2개 이상)**: 축제 이름이 2개 이상 포함되어 있고 '마케팅', '전략' 등의 키워드가 있는 경우
            - → `create_marketing_strategies_for_multiple_festivals`
        - **2-3. 축제 상세 분석 요청**: "~축제 어때?", "분석해줘"
            - → `analyze_festival_profile`

        **[3순위] 가게 분석 요청**
        - 키워드: “우리 가게”, “SWOT”, “고객 특성”, “분석해줘”
        - → `analyze_merchant_profile`

        **[4순위] 일반 마케팅/홍보 요청**
        - 키워드: “마케팅”, “홍보”, “매출”, “전략”
        - → `search_contextual_marketing_strategy`

        **[기타]**
        - 명확히 분류되지 않으면 4순위 도구 사용
        - → `search_contextual_marketing_strategy`

        ---
        ✅ **[행동 체크리스트]**
        - 1️⃣ 사용자의 요청이 **완전히 해결될 때까지** 필요한 모든 도구를 호출할 것  
        - 2️⃣ [1순위] 작업 시, 마케팅 전략 요청이 있었는지 **반드시 재확인**하고 2단계 도구 호출을 결정할 것
        - 3️⃣ 도구 호출 없이 종료하지 말 것
        - 4️⃣ 최종 답변은 자연스러운 한국어(마크다운)로 생성할 것

        ---
        ✍️ **[최종 답변 가이드라인] (매우 중요)**
        1.  **친절한 전문가 말투**: 항상 사장님을 대하듯, 전문적이면서도 친절하고 이해하기 쉬운 말투를 사용합니다.
        2.  **(요청 2) 추천 점수 표시**: `recommend_festivals` 도구의 결과를 포맷팅할 때, 각 축제 이름 옆이나 바로 아래에 **(추천 점수: XX.X점)**과 같이 '추천_점수'를 **반드시** 명시하세요.
        3.  **(요청 4) 취소선 금지**: 절대로 `~~text~~`와 같은 취소선 마크다운을 사용하지 마세요.
        4.  **(요청 3) 다음 질문 제안**: 사용자가 다음에 무엇을 할 수 있을지 알 수 있도록, 답변의 **가장 마지막**에 아래와 같은 [다음 질문 예시]를 2~3개 제안하세요.

        [다음 질문 예시]
        * "방금 추천해준 축제들의 마케팅 전략을 알려줘"
        * "[축제이름]에 대한 마케팅 전략을 짜줘"
        * "내 가게의 강점을 활용한 다른 홍보 방법은?"
        """
    
    def invoke_agent(
        self,
        user_query: str,                  
        store_profile_dict: dict,          
        chat_history: list,
        last_recommended_festivals: Optional[List[str]] = None,
    ):

        """사용자 입력을 받아 Agent를 실행하고 결과를 반환"""
        logger.info(f"--- [Orchestrator] Agent 실행 시작 (Query: {user_query[:30]}...) ---")
        
        base_system_prompt = self.setup_system_prompt()
        store_profile_chat_json_str = _get_chat_profile_json_string(store_profile_dict)
        last_recommended_festivals_str = (
            "없음" if not last_recommended_festivals else str(last_recommended_festivals)
        )
        
        try:
            response = self.agent_executor.invoke({
                "input": user_query, 
                "chat_history": chat_history,
                "store_profile_context": store_profile_chat_json_str, 
                "store_profile": store_profile_chat_json_str,       
                "last_recommended_festivals": last_recommended_festivals_str,
                "base_system_prompt": base_system_prompt, 
            })

            output_text = response.get("output", "").strip()

            is_garbage_response = (
                len(output_text) < 10 and ("}" in output_text or "`" in output_text)
            )

            if not output_text or is_garbage_response:
                
                if is_garbage_response:
                    logger.warning(f"--- [Orchestrator WARNING] 비정상 응답 감지 ('{output_text}') → 재시도 수행 ---")
                else:
                    logger.warning("--- [Orchestrator WARNING] 응답 비어있음 → 재시도 수행 ---")

                retry_input = f"""
                [재시도 요청]
                이전 응답이 비어있거나 비정상적인 값('{output_text}')이었습니다.
                사용자 질문: "{user_query}" 

                당신은 반드시 하나의 도구를 호출해야 합니다.
                도구 라우팅 규칙(1~4순위)에 따라 적절한 도구를 선택하고 호출하십시오.
                """
                
                response = self.agent_executor.invoke({
                    "input": retry_input,
                    "chat_history": chat_history,
                    "store_profile_context": store_profile_chat_json_str, 
                    "store_profile": store_profile_chat_json_str,       
                    "last_recommended_festivals": last_recommended_festivals_str,
                    "base_system_prompt": base_system_prompt,
                })
                
                final_response = response.get("output", "").strip()
            
            else:
                final_response = output_text

            if not final_response:
                final_response = "죄송합니다. 요청을 처리하는 중 오류가 발생했습니다. 질문을 조금 더 명확히 말씀해주시겠어요?"

            logger.info("--- [Orchestrator] Agent 실행 완료 ---\n")
            
            return {
                "final_response": final_response,
                "intermediate_steps": response.get("intermediate_steps", [])
            }

        except ValidationError as e:
            logger.error(f"--- [Orchestrator Pydantic ERROR] {e} ---\n", exc_info=True)
            return {
                "final_response": f"죄송합니다. 도구 입력값(Pydantic) 오류가 발생했습니다: {e}",
                "intermediate_steps": []
            }

        except Exception as e:
            logger.critical(f"--- [Orchestrator CRITICAL ERROR] {e} ---\n", exc_info=True)
            return {
                "final_response": f"죄송합니다. 알 수 없는 오류가 발생했습니다: {e}",
                "intermediate_steps": []
            }
