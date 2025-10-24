# modules/llm_provider.py

from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Optional

import config

logger = config.get_logger(__name__)

_llm_instance: Optional[ChatGoogleGenerativeAI] = None

def set_llm(llm: ChatGoogleGenerativeAI):
    """
    Orchestrator가 생성한 기본 LLM 인스턴스를
    글로벌 변수에 저장합니다.
    """
    global _llm_instance
    if _llm_instance is None:
        logger.info(f"--- [LLM Provider] Global LLM instance set. (Model: {llm.model}, Temp: {llm.temperature}) ---")
        _llm_instance = llm
    else:
        logger.info("--- [LLM Provider] Global LLM instance already set. ---")


def get_llm(temperature: float = 0.1) -> ChatGoogleGenerativeAI:
    """
    저장된 글로벌 LLM 인스턴스를 검색합니다.
    만약 도구가 요청한 temperature가 기본값과 다르면,
    기본 인스턴스의 설정을 복사하여 temperature만 변경한
    새로운 인스턴스를 반환합니다. (API 키 등은 재사용)
    """
    global _llm_instance
    if _llm_instance is None:
        logger.error("--- [LLM Provider] LLM not initialized. ---")
        raise RuntimeError(
            "LLM not initialized. The Orchestrator must call set_llm() before any tools are used."
        )
    
    if _llm_instance.temperature == temperature:
        logger.debug(f"--- [LLM Provider] Reusing global LLM instance (temp={temperature}) ---")
        return _llm_instance
    
    logger.info(f"--- [LLM Provider] Creating new LLM instance with temp={temperature} (default was {_llm_instance.temperature}) ---")
    
    try:
        # Pydantic v2+ (langchain-core 0.1.23+)
        return _llm_instance.model_copy(update={"temperature": temperature})
    except AttributeError:
        # Pydantic v1 (fallback)
        logger.warning("--- [LLM Provider] Using .copy() fallback (Pydantic v1) ---")
        return _llm_instance.copy(update={"temperature": temperature})