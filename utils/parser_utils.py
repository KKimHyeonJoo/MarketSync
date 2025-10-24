# utils/parser_utils.py

import re
import json
from typing import List, Dict, Any, Union

def extract_json_from_llm_response(response_text: str) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """
    LLM 응답 텍스트에서 ```json ... ``` 또는 [...] 또는 {...} 블록을
    안전하게 추출하고 파싱합니다.
    실패 시 ValueError를 발생시킵니다.
    """
    json_str = None
    
    # 1. ```json [...] ``` 마크다운 블록 검색 (가장 우선)
    # re.DOTALL (s) 플래그: 줄바꿈 문자를 포함하여 매칭
    # re.MULTILINE (m) 플래그: ^, $가 각 줄의 시작/끝에 매칭
    json_match = re.search(
        r'```json\s*([\s\S]*?)\s*```', 
        response_text, 
        re.DOTALL | re.IGNORECASE
    )
    
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        # 2. 마크다운이 없다면, 첫 번째 { 또는 [ 를 찾음
        first_bracket_match = re.search(r'[{|\[]', response_text)
        if first_bracket_match:
            start_index = first_bracket_match.start()
            
            # 응답이 리스트([])로 시작하는 경우
            if response_text[start_index] == '[':
                list_match = re.search(r'(\[[\s\S]*\])', response_text[start_index:], re.DOTALL)
                if list_match:
                    json_str = list_match.group(0)
            
            # 응답이 딕셔너리({})로 시작하는 경우
            elif response_text[start_index] == '{':
                 dict_match = re.search(r'(\{[\s\S]*\})', response_text[start_index:], re.DOTALL)
                 if dict_match:
                    json_str = dict_match.group(0)

    if json_str is None:
        raise ValueError(f"응답에서 JSON 블록을 찾지 못했습니다. (응답 시작: {response_text[:150]}...)")
    
    try:
        # (디버깅) 추출된 문자열 로깅
        # print(f"--- [Parser DEBUG] Extracted JSON String: {json_str[:200]}... ---")
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 파싱에 실패했습니다: {e}. (추출된 문자열: {json_str[:150]}...)")