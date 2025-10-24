import pandas as pd
import os
import traceback
from pathlib import Path
import time

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# --- 1. 축제 데이터 로더 ---
def _load_and_process_festivals_for_indexing():
    """
    1. 날짜 처리, dropna, 컬럼명 변경 로직을 모두 제거합니다.
    2. 원본 CSV를 그대로 읽고 NaN을 빈 문자열("")로 변환합니다.
       (Filtering 단계에서 모든 원본 컬럼을 metadata로 사용하기 위함)
    """
    print("--- [Indexer] 'festival_df.csv' 로딩 및 전처리 시작... ---")
    try:
        project_root = Path(__file__).resolve().parent
        file_path = project_root / 'data' / 'festival_df.csv'
        if not file_path.exists():
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {file_path}")
        
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError("'festival_df.csv' 파일에 데이터가 없습니다.")

        # 모든 NaN 값을 빈 문자열로 대체 (데이터 유실 방지)
        df = df.fillna("")
        
        print(f"--- [Indexer] 'festival_df.csv' 로딩 성공. {len(df)}개 축제 발견 ---")
        return df.to_dict('records')
        
    except Exception as e:
        print(f"--- [Indexer CRITICAL] 'festival_df.csv' 로딩 실패: {e}\n{traceback.format_exc()} ---")
        return None

# --- 2. 임베딩 모델 준비 (유지) ---
def get_embeddings_model():
    print("--- [Indexer] HuggingFace 임베딩 모델 로딩 시작... ---")
    model_name = "dragonkue/BGE-m3-ko"
    model_kwargs = {'device': 'cpu'} 
    encode_kwargs = {'normalize_embeddings': True} 
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print("--- [Indexer] HuggingFace 임베딩 모델 로딩 완료 ---")
    return embeddings

# --- 3. 벡터 스토어 구축 및 저장  ---
def build_and_save_vector_store():
    start_time = time.time()
    
    # 1. 축제 데이터 로드 (유지)
    festivals = _load_and_process_festivals_for_indexing()
    if not festivals:
        print("--- [Indexer ERROR] 축제 데이터가 없어 인덱싱을 중단합니다.")
        return

    # 2. 임베딩 모델 로드 (유지)
    embeddings = get_embeddings_model()

    # 3. LangChain Document 객체 생성
    documents = []
    print("--- [Indexer] 축제 정보 -> 문서(Document) 변환 시작 ---")
    for festival in festivals:
        
        # 1번 점수(임베딩)에 '축제명'을 다시 추가합니다. (지적해주신 사항 반영)
        content = (
            f"축제명: {festival.get('축제명', '')}\n"  # <-- 이 부분 추가
            f"축제 키워드: {festival.get('키워드', '')}\n"
            f"축제 소개: {festival.get('소개', '')}"
        )
        
        # 2번 점수(동적)에 사용될 메타데이터 (가장 중요)
        # (축제명, 주요고객층, 인기도_점수 등 모든 원본 컬럼이 포함됨)
        metadata = festival
        
        documents.append(Document(page_content=content, metadata=metadata))
    
    print(f"--- [Indexer] 문서 변환 완료. 총 {len(documents)}개 문서 생성 ---")

    # 4. FAISS 벡터 스토어 생성 (유지)
    print("--- [Indexer] FAISS 벡터 스토어 생성 시작 (시간이 걸릴 수 있습니다)... ---")
    vector_store = FAISS.from_documents(documents, embeddings)
    print("--- [Indexer] FAISS 벡터 스토어 생성 완료 ---")

    # 5. 로컬에 저장 (유지)
    project_root = Path(__file__).resolve().parent
    save_path = project_root / 'faiss_festival'
    
    os.makedirs(save_path.parent, exist_ok=True)
    vector_store.save_local(str(save_path))
    
    end_time = time.time()
    print("=" * 50)
    print(f"🎉 성공! FAISS 벡터 스토어를 생성하여 '{save_path}'에 저장했습니다.")
    print(f"총 소요 시간: {end_time - start_time:.2f}초")
    print("=" * 50)

if __name__ == "__main__":
    build_and_save_vector_store()