# create_marketing_retriever.py
# -*- coding: utf-8 -*-

import os
import time
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# [변경] Google 대신 HuggingFace 임베딩을 가져옵니다.
from langchain_community.embeddings import HuggingFaceEmbeddings
# [삭제] from dotenv import load_dotenv (더 이상 필요 없음)

def create_and_save_retriever():
    """
    (수정됨) 로컬 Hugging Face 임베딩 모델('dragonkue/BGE-m3-ko')을 사용하여
    마케팅 PDF 문서로부터 Retriever를 생성하고 파일로 저장합니다.
    """
    try:
        # 0. [변경] API 키 로딩 로직 삭제
        print("✅ 로컬 임베딩 모델을 사용합니다. (API 키 필요 없음)")

        # 1. 데이터 로드
        loader = DirectoryLoader(
            './marketing',
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True
        )
        documents = loader.load()
        print(f"✅ 총 {len(documents)}개의 PDF 문서를 불러왔습니다.")

        if not documents:
            raise ValueError("🚨 'marketing' 폴더에 PDF 파일이 없습니다. 문서를 추가해주세요.")

        # 2. 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        docs = text_splitter.split_documents(documents)
        print(f"✅ 문서를 총 {len(docs)}개의 청크(chunk)로 분할했습니다.")

        if not docs:
            raise ValueError("🚨 문서를 청크로 분할하는 데 실패했습니다.")

        # 3. [변경] 임베딩 모델 설정
        print(f"✅ 임베딩 모델 'dragonkue/BGE-m3-ko' 로드를 시작합니다...")
        
        model_name = "dragonkue/BGE-m3-ko"
        # 💡 [참고] 로컬 PC/서버에 GPU가 있다면 {'device': 'cuda'}로 변경하세요.
        model_kwargs = {'device': 'cpu'} 
        # 💡 [중요] BGE 모델은 검색 성능을 위해 정규화(normalize)를 강력히 권장합니다.
        encode_kwargs = {'normalize_embeddings': True}

        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        print(f"✅ 임베딩 모델을 성공적으로 로드했습니다.")

        # 4. Vector Store 생성 (FAISS)
        vectorstore = None
        
        # [변경] 로컬 모델은 배치 처리가 매우 빠르므로, API 제한(time.sleep)이 필요 없습니다.
        # [참고] BGE-m3-ko 모델은 배치 처리를 지원하므로, FAISS.from_documents가 내부적으로 효율적으로 처리합니다.
        print(f"🔄 총 {len(docs)}개의 청크에 대한 임베딩을 시작합니다. (시간이 걸릴 수 있음)")
        
        vectorstore = FAISS.from_documents(docs, embeddings)

        # 5. [변경] 로컬 저장 (경로는 동일)
        save_dir = './retriever/marketing_retriever' # [경로 수정] knowledge_base.py와 맞춤
        os.makedirs(save_dir, exist_ok=True)

        vectorstore.save_local(save_dir)

        print(f"🎉 Retriever가 성공적으로 생성되어 '{save_dir}' 폴더에 저장되었습니다!")

    except Exception as e:
        print(f"🚨🚨 치명적인 오류 발생 🚨🚨: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # 1. 필요한 라이브러리가 설치되었는지 확인
    try:
        import langchain_community
        import sentence_transformers
        import faiss
        import torch
    except ImportError as e:
        print(f"🚨 [오류] {e.name} 라이브러리가 설치되지 않았습니다.")
        print("👉 다음 명령어를 실행하여 필요한 라이브러리를 설치해주세요:")
        print("pip install langchain-community sentence-transformers faiss-cpu torch")
        print("(GPU 사용 시: pip install langchain-community sentence-transformers faiss-gpu torch)")
        exit(1)
        
    create_and_save_retriever()