# modules/knowledge_base.py

import os
import streamlit as st
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings 
import traceback 

import config 

logger = config.get_logger(__name__)

@st.cache_resource
def _load_embedding_model():
    """
    임베딩 모델을 별도 함수로 분리하여 캐싱 (FAISS 로드 시 재사용)
    """
    try:
        logger.info("--- [Cache] HuggingFace 임베딩 모델 최초 로딩 시작 ---")

        model_name = config.EMBEDDING_MODEL
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}

        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        logger.info(f"--- [Cache] HuggingFace 임베딩 모델 ({model_name}) 로딩 성공 ---")
        return embeddings
    except Exception as e:
        logger.critical(f"--- [CRITICAL ERROR] 임베딩 모델 로딩 실패: {e} ---", exc_info=True)
        st.error(f"임베딩 모델('{config.EMBEDDING_MODEL}') 로딩 중 심각한 오류가 발생했습니다: {e}")
        return None

@st.cache_resource
def load_marketing_vectorstore():
    """
    '마케팅 전략' FAISS Vector Store를 로드하여 Retriever를 생성합니다.
    """
    try:
        logger.info("--- [Cache] '마케팅' FAISS Vector Store 최초 로딩 시작 ---")
        embeddings = _load_embedding_model()
        
        if embeddings is None:
            raise RuntimeError("임베딩 모델 로딩에 실패하여 Retriever를 생성할 수 없습니다.")

        vector_db_path = config.PATH_FAISS_MARKETING
        
        if not vector_db_path.exists():
            logger.critical(f"--- [CRITICAL ERROR] '마케팅' Vector DB 경로를 찾을 수 없습니다: {vector_db_path}")
            st.error(f"'마케팅' Vector DB 파일을 찾을 수 없습니다. (경로: {vector_db_path})")
            return None
        
        db = FAISS.load_local(
            folder_path=str(vector_db_path), 
            embeddings=embeddings, 
            allow_dangerous_deserialization=True 
        )
        
        retriever = db.as_retriever(search_kwargs={"k": 2})
        
        logger.info("--- [Cache] '마케팅' FAISS Vector Store 로딩 성공 ---")
        return retriever
        
    except Exception as e:
        logger.critical(f"--- [CRITICAL ERROR] '마케팅' FAISS 로딩 실패: {e} ---", exc_info=True)
        st.error(f"'마케팅' Vector Store 로딩 중 오류 발생: {e}")
        return None

@st.cache_resource
def load_festival_vectorstore():
    """
    '축제 정보' FAISS Vector Store를 로드합니다.
    """
    try:
        logger.info("--- [Cache] '축제' FAISS Vector Store 최초 로딩 시작 ---")
        embeddings = _load_embedding_model() 

        if embeddings is None:
            raise RuntimeError("임베딩 모델 로딩에 실패하여 '축제' Vector Store를 로드할 수 없습니다.")

        vector_db_path = config.PATH_FAISS_FESTIVAL

        if not vector_db_path.exists():
            logger.critical(f"--- [CRITICAL ERROR] '축제' Vector DB 경로를 찾을 수 없습니다: {vector_db_path}")
            st.error(f"'축제' Vector DB 파일을 찾을 수 없습니다. (경로: {vector_db_path})")
            return None

        db = FAISS.load_local(
            folder_path=str(vector_db_path),
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info("--- [Cache] '축제' FAISS Vector Store 로딩 성공 ---")
        return db

    except Exception as e:
        logger.critical(f"--- [CRITICAL ERROR] '축제' FAISS 로딩 실패: {e} ---", exc_info=True)
        st.error(f"'축제' Vector Store 로딩 중 오류 발생: {e}")
        return None