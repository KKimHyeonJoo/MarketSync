# modules/visualization.py

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import streamlit as st

import config
from modules.profile_utils import get_chat_profile_dict

logger = config.get_logger(__name__)

def set_korean_font():
    """
    시스템에 설치된 한글 폰트를 찾아 Matplotlib에 설정합니다.
    """
    font_list = ['Malgun Gothic', 'AppleGothic', 'NanumGothic']
    
    found_font = False
    for font_name in font_list:
        if any(font.name == font_name for font in font_manager.fontManager.ttflist):
            plt.rc('font', family=font_name)
            logger.info(f"✅ 한글 폰트 '{font_name}'을(를) 찾아 그래프에 적용합니다.")
            found_font = True
            break
            
    if not found_font:
        logger.warning("⚠️ 경고: Malgun Gothic, AppleGothic, NanumGothic 폰트를 찾을 수 없습니다.")
    
    plt.rcParams['axes.unicode_minus'] = False


def display_merchant_profile(profile_data: dict):
    set_korean_font()

    """
    분석된 가맹점 프로필 전체를 Streamlit 화면에 시각화합니다.
    """
    if not profile_data or "store_profile" not in profile_data:
        st.error("분석할 가맹점 데이터가 없습니다.")
        return

    store_data = profile_data["store_profile"]
    store_name = store_data.get('가맹점명', '선택 매장')

    st.info(f"**'{store_name}'**의 상세 분석 결과입니다.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 기본 정보", 
        "🧑‍🤝‍🧑 주요 고객층 (성별/연령대)", 
        "🚶 주요 고객 유형 (상권)", 
        "🔁 고객 충성도 (신규/재방문)"
    ])

    with tab1:
        render_basic_info_table(store_data)

    with tab2:
        st.subheader("🧑‍🤝‍🧑 주요 고객층 분포 (성별/연령대)")
        fig2 = plot_customer_distribution(store_data)
        st.pyplot(fig2)

    with tab3:
        st.subheader("🚶 주요 고객 유형 (상권)")
        fig3 = plot_customer_type_pie(store_data)
        st.pyplot(fig3)

    with tab4:
        st.subheader("🔁 신규 vs 재방문 고객 비율") 
        fig4 = plot_loyalty_donut(store_data)
        st.pyplot(fig4)


def get_main_customer_segment(store_data):
    """주요 고객층(성별/연령대) 텍스트를 반환합니다."""
    segments = {
        '남성 20대 이하': store_data.get('남성20대이하비율', 0),
        '남성 30대': store_data.get('남성30대비율', 0),
        '남성 40대': store_data.get('남성40대비율', 0),
        '남성 50대 이상': store_data.get('남성50대비율', 0) + store_data.get('남성60대이상비율', 0),
        '여성 20대 이하': store_data.get('여성20대이하비율', 0),
        '여성 30대': store_data.get('여성30대비율', 0),
        '여성 40대': store_data.get('여성40대비율', 0),
        '여성 50대 이상': store_data.get('여성50대비율', 0) + store_data.get('여성60대이상비율', 0)
    }
    
    if not any(segments.values()):
        return None
    
    max_segment = max(segments, key=segments.get)
    max_value = segments[max_segment]
    
    if max_value == 0:
        return None
        
    return f"'{max_segment}({max_value:.1f}%)'"


def render_basic_info_table(store_data):
    """(Tab 1) 기본 정보 요약 표와 텍스트를 렌더링합니다."""
    
    summary_data = get_chat_profile_dict(store_data)
    
    st.subheader("📋 가맹점 기본 정보")
    summary_df = pd.DataFrame(summary_data.items(), columns=["항목", "내용"])
    summary_df = summary_df[summary_df['항목'] != '자동추출특징']
    summary_df = summary_df.astype(str) 
    st.table(summary_df.set_index('항목'))

    st.subheader("📌 분석 요약")
    st.write(f"✅ **{summary_data.get('가맹점명', 'N/A')}**은(는) '{summary_data.get('상권', 'N/A')}' 상권의 '{summary_data.get('업종', 'N/A')}' 업종 가맹점입니다.")
    st.write(f"📈 매출 수준은 **{summary_data.get('매출 수준', 'N/A')}**이며, 동일 상권 내 매출 순위는 **{summary_data.get('동일 상권 대비 매출 순위', 'N/A')}**입니다.")
    st.write(f"💰 방문 고객수는 **{summary_data.get('방문 고객수 수준', 'N/A')}** 수준이며, 객단가는 **{summary_data.get('객단가 수준', 'N/A')}** 수준입니다.")
    
    main_customer = get_main_customer_segment(store_data)
    if main_customer:
        st.write(f"👥 주요 고객층은 **{main_customer}**이(가) 가장 많습니다.")


def plot_customer_distribution(store_data):
    """(Tab 2) 고객 특성 분포 (성별/연령대)를 보여주는 막대 그래프를 생성합니다."""
    labels = ['20대 이하', '30대', '40대', '50대 이상']
    male_percents = [
        store_data.get('남성20대이하비율', 0), store_data.get('남성30대비율', 0),
        store_data.get('남성40대비율', 0),
        store_data.get('남성50대비율', 0) + store_data.get('남성60대이상비율', 0)
    ]
    female_percents = [
        store_data.get('여성20대이하비율', 0), store_data.get('여성30대비율', 0),
        store_data.get('여성40대비율', 0),
        store_data.get('여성50대비율', 0) + store_data.get('여성60대이상비율', 0)
    ]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, male_percents, width, label='남성', color='cornflowerblue')
    rects2 = ax.bar(x + width/2, female_percents, width, label='여성', color='salmon')

    ax.set_ylabel('고객 비율 (%)')
    ax.set_title('주요 고객층 분포 (성별/연령대)', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    ax.bar_label(rects1, padding=3, fmt='%.1f')
    ax.bar_label(rects2, padding=3, fmt='%.1f')
    
    fig.tight_layout()
    return fig


def plot_customer_type_pie(store_data):
    """(Tab 3) 주요 고객 유형 (거주자, 직장인, 유동인구)을 파이 차트로 생성합니다."""
    
    customer_data = {
        '유동인구': store_data.get("유동인구이용비율", 0),
        '거주자': store_data.get("거주자이용비율", 0),
        '직장인': store_data.get("직장인이용비율", 0)
    }
    
    filtered_data = {label: (size or 0) for label, size in customer_data.items()}
    filtered_data = {label: size for label, size in filtered_data.items() if size > 0}
    
    sizes = list(filtered_data.values())
    labels = list(filtered_data.keys())

    if not sizes or sum(sizes) == 0:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.text(0.5, 0.5, "데이터 없음", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title("주요 고객 유형", fontsize=13)
        return fig

    pie_labels = [f"{label} ({size:.1f}%)" for label, size in zip(labels, sizes)]

    fig, ax = plt.subplots(figsize=(6, 6))
    
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=pie_labels,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.8
    )
    
    plt.setp(autotexts, size=9, weight="bold", color="white")
    ax.set_title("주요 고객 유형", fontsize=13)
    ax.axis('equal')
    
    return fig


def plot_loyalty_donut(store_data):
    """(Tab 4) 신규 vs 재방문 고객 비율을 도넛 차트로 생성합니다."""
    
    visit_ratio = {
        '신규 고객': store_data.get('신규고객비율') or 0,
        '재이용 고객': store_data.get('재이용고객비율') or 0
    }
    
    sizes = list(visit_ratio.values())
    labels = list(visit_ratio.keys())
    
    if not sizes or sum(sizes) == 0:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.text(0.5, 0.5, "데이터 없음", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title("신규 vs 재방문 고객 비율")
        return fig

    fig, ax = plt.subplots(figsize=(5, 5))
    
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.85,
        colors=['lightcoral', 'skyblue']
    )
    
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax.add_artist(centre_circle)
    
    plt.setp(autotexts, size=10, weight="bold")
    ax.set_title("신규 vs 재방문 고객 비율", fontsize=14)
    ax.axis('equal')
    
    return fig