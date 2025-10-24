<<<<<<< HEAD
# final_df.py 

import pandas as pd
import numpy as np
import os
import sys

# 스크립트 파일이 위치한 디렉토리를 기준으로 경로 설정
script_path = os.path.abspath(sys.argv[0]) 
script_dir = os.path.dirname(script_path)

# 데이터 폴더 경로
data_dir = os.path.join(script_dir, 'data')

# 'data' 폴더가 실제로 존재하는지 확인 (안전장치)
if not os.path.exists(data_dir):
    print(f"Error: Data directory not found at {data_dir}. Please check your folder structure.")
    sys.exit(1)

# 파일 경로 함수
def get_file_path(filename):
    """data 폴더 내의 파일 경로를 반환합니다."""
    return os.path.join(data_dir, filename)

# --------------------------------------------------------------------------
#### 1) 데이터 1 - **가맹점 개요정보**
# --------------------------------------------------------------------------

file_path1 = get_file_path('big_data_set1_f.csv')

try:
    df1 = pd.read_csv(file_path1, encoding="cp949")
except FileNotFoundError:
    print(f"Error: File not found at {file_path1}. Please ensure big_data_set1_f.csv is in the 'data' folder.")
    sys.exit(1)

col_map1 = {
    "ENCODED_MCT": "가맹점ID",
    "MCT_BSE_AR": "가맹점주소",
    "MCT_NM": "가맹점명",
    "MCT_BRD_NUM": "브랜드구분코드",
    "MCT_SIGUNGU_NM": "지역명",
    "HPSN_MCT_ZCD_NM": "업종",
    "HPSN_MCT_BZN_CD_NM": "상권",
    "ARE_D": "개설일",
    "MCT_ME_D": "폐업여부"
}

df1 = df1.rename(columns=col_map1)

# - 전처리
df1['브랜드구분코드'] = df1['브랜드구분코드'].fillna('미확인')
df1['상권'] = df1['상권'].fillna('미확인')

df1['개설일'] = df1['개설일'].astype(str)
# errors='coerce' 추가: ipynb 코드에는 없지만, 안전한 datetime 변환을 위해 유지 (원본 py 코드 유지)
df1['개설일'] = pd.to_datetime(df1['개설일'], format='%Y%m%d', errors='coerce') 

# ipynb 파일에서는 errors='coerce'가 없었지만, int 변환 시 오류 방지를 위해 원본 py 코드의 안전 로직을 따름.
df1['폐업여부'] = df1['폐업여부'].apply(lambda x: pd.to_datetime(int(x), format='%Y%m%d', errors='coerce') if pd.notna(x) and str(x).isdigit() else pd.NaT)
df1['운영상태'] = df1['폐업여부'].apply(lambda x: '운영중' if pd.isna(x) else '폐업')

# --------------------------------------------------------------------------
#### 2) 데이터 2 - **가맹점 월별 이용정보**
# --------------------------------------------------------------------------

file_path2 = get_file_path('big_data_set2_f.csv')

try:
    df2 = pd.read_csv(file_path2, encoding="cp949")
except FileNotFoundError:
    print(f"Error: File not found at {file_path2}. Please ensure big_data_set2_f.csv is in the 'data' folder.")
    sys.exit(1)

col_map2 = {
    "ENCODED_MCT": "가맹점ID",
    "TA_YM": "기준년월",
    "MCT_OPE_MS_CN": "운영개월수_구간",
    "RC_M1_SAA": "월매출금액_구간",
    "RC_M1_TO_UE_CT": "월매출건수_구간",
    "RC_M1_UE_CUS_CN": "월유니크고객수_구간",
    "RC_M1_AV_NP_AT": "월객단가_구간",
    "APV_CE_RAT": "취소율_구간",
    "DLV_SAA_RAT": "배달매출비율",
    "M1_SME_RY_SAA_RAT": "동일업종매출대비비율",
    "M1_SME_RY_CNT_RAT": "동일업종건수대비비율",
    "M12_SME_RY_SAA_PCE_RT": "동일업종내매출순위비율",
    "M12_SME_BZN_SAA_PCE_RT": "동일상권내매출순위비율",
    "M12_SME_RY_ME_MCT_RAT": "동일업종해지가맹점비중",
    "M12_SME_BZN_ME_MCT_RAT": "동일상권해지가맹점비중"
}

df2 = df2.rename(columns=col_map2)

# - 전처리
df2['기준년월'] = pd.to_datetime(df2['기준년월'].astype(str), format='%Y%m')

df2.replace(-999999.9, np.nan, inplace=True)

# --------------------------------------------------------------------------
#### 3) 데이터 3 - **가맹점 월별 이용 고객정보**
# --------------------------------------------------------------------------

file_path3 = get_file_path('big_data_set3_f.csv')

try:
    df3 = pd.read_csv(file_path3, encoding="cp949")
except FileNotFoundError:
    print(f"Error: File not found at {file_path3}. Please ensure big_data_set3_f.csv is in the 'data' folder.")
    sys.exit(1)

col_map3 = {
    "ENCODED_MCT": "가맹점ID",
    "TA_YM": "기준년월",
    "M12_MAL_1020_RAT": "남성20대이하비율",
    "M12_MAL_30_RAT": "남성30대비율",
    "M12_MAL_40_RAT": "남성40대비율",
    "M12_MAL_50_RAT": "남성50대비율",
    "M12_MAL_60_RAT": "남성60대이상비율",
    "M12_FME_1020_RAT": "여성20대이하비율",
    "M12_FME_30_RAT": "여성30대비율",
    "M12_FME_40_RAT": "여성40대비율",
    "M12_FME_50_RAT": "여성50대비율",
    "M12_FME_60_RAT": "여성60대이상비율",
    "MCT_UE_CLN_REU_RAT": "재이용고객비율",
    "MCT_UE_CLN_NEW_RAT": "신규고객비율",
    "RC_M1_SHC_RSD_UE_CLN_RAT": "거주자이용비율",
    "RC_M1_SHC_WP_UE_CLN_RAT": "직장인이용비율",
    "RC_M1_SHC_FLP_UE_CLN_RAT": "유동인구이용비율"
}

df3 = df3.rename(columns=col_map3)

# - 전처리
df3['기준년월'] = pd.to_datetime(df3['기준년월'].astype(str), format='%Y%m')

df3.replace(-999999.9, np.nan, inplace=True)

# --------------------------------------------------------------------------
#### 데이터 통합
# --------------------------------------------------------------------------

df23 = pd.merge(df2, df3, on=["가맹점ID", "기준년월"], how="inner")

final_df = pd.merge(df23, df1, on="가맹점ID", how="left")

# --------------------------------------------------------------------------
#### 이상값 처리
# --------------------------------------------------------------------------

non_seongdong_areas = [
    '압구정로데오', '풍산지구', '미아사거리', '방배역',
    '자양', '동대문역사문화공원역', '건대입구',
    '서면역', '오남'
]

# Step 1️⃣ 주소가 '성동구'에 포함된 데이터만 남기기
mask_seongdong_addr = final_df['가맹점주소'].str.contains('성동구', na=False)
seongdong_df = final_df[mask_seongdong_addr].copy()

# Step 2️⃣ 상권명이 성동구 외인데 주소는 성동구인 경우 → 라벨 교정
mask_mislabel = seongdong_df['상권'].isin(non_seongdong_areas)
seongdong_df.loc[mask_mislabel, '상권'] = '미확인(성동구)'

# Step 3️⃣ (ipynb 코드 로직 적용) 상권명이 '미확인'인데 주소가 성동구가 아닌 경우 제거
# 주피터 노트북의 로직을 그대로 반영합니다. (실제 필터링 효과는 없지만, 코드 일치성 확보)
final_clean_df = seongdong_df[
    ~(
        (seongdong_df['상권'].str.contains('미확인')) &
        (~seongdong_df['가맹점주소'].str.contains('성동구', na=False))
    )
].copy()

# 업종 - 한 업종이 100퍼인 경우 제외(이상치 취급)
final_clean_df = final_clean_df[final_clean_df['업종'] != '유제품'].copy()

# 월매출금액_구간 컬럼의 고유값
unique_sales_bins = final_clean_df['월매출금액_구간'].dropna().unique()

# 매출구간수준 매핑 딕셔너리 정의
# --------------------------------------------------------------------------
# ❇️ [수정] 구간 -> 수준 변환 (컬럼별 다른 명칭 적용)
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# ❇️ [수정] 구간 -> 수준 변환 (모든 컬럼 적용)
# --------------------------------------------------------------------------

# 1. '월매출금액_구간' (규모/순위 기준)
sales_volume_map = {
    '1_10%이하': '최상위',  
    '2_10-25%': '상위',    
    '3_25-50%': '중상위',  
    '4_50-75%': '중하위',  
    '5_75-90%': '하위',    
    '6_90%초과(하위 10% 이하)': '최하위' 
}

# 2. '월객단가_구간' (가격대 기준)
price_level_map = {
    '1_10%이하': '최고가',  
    '2_10-25%': '고가',    
    '3_25-50%': '중가',    
    '4_50-75%': '중저가',  
    '5_75-90%': '저가',    
    '6_90%초과(하위 10% 이하)': '최저가' 
}

# 3. '운영개월수_구간' (경험/연차 기준)
operation_period_map = {
    '1_10%이하': '최장기',  # 가장 오래 운영
    '2_10-25%': '장기',
    '3_25-50%': '중기',
    '4_50-75%': '단기',
    '5_75-90%': '신규',
    '6_90%초과(하위 10% 이하)': '최신규' # 가장 최근 개업
}

# 4. '월매출건수_구간' (거래량/빈도 기준)
transaction_count_map = {
    '1_10%이하': '거래 최다', # 거래가 가장 많음
    '2_10-25%': '거래 많음',
    '3_25-50%': '거래 보통',
    '4_50-75%': '거래 적음',
    '5_75-90%': '거래 희소',
    '6_90%초과(하위 10% 이하)': '거래 최저' 
}

# 5. '월유니크고객수_구간' (고객 규모 기준)
customer_count_map = {
    '1_10%이하': '고객 최다', # 고객 수가 가장 많음
    '2_10-25%': '고객 많음',
    '3_25-50%': '고객 보통',
    '4_50-75%': '고객 적음',
    '5_75-90%': '고객 희소',
    '6_90%초과(하위 10% 이하)': '고객 최저'
}


# --- 새 컬럼 생성 ---

final_clean_df['매출구간_수준'] = final_clean_df['월매출금액_구간'].map(sales_volume_map)
final_clean_df['월객단가_수준'] = final_clean_df['월객단가_구간'].map(price_level_map)
final_clean_df['운영개월수_수준'] = final_clean_df['운영개월수_구간'].map(operation_period_map)
final_clean_df['월매출건수_수준'] = final_clean_df['월매출건수_구간'].map(transaction_count_map)
final_clean_df['월유니크고객수_수준'] = final_clean_df['월유니크고객수_구간'].map(customer_count_map)

# --- 미확인 값 처리 ---
final_clean_df['매출구간_수준'] = final_clean_df['매출구간_수준'].fillna('미확인')
final_clean_df['월객단가_수준'] = final_clean_df['월객단가_수준'].fillna('미확인')
final_clean_df['운영개월수_수준'] = final_clean_df['운영개월수_수준'].fillna('미확인')
final_clean_df['월매출건수_수준'] = final_clean_df['월매출건수_수준'].fillna('미확인')
final_clean_df['월유니크고객수_수준'] = final_clean_df['월유니크고객수_수준'].fillna('미확인')
# --------------------------------------------------------------------------
# final_df 저장
# --------------------------------------------------------------------------

# 'data' 폴더 내에 저장
save_path = get_file_path("final_df.csv")

# CSV 파일 저장 (인덱스 제외)
final_clean_df.to_csv(save_path, index=False, encoding="utf-8-sig")

=======
# final_df.py 

import pandas as pd
import numpy as np
import os
import sys

# 스크립트 파일이 위치한 디렉토리를 기준으로 경로 설정
script_path = os.path.abspath(sys.argv[0]) 
script_dir = os.path.dirname(script_path)

# 데이터 폴더 경로
data_dir = os.path.join(script_dir, 'data')

# 'data' 폴더가 실제로 존재하는지 확인 (안전장치)
if not os.path.exists(data_dir):
    print(f"Error: Data directory not found at {data_dir}. Please check your folder structure.")
    sys.exit(1)

# 파일 경로 함수
def get_file_path(filename):
    """data 폴더 내의 파일 경로를 반환합니다."""
    return os.path.join(data_dir, filename)

# --------------------------------------------------------------------------
#### 1) 데이터 1 - **가맹점 개요정보**
# --------------------------------------------------------------------------

file_path1 = get_file_path('big_data_set1_f.csv')

try:
    df1 = pd.read_csv(file_path1, encoding="cp949")
except FileNotFoundError:
    print(f"Error: File not found at {file_path1}. Please ensure big_data_set1_f.csv is in the 'data' folder.")
    sys.exit(1)

col_map1 = {
    "ENCODED_MCT": "가맹점ID",
    "MCT_BSE_AR": "가맹점주소",
    "MCT_NM": "가맹점명",
    "MCT_BRD_NUM": "브랜드구분코드",
    "MCT_SIGUNGU_NM": "지역명",
    "HPSN_MCT_ZCD_NM": "업종",
    "HPSN_MCT_BZN_CD_NM": "상권",
    "ARE_D": "개설일",
    "MCT_ME_D": "폐업여부"
}

df1 = df1.rename(columns=col_map1)

# - 전처리
df1['브랜드구분코드'] = df1['브랜드구분코드'].fillna('미확인')
df1['상권'] = df1['상권'].fillna('미확인')

df1['개설일'] = df1['개설일'].astype(str)
# errors='coerce' 추가: ipynb 코드에는 없지만, 안전한 datetime 변환을 위해 유지 (원본 py 코드 유지)
df1['개설일'] = pd.to_datetime(df1['개설일'], format='%Y%m%d', errors='coerce') 

# ipynb 파일에서는 errors='coerce'가 없었지만, int 변환 시 오류 방지를 위해 원본 py 코드의 안전 로직을 따름.
df1['폐업여부'] = df1['폐업여부'].apply(lambda x: pd.to_datetime(int(x), format='%Y%m%d', errors='coerce') if pd.notna(x) and str(x).isdigit() else pd.NaT)
df1['운영상태'] = df1['폐업여부'].apply(lambda x: '운영중' if pd.isna(x) else '폐업')

# --------------------------------------------------------------------------
#### 2) 데이터 2 - **가맹점 월별 이용정보**
# --------------------------------------------------------------------------

file_path2 = get_file_path('big_data_set2_f.csv')

try:
    df2 = pd.read_csv(file_path2, encoding="cp949")
except FileNotFoundError:
    print(f"Error: File not found at {file_path2}. Please ensure big_data_set2_f.csv is in the 'data' folder.")
    sys.exit(1)

col_map2 = {
    "ENCODED_MCT": "가맹점ID",
    "TA_YM": "기준년월",
    "MCT_OPE_MS_CN": "운영개월수_구간",
    "RC_M1_SAA": "월매출금액_구간",
    "RC_M1_TO_UE_CT": "월매출건수_구간",
    "RC_M1_UE_CUS_CN": "월유니크고객수_구간",
    "RC_M1_AV_NP_AT": "월객단가_구간",
    "APV_CE_RAT": "취소율_구간",
    "DLV_SAA_RAT": "배달매출비율",
    "M1_SME_RY_SAA_RAT": "동일업종매출대비비율",
    "M1_SME_RY_CNT_RAT": "동일업종건수대비비율",
    "M12_SME_RY_SAA_PCE_RT": "동일업종내매출순위비율",
    "M12_SME_BZN_SAA_PCE_RT": "동일상권내매출순위비율",
    "M12_SME_RY_ME_MCT_RAT": "동일업종해지가맹점비중",
    "M12_SME_BZN_ME_MCT_RAT": "동일상권해지가맹점비중"
}

df2 = df2.rename(columns=col_map2)

# - 전처리
df2['기준년월'] = pd.to_datetime(df2['기준년월'].astype(str), format='%Y%m')

df2.replace(-999999.9, np.nan, inplace=True)

# --------------------------------------------------------------------------
#### 3) 데이터 3 - **가맹점 월별 이용 고객정보**
# --------------------------------------------------------------------------

file_path3 = get_file_path('big_data_set3_f.csv')

try:
    df3 = pd.read_csv(file_path3, encoding="cp949")
except FileNotFoundError:
    print(f"Error: File not found at {file_path3}. Please ensure big_data_set3_f.csv is in the 'data' folder.")
    sys.exit(1)

col_map3 = {
    "ENCODED_MCT": "가맹점ID",
    "TA_YM": "기준년월",
    "M12_MAL_1020_RAT": "남성20대이하비율",
    "M12_MAL_30_RAT": "남성30대비율",
    "M12_MAL_40_RAT": "남성40대비율",
    "M12_MAL_50_RAT": "남성50대비율",
    "M12_MAL_60_RAT": "남성60대이상비율",
    "M12_FME_1020_RAT": "여성20대이하비율",
    "M12_FME_30_RAT": "여성30대비율",
    "M12_FME_40_RAT": "여성40대비율",
    "M12_FME_50_RAT": "여성50대비율",
    "M12_FME_60_RAT": "여성60대이상비율",
    "MCT_UE_CLN_REU_RAT": "재이용고객비율",
    "MCT_UE_CLN_NEW_RAT": "신규고객비율",
    "RC_M1_SHC_RSD_UE_CLN_RAT": "거주자이용비율",
    "RC_M1_SHC_WP_UE_CLN_RAT": "직장인이용비율",
    "RC_M1_SHC_FLP_UE_CLN_RAT": "유동인구이용비율"
}

df3 = df3.rename(columns=col_map3)

# - 전처리
df3['기준년월'] = pd.to_datetime(df3['기준년월'].astype(str), format='%Y%m')

df3.replace(-999999.9, np.nan, inplace=True)

# --------------------------------------------------------------------------
#### 데이터 통합
# --------------------------------------------------------------------------

df23 = pd.merge(df2, df3, on=["가맹점ID", "기준년월"], how="inner")

final_df = pd.merge(df23, df1, on="가맹점ID", how="left")

# --------------------------------------------------------------------------
#### 이상값 처리
# --------------------------------------------------------------------------

non_seongdong_areas = [
    '압구정로데오', '풍산지구', '미아사거리', '방배역',
    '자양', '동대문역사문화공원역', '건대입구',
    '서면역', '오남'
]

# Step 1️⃣ 주소가 '성동구'에 포함된 데이터만 남기기
mask_seongdong_addr = final_df['가맹점주소'].str.contains('성동구', na=False)
seongdong_df = final_df[mask_seongdong_addr].copy()

# Step 2️⃣ 상권명이 성동구 외인데 주소는 성동구인 경우 → 라벨 교정
mask_mislabel = seongdong_df['상권'].isin(non_seongdong_areas)
seongdong_df.loc[mask_mislabel, '상권'] = '미확인(성동구)'

# Step 3️⃣ (ipynb 코드 로직 적용) 상권명이 '미확인'인데 주소가 성동구가 아닌 경우 제거
# 주피터 노트북의 로직을 그대로 반영합니다. (실제 필터링 효과는 없지만, 코드 일치성 확보)
final_clean_df = seongdong_df[
    ~(
        (seongdong_df['상권'].str.contains('미확인')) &
        (~seongdong_df['가맹점주소'].str.contains('성동구', na=False))
    )
].copy()

# 업종 - 한 업종이 100퍼인 경우 제외(이상치 취급)
final_clean_df = final_clean_df[final_clean_df['업종'] != '유제품'].copy()

# 월매출금액_구간 컬럼의 고유값
unique_sales_bins = final_clean_df['월매출금액_구간'].dropna().unique()

# 매출구간수준 매핑 딕셔너리 정의
# --------------------------------------------------------------------------
# ❇️ [수정] 구간 -> 수준 변환 (컬럼별 다른 명칭 적용)
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# ❇️ [수정] 구간 -> 수준 변환 (모든 컬럼 적용)
# --------------------------------------------------------------------------

# 1. '월매출금액_구간' (규모/순위 기준)
sales_volume_map = {
    '1_10%이하': '최상위',  
    '2_10-25%': '상위',    
    '3_25-50%': '중상위',  
    '4_50-75%': '중하위',  
    '5_75-90%': '하위',    
    '6_90%초과(하위 10% 이하)': '최하위' 
}

# 2. '월객단가_구간' (가격대 기준)
price_level_map = {
    '1_10%이하': '최고가',  
    '2_10-25%': '고가',    
    '3_25-50%': '중가',    
    '4_50-75%': '중저가',  
    '5_75-90%': '저가',    
    '6_90%초과(하위 10% 이하)': '최저가' 
}

# 3. '운영개월수_구간' (경험/연차 기준)
operation_period_map = {
    '1_10%이하': '최장기',  # 가장 오래 운영
    '2_10-25%': '장기',
    '3_25-50%': '중기',
    '4_50-75%': '단기',
    '5_75-90%': '신규',
    '6_90%초과(하위 10% 이하)': '최신규' # 가장 최근 개업
}

# 4. '월매출건수_구간' (거래량/빈도 기준)
transaction_count_map = {
    '1_10%이하': '거래 최다', # 거래가 가장 많음
    '2_10-25%': '거래 많음',
    '3_25-50%': '거래 보통',
    '4_50-75%': '거래 적음',
    '5_75-90%': '거래 희소',
    '6_90%초과(하위 10% 이하)': '거래 최저' 
}

# 5. '월유니크고객수_구간' (고객 규모 기준)
customer_count_map = {
    '1_10%이하': '고객 최다', # 고객 수가 가장 많음
    '2_10-25%': '고객 많음',
    '3_25-50%': '고객 보통',
    '4_50-75%': '고객 적음',
    '5_75-90%': '고객 희소',
    '6_90%초과(하위 10% 이하)': '고객 최저'
}


# --- 새 컬럼 생성 ---

final_clean_df['매출구간_수준'] = final_clean_df['월매출금액_구간'].map(sales_volume_map)
final_clean_df['월객단가_수준'] = final_clean_df['월객단가_구간'].map(price_level_map)
final_clean_df['운영개월수_수준'] = final_clean_df['운영개월수_구간'].map(operation_period_map)
final_clean_df['월매출건수_수준'] = final_clean_df['월매출건수_구간'].map(transaction_count_map)
final_clean_df['월유니크고객수_수준'] = final_clean_df['월유니크고객수_구간'].map(customer_count_map)

# --- 미확인 값 처리 ---
final_clean_df['매출구간_수준'] = final_clean_df['매출구간_수준'].fillna('미확인')
final_clean_df['월객단가_수준'] = final_clean_df['월객단가_수준'].fillna('미확인')
final_clean_df['운영개월수_수준'] = final_clean_df['운영개월수_수준'].fillna('미확인')
final_clean_df['월매출건수_수준'] = final_clean_df['월매출건수_수준'].fillna('미확인')
final_clean_df['월유니크고객수_수준'] = final_clean_df['월유니크고객수_수준'].fillna('미확인')
# --------------------------------------------------------------------------
# final_df 저장
# --------------------------------------------------------------------------

# 'data' 폴더 내에 저장
save_path = get_file_path("final_df.csv")

# CSV 파일 저장 (인덱스 제외)
final_clean_df.to_csv(save_path, index=False, encoding="utf-8-sig")

>>>>>>> 4025576cc0b52c8393af0ca720a1f6fabeb5e43a
print(f"CSV 파일 저장 완료: {save_path}")