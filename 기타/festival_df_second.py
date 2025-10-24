import pandas as pd
import re

# --- 1. 데이터 불러오기 ---
# 3개의 통합 CSV 파일 경로를 지정합니다.
path_indicators = 'C:/projects/shcard_2025_bigcontest/data/통합_문화관광축제_주요_지표.csv'
path_demographics = 'C:/projects/shcard_2025_bigcontest/data/통합_성_연령별_내국인_방문자.csv'
path_trend = 'C:/projects/shcard_2025_bigcontest/data/통합_연도별_방문자_추이.csv'

df_indicators = pd.read_csv(path_indicators)
df_demographics = pd.read_csv(path_demographics)
df_trend = pd.read_csv(path_trend)
print("✅ 3개의 통합 파일을 성공적으로 불러왔습니다.\n")


# --- 2. 연도별 데이터를 Wide 형식으로 변환하는 함수 ---
def pivot_by_year(df, index_col, year_col, drop_cols=None):
    """연도별 데이터를 (년도)_(컬럼명) 형태로 변환하는 함수"""
    if drop_cols:
        df = df.drop(columns=drop_cols)
    
    df_wide = df.pivot_table(index=index_col, columns=year_col)
    
    # 멀티레벨 컬럼을 (년도)_(컬럼명) 형식으로 합치기
    df_wide.columns = [f"{int(col[1])}_{col[0]}" for col in df_wide.columns]
    return df_wide.reset_index()


# --- 3. 각 데이터 정제 및 변환 ---

# 3-1. '연도별 방문자 추이' 데이터 변환
# 불필요하거나 중복될 수 있는 컬럼은 미리 제거
trend_drop_cols = ['일평균 방문자수 증감률', '(이전)전체방문자', '(전체)방문자증감', '전년대비방문자증감비율']
df_trend_wide = pivot_by_year(df_trend, '축제명', '개최년도', drop_cols=trend_drop_cols)
print("✅ '연도별 방문자 추이' 데이터를 Wide 형태로 변환했습니다.")

# 3-2. '주요 지표' 데이터 변환
# '그룹명'과 '구분명'을 합쳐 새로운 컬럼 생성
df_indicators['지표구분'] = df_indicators['그룹명'] + '_' + df_indicators['구분명']
df_indicators_intermediate = df_indicators.pivot_table(
    index=['축제명', '개최년도'], 
    columns='지표구분', 
    values='지표값'
).reset_index()
df_indicators_wide = pivot_by_year(df_indicators_intermediate, '축제명', '개최년도')
print("✅ '주요 지표' 데이터를 Wide 형태로 변환했습니다.")

# 3-3. '성_연령별 방문자' 데이터 변환 (이 데이터는 연도 정보가 없으므로 이전과 동일)
df_demographics_wide = df_demographics.pivot_table(
    index='축제명', 
    columns='연령대', 
    values=['남성비율', '여성비율']
).reset_index()
# 컬럼명 정리
df_demographics_wide.columns = [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in df_demographics_wide.columns]
df_demographics_wide.columns = [re.sub(r'[^A-Za-z0-9_가-힣]', '', col) for col in df_demographics_wide.columns]
print("✅ '성_연령별 방문자' 데이터를 Wide 형태로 변환했습니다.\n")


# --- 4. 모든 Wide 데이터 병합 (Merging) ---
# '성_연령별' 데이터를 기준으로 '연도별 추이'와 '주요 지표'를 합칩니다.
# how='outer'는 한쪽에만 있는 축제 정보도 누락시키지 않기 위함입니다.
final_df = pd.merge(df_demographics_wide, df_trend_wide, on='축제명', how='outer')
final_df = pd.merge(final_df, df_indicators_wide, on='축제명', how='outer')
print("✅ 모든 데이터를 하나의 DataFrame으로 최종 병합했습니다.")


# --- 5. 결과 확인 및 저장 ---
print("\n🎉 최종 통합 데이터(Wide) 샘플")
# 축제명과 연도 관련 컬럼 일부만 샘플로 출력
sample_cols = [col for col in final_df.columns if '2023' in col or '축제명' in col or '남성' in col]
print(final_df[sample_cols].head())

print(f"\n- 최종 데이터는 총 {len(final_df.columns)}개의 컬럼과 {len(final_df)}개의 행으로 구성됩니다.")

# 최종 데이터를 새로운 CSV 파일로 저장
final_df.to_csv('C:/projects/shcard_2025_bigcontest/data/festival_df.csv', index=False, encoding='utf-8-sig')
print("\n💾 'festival_df.csv' 파일이 성공적으로 저장되었습니다.")