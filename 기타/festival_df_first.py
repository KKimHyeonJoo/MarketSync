import pandas as pd
import glob
import os

# --- 설정 부분 ---
# 1. 데이터 파일들이 저장된 폴더 경로를 지정합니다.
# Windows 경로의 경우, 역슬래시(\)를 두 번 쓰거나(C:\\...) 슬래시(/)로 변경해야 합니다.
folder_path = 'C:/projects/shcard_2025_bigcontest/data/festival'

# 2. 통합된 파일을 저장할 경로를 지정합니다. (결과를 같은 폴더에 저장)
output_path = 'C:/projects/shcard_2025_bigcontest/data'

# --- 데이터 통합 함수 정의 ---
def combine_festival_data(path, pattern, output_filename):
    """
    지정된 경로에서 특정 패턴의 CSV 파일들을 찾아 하나로 통합하고 저장하는 함수.
    
    :param path: CSV 파일들이 있는 폴더 경로
    :param pattern: 찾을 파일 이름의 패턴 (예: '*_문화관광축제 주요 지표.csv')
    :param output_filename: 저장할 최종 CSV 파일 이름
    """
    # 지정된 경로와 패턴을 결합하여 파일 목록을 가져옵니다.
    file_list = glob.glob(os.path.join(path, pattern))
    
    if not file_list:
        print(f"⚠️ 경고: '{pattern}' 패턴에 해당하는 파일을 찾을 수 없습니다.")
        print(f"경로를 확인해주세요: {path}\n")
        return

    # 각 파일을 DataFrame으로 읽어 리스트에 추가합니다.
    df_list = [pd.read_csv(file) for file in file_list]
    
    # 모든 DataFrame을 하나로 합칩니다.
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # 통합된 데이터를 CSV 파일로 저장합니다.
    # encoding='utf-8-sig'는 Excel에서 한글이 깨지지 않도록 해줍니다.
    output_filepath = os.path.join(output_path, output_filename)
    combined_df.to_csv(output_filepath, index=False, encoding='utf-8-sig')
    
    print(f"✅ 성공: {len(file_list)}개의 파일을 통합하여 '{output_filename}'으로 저장했습니다.")
    print(f"    - 총 {len(combined_df)}개의 행이 생성되었습니다.")
    print(f"    - 저장 경로: {output_filepath}\n")
    

# --- 메인 코드 실행 ---
print("===== 축제 데이터 통합을 시작합니다. =====\n")

# 1. 문화관광축제 주요 지표 통합
combine_festival_data(folder_path, '*_문화관광축제 주요 지표.csv', '통합_문화관광축제_주요_지표.csv')

# 2. 성_연령별 내국인 방문자 통합
combine_festival_data(folder_path, '*_성_연령별 내국인 방문자.csv', '통합_성_연령별_내국인_방문자.csv')

# 3. 연도별 방문자 추이 통합
combine_festival_data(folder_path, '*_연도별 방문자 추이.csv', '통합_연도별_방문자_추이.csv')

print("===== 모든 데이터 통합 작업이 완료되었습니다. =====")