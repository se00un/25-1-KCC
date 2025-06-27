import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import matplotlib.font_manager as fm
from sklearn.decomposition import PCA
import platform

# 한글 폰트 설정
if platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
elif platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'

# 마이너스 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

#1. 연령대 분류 함수
def get_age_group(age):
    if 3 <= age <= 5:
        return '3-5세'
    elif 6 <= age <= 8:
        return '6-8세'
    elif 9 <= age <= 11:
        return '9-11세'
    else:
        raise ValueError("지원 연령은 3세~11세입니다!")
    
# 2. HHI 계산 함수
def hhi(row, cols):
    values = row[cols].values
    total = values.sum()
    if total == 0:
        return 0
    shares = values / total
    return sum(shares ** 2)

# 3. 클러스터 설명 매핑
cluster_descriptions = {
    '3-5세': {
        0: {'type': 'Heavy 멀티유저형 유아', 'description': '장시간, 다매체, 고빈도 이용 형태로, 조기 미디어 습관 형성 가능성이 높습니다.', 'keywords': ['#고사용량', '#다매체', '#고빈도', '#몰입형유아'], 
            'image_url': 'Baby_pic.png'},
        1: {'type': '저이용 집중형', 'description': '짧은 시간, 제한된 매체 중심의 저이용 집단입니다. 미디어 이용 자체가 제한적입니다.', 'keywords': ['#저이용', '#편중형', '#비이용근접', '#3-5세 한정 시청'], 
            'image_url': 'Baby_pic.png'},
        2: {'type': '균형 사용자형', 'description': '중간 수준의 시청 시간과 빈도, 낮은 편향도를 가진 이상적인 유형입니다.', 'keywords': ['#적정시청', '#균형사용자', '#고른이용', '#안정형유아']
            , 'image_url': 'Baby_pic.png'}
    },
    '6-8세': {
        0: {'type': '저이용 집중형', 'description': '낮은 사용량과 특정 매체 편중이 특징이며, 학습 중심 또는 보호자 통제가 작용한 것으로 보입니다.', 'keywords': ['#학습중심', '#편중형', '#저이용초등', '#제한사용자'],
            'image_url': 'Kindergarden_pic.png'},
        1: {'type': 'Heavy 멀티 유저형', 'description': '자주, 오래, 다양한 매체를 탐색하며 사용하는 능동적인 시청자입니다.', 'keywords': ['#고이용초등', '#다매체', '#탐색형'], 
            'image_url': 'Kindergarden_pic.png'},
        2: {'type': '주말 전용형', 'description': '주중에는 거의 사용하지 않고, 주말에만 집중적으로 사용하는 유형입니다.', 'keywords': ['#주말전용', '#시간제한형', '#요일기반', '#보호자주도형'], 
            'image_url': 'Kindergarden_pic.png'}
    },
    '9-11세': {
        0: {'type': '저이용 집중형 이용자', 'description': '낮은 사용량과 빈도, 특정 콘텐츠에 몰입하는 루틴 소비형입니다.', 'keywords': ['#고정몰입형', '#비확산', '#저이용고학년', '#루틴소비'], 
            'image_url': 'Elementay_pic.png'},
        1: {'type': '균형 사용자형', 'description': '매체 사용량과 분산도 모두 적당한, 초등 고학년의 안정적인 일반형입니다.', 'keywords': ['#일상시청', '#중간사용량', '#규칙적', '#균형소비'], 
            'image_url': 'Elementay_pic.png'},
        2: {'type': '탐색적 사용자형', 'description': '많고 자주 시청하며, 다매체를 탐색적으로 활용하는 확장형 사용자입니다.', 'keywords': ['#탐색형고학년', '#다매체', '#고빈도', '#자율시청확장형'], 
            'image_url': 'Elementay_pic.png'}
    }
}

# 4. 사용자 입력 → 파생 변수 생성
def preprocess_input(user_info):
    time_weekday = ['TV_주중', '컴퓨터_주중', '스마트폰_주중', '태블릿_주중']
    time_weekend = ['TV_주말', '컴퓨터_주말', '스마트폰_주말', '태블릿_주말']
    freq_weekday = ['TV빈도_주중', '컴퓨터빈도_주중', '스마트폰빈도_주중', '태블릿빈도_주중']
    freq_weekend = ['TV빈도_주말', '컴퓨터빈도_주말', '스마트폰빈도_주말', '태블릿빈도_주말']

    df = pd.DataFrame([user_info])
    df['이름'] = user_info.get('이름')
    df['나이'] = user_info.get('나이')
    df['연령대'] = get_age_group(user_info['나이'])
    df['총_주중_이용시간'] = df[time_weekday].sum(axis=1)
    df['총_주말_이용시간'] = df[time_weekend].sum(axis=1)
    df['평균_주중_빈도'] = df[freq_weekday].mean(axis=1)
    df['평균_주말_빈도'] = df[freq_weekend].mean(axis=1)
    df['편중_HHI_주중'] = df.apply(lambda row: hhi(row, time_weekday), axis=1)
    df['편중_HHI_주말'] = df.apply(lambda row: hhi(row, time_weekend), axis=1)
    df['TV_시작시기'] = user_info.get('TV_시작시기', None)
    df['스마트폰_시작시기'] = user_info.get('스마트폰_시작시기', None)

    processed = df[['이름', '나이', '연령대', '총_주중_이용시간', '총_주말_이용시간', '평균_주중_빈도', '평균_주말_빈도', '편중_HHI_주중', '편중_HHI_주말', 'TV_시작시기', '스마트폰_시작시기']]
    return processed, df['연령대'].values[0]


# 5. 클러스터 결과 확인
def predict_cluster_from_df(df_with_labels, user_info, features):
    processed, age_group = preprocess_input(user_info)
    df_group = df_with_labels[df_with_labels['연령대'] == age_group].dropna(subset=features + [f'cluster_{age_group}'])

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_group = scaler.fit_transform(df_group[features])
    scaled_input = scaler.transform(processed[features])

    distances = np.linalg.norm(scaled_group - scaled_input, axis=1)
    nearest_index = df_group.index[np.argmin(distances)]
    predicted_cluster = int(df_group.loc[nearest_index, f'cluster_{age_group}'])
    desc = cluster_descriptions[age_group][predicted_cluster]

    st.header("📺 우리 아이의 미디어 이용 분석 결과! 📊")
    st.markdown("---")
    st.markdown(f"### 🧒 {user_info.get('이름')}님은 **{age_group}** 그룹에 속하며...")
    st.markdown(f"## 🌈 {desc['type']} 유형이에요!")
    st.markdown(f" 키워드: `{', '.join(desc['keywords'])}`")
    st.image(desc.get('image_url', ''), width=300)
    st.markdown(f"{desc['description']}")

    st.markdown("---")
    st.markdown("#### 📌 전체 클러스터 속 내 자녀의 위치는 어디일까요?")
    plot_radar_chart(df_group, df_group.loc[nearest_index])

    return {
        '이름': user_info.get('이름'),
        '연령대': age_group,
        '클러스터': predicted_cluster,
        '유형': desc['type'],
        '설명': desc['description'],
        '키워드': desc['keywords'],
        'TV_시작시기': user_info.get('TV_시작시기'),
        '스마트폰_시작시기': user_info.get('스마트폰_시작시기')
    }

# 6. 시각화 함수
def visualize_child_in_group(df_with_labels, child_info):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import streamlit as st

    age_group = get_age_group(child_info['나이'])
    df_group = df_with_labels[df_with_labels['연령대'] == age_group]

    weekday_time = sum(child_info.get(k, 0) for k in ['TV_주중', '컴퓨터_주중', '스마트폰_주중', '태블릿_주중'])
    weekend_time = sum(child_info.get(k, 0) for k in ['TV_주말', '컴퓨터_주말', '스마트폰_주말', '태블릿_주말'])
    weekday_freq = sum(child_info.get(k, 0) for k in ['TV빈도_주중', '컴퓨터빈도_주중', '스마트폰빈도_주중', '태블릿빈도_주중']) / 4
    weekend_freq = sum(child_info.get(k, 0) for k in ['TV빈도_주말', '컴퓨터빈도_주말', '스마트폰빈도_주말', '태블릿빈도_주말']) / 4

    def hhi_from_values(values):
        total = sum(values)
        if total == 0:
            return None
        shares = [v / total for v in values]
        return sum([s ** 2 for s in shares])

    hhi_weekday = hhi_from_values([child_info.get(k, 0) for k in ['TV_주중', '컴퓨터_주중', '스마트폰_주중', '태블릿_주중']])
    hhi_weekend = hhi_from_values([child_info.get(k, 0) for k in ['TV_주말', '컴퓨터_주말', '스마트폰_주말', '태블릿_주말']])

    child_point = {
        '이름': child_info.get('이름'),
        '연령대': age_group,
        '총_주중_이용시간': weekday_time,
        '총_주말_이용시간': weekend_time,
        '평균_주중_빈도': weekday_freq,
        '평균_주말_빈도': weekend_freq,
        '편중_HHI_주중': hhi_weekday,
        '편중_HHI_주말': hhi_weekend
    }

    feature_pairs = [
        (('총_주중_이용시간', '총_주말_이용시간'), '총 이용시간 (분)'),
        (('평균_주중_빈도', '평균_주말_빈도'), '평균 이용 빈도'),
        (('편중_HHI_주중', '편중_HHI_주말'), '매체 편중도 (HHI)')
    ]
    st.success("📌 시각화를 통해 자녀의 이용 상태를 또래와 비교해 보세요!")

    for (col1, col2), label in feature_pairs:
        for col, subtitle in zip([col1, col2], ['주중', '주말']):
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.boxplot(data=df_group, y=col, ax=ax, width=0.4, fliersize=3, linewidth=1.2)
            if child_point[col] is not None and pd.notna(child_point[col]):
                ax.scatter(x=0, y=child_point[col], color='red', s=50, marker='D', label='내 자녀', zorder=10)
                mean_val = df_group[col].mean()
                st.markdown(f"🔍 평균보다 **{child_point[col] - mean_val:.1f}** 만큼 {'많이' if child_point[col] > mean_val else '적게'} 이용하고 있어요!")
                ax.legend()
            ax.set_title(f"{age_group} - {label} ({subtitle})", fontsize=12)
            ax.set_ylabel(label)
            ax.set_xlabel("")
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
            st.pyplot(fig)

    st.markdown("---")
    


# 7. 클러스터 시각화
def plot_radar_chart(df_group, child_info):
    st.subheader("📊 유형별 특성 비교 (Radar Chart)")

    features = ['총_주중_이용시간', '총_주말_이용시간',
                '평균_주중_빈도', '평균_주말_빈도',
                '편중_HHI_주중', '편중_HHI_주말']

    group_mean = df_group[features].mean()
    child_values = [
        child_info.get('총_주중_이용시간', 0),
        child_info.get('총_주말_이용시간', 0),
        child_info.get('평균_주중_빈도', 0),
        child_info.get('평균_주말_빈도', 0),
        child_info.get('편중_HHI_주중', 0),
        child_info.get('편중_HHI_주말', 0),
    ]

    labels = ['주중 이용시간', '주말 이용시간', '주중 빈도', '주말 빈도', '매체 편중도 주중', '매체 편중도 주말']
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    child_values += child_values[:1]
    group_mean = group_mean.tolist() + [group_mean.tolist()[0]]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, child_values, label='우리 아이', color='red', linewidth=2)
    ax.fill(angles, child_values, color='red', alpha=0.25)

    ax.plot(angles, group_mean, label='평균', color='gray', linestyle='dashed')
    ax.fill(angles, group_mean, color='gray', alpha=0.15)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title("자녀 vs 또래 평균 비교", fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    st.pyplot(fig)

def main():
    st.set_page_config(page_title="쪼꼬미디어", page_icon="🍫")
    st.image("logo.png", width=200)
    st.title("쪼꼬미디어 🐻📱")
    st.write("우리 아이의 미디어 이용 습관을 함께 지켜봐요!")

    if "user_input" not in st.session_state:
        st.session_state.user_input = None
    if "show_viz" not in st.session_state:
        st.session_state.show_viz = False

    with st.form("user_input_form"):
        st.subheader("👨‍👩‍👧 자녀 정보 입력")
        이름 = st.text_input("자녀 이름")
        나이 = st.number_input("나이 (3~11세)", min_value=3, max_value=11, step=1)
        st.subheader("📺 주중/주말 기기 이용 시간 (분)")
        TV_주중 = st.number_input("TV (주중)", min_value=0)
        컴퓨터_주중 = st.number_input("컴퓨터 (주중)", min_value=0)
        스마트폰_주중 = st.number_input("스마트폰 (주중)", min_value=0)
        태블릿_주중 = st.number_input("태블릿 (주중)", min_value=0)
        TV_주말 = st.number_input("TV (주말)", min_value=0)
        컴퓨터_주말 = st.number_input("컴퓨터 (주말)", min_value=0)
        스마트폰_주말 = st.number_input("스마트폰 (주말)", min_value=0)
        태블릿_주말 = st.number_input("태블릿 (주말)", min_value=0)
        st.subheader("📊 주중/주말 이용 빈도 (0~7일 기준)")
        TV빈도_주중 = st.number_input("TV 빈도 (주중)", min_value=0, max_value=5)
        컴퓨터빈도_주중 = st.number_input("컴퓨터 빈도 (주중)", min_value=0, max_value=5)
        스마트폰빈도_주중 = st.number_input("스마트폰 빈도 (주중)", min_value=0, max_value=5)
        태블릿빈도_주중 = st.number_input("태블릿 빈도 (주중)", min_value=0, max_value=5)
        TV빈도_주말 = st.number_input("TV 빈도 (주말)", min_value=0, max_value=2)
        컴퓨터빈도_주말 = st.number_input("컴퓨터 빈도 (주말)", min_value=0, max_value=2)
        스마트폰빈도_주말 = st.number_input("스마트폰 빈도 (주말)", min_value=0, max_value=2)
        태블릿빈도_주말 = st.number_input("태블릿 빈도 (주말)", min_value=0, max_value=2)
        st.subheader("🕒 미디어 이용 시작 시기")
        TV_시작시기 = st.number_input("TV 이용 시작 나이", min_value=0)
        스마트폰_시작시기 = st.number_input("스마트폰 이용 시작 나이", min_value=0)

        submitted = st.form_submit_button("분석하기 🔍")

    if submitted:
        st.session_state.user_input = {
            '이름': 이름, '나이': 나이,
            'TV_주중': TV_주중, '컴퓨터_주중': 컴퓨터_주중, '스마트폰_주중': 스마트폰_주중, '태블릿_주중': 태블릿_주중,
            'TV_주말': TV_주말, '컴퓨터_주말': 컴퓨터_주말, '스마트폰_주말': 스마트폰_주말, '태블릿_주말': 태블릿_주말,
            'TV빈도_주중': TV빈도_주중, '컴퓨터빈도_주중': 컴퓨터빈도_주중, '스마트폰빈도_주중': 스마트폰빈도_주중, '태블릿빈도_주중': 태블릿빈도_주중,
            'TV빈도_주말': TV빈도_주말, '컴퓨터빈도_주말': 컴퓨터빈도_주말, '스마트폰빈도_주말': 스마트폰빈도_주말, '태블릿빈도_주말': 태블릿빈도_주말,
            'TV_시작시기': TV_시작시기,
            '스마트폰_시작시기': 스마트폰_시작시기
        }
        st.session_state.show_viz = False

    # 분석 결과 출력
    if st.session_state.user_input:
        df_clustered = pd.read_csv('media_summary_cluster.csv')
        result = predict_cluster_from_df(df_clustered, st.session_state.user_input, [
            '총_주중_이용시간', '총_주말_이용시간',
            '평균_주중_빈도', '평균_주말_빈도',
            '편중_HHI_주중', '편중_HHI_주말'
        ])

        if st.button("📊 분석 결과 자세히 보기"):
            st.session_state.show_viz = True

        if st.session_state.show_viz:
            visualize_child_in_group(df_clustered, st.session_state.user_input)


main()
