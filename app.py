import streamlit as st
import cv2
import time
from getposedata import process_frame
import mediapipe as mp
import datetime
from zoneinfo import ZoneInfo

st.set_page_config(
    page_title="지능형 노인 낙상 감지 시스템",
    layout="wide",
)

kst_now = datetime.datetime.now(ZoneInfo("Asia/Seoul"))
timestamp = kst_now.strftime("%Y%m%d_%H%M%S")

if 'history' not in st.session_state:
    st.session_state.history = []

if 'camera' not in st.session_state:
    st.session_state.camera = None

# 상단 상태바
st.markdown("### 🛡️ 시스템 상태")
status_col1, status_col2 = st.columns([1, 3])
with status_col1:
    camera_status = "🟢 켜짐" if st.session_state.camera else "🔴 꺼짐"
    st.metric(label="📷 카메라 상태", value=camera_status)
with status_col2:
    recent_status = st.session_state.history[-1] if st.session_state.history else "대기 중"
    st.metric(label="📊 최근 낙상 상태", value=recent_status)

# 메인 화면 2분할
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("🔍 실시간 관절 추출")
    frame_display = st.empty()

    button_col1, button_col2 = st.columns([1, 1])
    with button_col1:
        start = st.button("▶ 카메라 시작", use_container_width=True)
    with button_col2:
        stop = st.button("⏹ 카메라 종료", use_container_width=True)

with col2:
    st.subheader("🦴 관절 정보")
    landmark_info = st.empty()
    st.subheader("📜 상태 기록")
    history_area = st.empty()

# 카메라 제어
if start:
    st.session_state.camera = cv2.VideoCapture(0)
if stop and st.session_state.camera:
    st.session_state.camera.release()
    st.session_state.camera = None

# 프레임 처리 루프
last_print_time = 0
if st.session_state.camera and st.session_state.camera.isOpened():
    while True:
        ret, frame = st.session_state.camera.read()
        if not ret:
            st.warning("⚠️ 카메라 프레임을 읽을 수 없습니다.")
            break

        image, landmarks = process_frame(frame)
        frame_display.image(image, channels="RGB", use_container_width=True)

        current_time = time.time()
        if landmarks and current_time - last_print_time >= 1:
            left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
            left_knee = landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE]
            right_knee = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE]

            table_md = f"""
            |관절|X|Y|Z|신뢰도|적합|
            |:--:|:--:|:--:|:--:|:--:|:--:|
            |왼쪽 어깨|{left_shoulder.x:.3f}|{left_shoulder.y:.3f}|{left_shoulder.z:.3f}|{left_shoulder.visibility:.2f}|{"✅" if left_shoulder.visibility > 0.7 else "❌"}|
            |오른쪽 어깨|{right_shoulder.x:.3f}|{right_shoulder.y:.3f}|{right_shoulder.z:.3f}|{right_shoulder.visibility:.2f}|{"✅" if right_shoulder.visibility > 0.7 else "❌"}|
            |왼쪽 무릎|{left_knee.x:.3f}|{left_knee.y:.3f}|{left_knee.z:.3f}|{left_knee.visibility:.2f}|{"✅" if left_knee.visibility > 0.7 else "❌"}|
            |오른쪽 무릎|{right_knee.x:.3f}|{right_knee.y:.3f}|{right_knee.z:.3f}|{right_knee.visibility:.2f}|{"✅" if right_knee.visibility > 0.7 else "❌"}|
            """
            landmark_info.markdown(table_md)
            last_print_time = current_time

        time.sleep(0.03)

# 상태 기록 출력
if st.session_state.history:
    with history_area:
        st.markdown("#### 📝 감지 로그")
        for i, item in enumerate(reversed(st.session_state.history[-5:]), 1):
            st.markdown(f"{i}. {item}")
