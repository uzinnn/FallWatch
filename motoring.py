import streamlit as st
import cv2
import mediapipe as mp
import psutil
import pandas as pd
import os
import datetime
from zoneinfo import ZoneInfo

from script import util
# from script import fallpredict  # ← 여기 주석 해제하면 실제 감지 모듈 연결 가능
import time
import random  # 테스트용


def show():
    st.title("🛡️ 감시 모드")
    st.write("낙상 여부를 실시간으로 감지합니다.")

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    landmark_logs = []

    # 초기화
    if 'camera' not in st.session_state:
        st.session_state.camera = None
    if 'fall_count' not in st.session_state:
        st.session_state.fall_count = 0

    # 현재 시간(KST) 설정
    kst_now = datetime.datetime.now(ZoneInfo("Asia/Seoul"))
    timestamp = kst_now.strftime("%Y-%m-%d %H:%M:%S")


    
    st.markdown(
    """
    <style>
    .status-box {
        border: 2px solid #d0d0d0;
        border-radius: 10px;
        padding: 15px;
        background-color: #f9f9f9;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
    }
     .status-card {
        background-color: #F9FAFB;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .status-on { color: green; }
    .status-off { color: red; }
    </style>
    """,
    unsafe_allow_html=True
)

    

    col1_top, col3_top = st.columns([5, 5])
    with col1_top:
        col1_box = st.empty()
        col1_box.warning("📷 카메라 대기 중")
        st.markdown(f"**CPU 사용량:** {psutil.cpu_percent()}%")
        memory = psutil.virtual_memory()
        st.markdown(f"**메모리 사용량:** {memory.percent}%")

    with col3_top:
        col3_box = st.empty()
        col3_box.info("🔍 낙상 감지 결과 대기 중...")

    st.markdown("---")

    col1, col2 = st.columns([6, 4])
    with col2:
        st.markdown("### 📥 실시간 분석 데이터를 추출합니다.")
        landmarks_box = st.empty()

    with col1:
        st.markdown("### 🎥 웹캠을 이용한 실시간 분석")
        button_col1, button_col2, _ = st.columns([2, 2, 2])
        with button_col1:
            start = st.button("카메라 시작")
        with button_col2:
            stop = st.button("카메라 종료")

        frame_placeholder = st.empty()
        landmark_data = []

        if start:
            st.session_state.camera = cv2.VideoCapture(0)
            col1_box.success("🟢 카메라 켜짐")

            pose = mp_pose.Pose()
            analyzing = True
            last_check_time = time.time()
            check_interval = 3  # 3초마다 체크
            buffer_limit = 3
            frame_buffer = []

            while analyzing:
                ret, frame = st.session_state.camera.read()
                if not ret:
                    st.warning("카메라 프레임을 읽을 수 없습니다.")
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    lm = results.pose_landmarks.landmark
                    frame_landmarks = {
                        "timestamp": util.now_kst(),
                        "left_shoulder": extract_landmark(lm, mp_pose.PoseLandmark.LEFT_SHOULDER),
                        "right_shoulder": extract_landmark(lm, mp_pose.PoseLandmark.RIGHT_SHOULDER),
                        "left_knee": extract_landmark(lm, mp_pose.PoseLandmark.LEFT_KNEE),
                        "right_knee": extract_landmark(lm, mp_pose.PoseLandmark.RIGHT_KNEE),
                    }
                    landmark_data.append(frame_landmarks)
                    frame_buffer.append(frame_landmarks)
                    if len(frame_buffer) > buffer_limit:
                        frame_buffer.pop(0)

                    # 로그 출력
                    log_text = f"""
                    ⏱ {frame_landmarks['timestamp']}  
        🦴 왼쪽 어깨: {frame_landmarks['left_shoulder']}  
        🦴 오른쪽 어깨: {frame_landmarks['right_shoulder']}  
        🦵 왼쪽 무릎: {frame_landmarks['left_knee']}  
        🦵 오른쪽 무릎: {frame_landmarks['right_knee']}
                    """
                    landmark_logs.append(log_text)
                    landmarks_box.markdown("### 📝 누적 좌표 로그\n\n" + '\n---\n'.join(landmark_logs[-3:]), unsafe_allow_html=True)

                # 프레임 출력
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame, channels="RGB")

                # 낙상 감지 (3초마다 실행)
                if time.time() - last_check_time >= check_interval and len(frame_buffer) == buffer_limit:
                    last_check_time = time.time()

                    # 실제 낙상 감지 모델 연결 (주석 해제)
                    # is_fall = fallpredict.is_fallen(frame_buffer)
                    is_fall = random.choice([True, False])  # 테스트용

                    if is_fall:
                        st.session_state.fall_count += 1
                        col3_box.error(f"🚨 낙상이 감지되었습니다! (총 감지 횟수: {st.session_state.fall_count})")
                    else:
                        col3_box.success(f"✅ 안전한 자세입니다. (총 낙상 감지 횟수: {st.session_state.fall_count})")

                # 종료 버튼 눌림
                if stop:
                   
                    col1_box.warning("🛑 카메라가 종료되었습니다.")
                    col3_box.info(f"📊 총 낙상 감지 횟수: {st.session_state.fall_count}")
                    analyzing = False
                    break

                time.sleep(0.1)

            st.session_state.camera.release()
            st.session_state.camera = None

# 푸터
   
    st.markdown(f"""
    <div style="text-align: center; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #E5E7EB; color: #9CA3AF; font-size: 0.875rem;">
    © 2025 지능형 노인 낙상 감지 시스템 | 현재 시간: {timestamp}
</div>
    """, unsafe_allow_html=True)  


# 랜드마크 정보 정리 함수
def extract_landmark(lm, point):
    return (
        round(lm[point].x, 2),
        round(lm[point].y, 2),
        round(lm[point].visibility, 2),
        1 if round(lm[point].visibility, 2) >= 0.7 else 0
    )
