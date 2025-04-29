import streamlit as st
import cv2
import time
import psutil
import numpy as np
import mediapipe as mp
import datetime
from zoneinfo import ZoneInfo
from getposedata import process_frame

# 페이지 설정
st.set_page_config(
    page_title="지능형 노인 낙상 감지 시스템",
    layout="wide",
)

# 커스텀 CSS는 기존 코드와 동일하게 유지
# 커스텀 CSS 적용
st.markdown("""
<style>
    .title {
        font-size: 2rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        padding: 0.5rem;
        border-bottom: 2px solid #EEF2FF;
    }
    .status-card {
        background-color: #F9FAFB;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .camera-on {
        color: #10B981;
        font-weight: bold;
    }
    .camera-off {
        color: #EF4444;
        font-weight: bold;
    }
    .status-normal {
        color: #10B981;
        font-weight: bold;
    }
    .status-warning {
        color: #F59E0B;
        font-weight: bold;
    }
    .status-danger {
        color: #EF4444;
        font-weight: bold;
    }
    .subheader {
        color: #1E3A8A;
        font-size: 1.3rem;
        font-weight: bold;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid #EEF2FF;
    }
    .info-text {
        background-color: #F3F4F6;
        padding: 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.9rem;
    }
    .log-item {
        padding: 0.5rem;
        margin-bottom: 0.3rem;
        border-radius: 0.3rem;
        background-color: #F9FAFB;
        border-left: 3px solid #3B82F6;
    }
</style>
""", unsafe_allow_html=True)


# 세션 상태 초기화
if 'history' not in st.session_state:
    st.session_state.history = []

if 'fall_count' not in st.session_state:
    st.session_state.fall_count = 0

if 'last_status' not in st.session_state:
    st.session_state.last_status = "대기 중"

if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False

# 현재 시간(KST) 설정
kst_now = datetime.datetime.now(ZoneInfo("Asia/Seoul"))
timestamp = kst_now.strftime("%Y-%m-%d %H:%M:%S")

# 낙상 감지 함수 (기존 코드와 동일)
def detect_fall(landmarks):
    """MediaPipe 관절 좌표를 기반으로 낙상 상태를 감지하는 함수"""
    if not landmarks:
        return "정상: 감지 중", False
    
    # 필요한 관절 좌표 추출
    key_points = [
        mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
        mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
        mp.solutions.pose.PoseLandmark.LEFT_HIP,
        mp.solutions.pose.PoseLandmark.RIGHT_HIP,
        mp.solutions.pose.PoseLandmark.LEFT_KNEE,
        mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
        mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
        mp.solutions.pose.PoseLandmark.RIGHT_ANKLE
    ]
    
    # 필요한 관절이 모두 감지되었는지 확인
    if not all(point.value in landmarks.landmark for point in key_points):
        return "주의: 일부 관절 감지 불가", False
    
    left_shoulder_y = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y
    right_shoulder_y = landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y
    left_hip_y = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y
    right_hip_y = landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].y
    
    shoulder_hip_diff = ((left_shoulder_y + right_shoulder_y) / 2) - ((left_hip_y + right_hip_y) / 2)
    abs_shoulder_hip_diff = abs(shoulder_hip_diff)
    
    left_knee_y = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].y
    right_knee_y = landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].y
    left_ankle_y = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].y
    right_ankle_y = landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].y
    
    confidences = [landmarks.landmark[point.value].visibility for point in key_points]
    avg_confidence = sum(confidences) / len(confidences)
    
    if abs_shoulder_hip_diff < 0.15 and avg_confidence > 0.6:
        knee_ankle_shoulder_diff = abs(((left_knee_y + right_knee_y) / 2) - ((left_shoulder_y + right_shoulder_y) / 2))
        
        if knee_ankle_shoulder_diff < 0.3:
            return "위험: 낙상 감지됨", True
        else:
            return "주의: 비정상적 자세", False
    
    elif shoulder_hip_diff < -0.2 and avg_confidence > 0.7:
        return "정상: 안정적 자세", False
    
    elif abs_shoulder_hip_diff < 0.2 and avg_confidence > 0.7:
        return "주의: 불안정한 자세", False
        
    return "정상: 모니터링 중", False

# 특정 관절의 정보를 테이블로 표시하는 함수 (기존 코드와 동일)
def display_landmarks(landmarks):
    if not landmarks:
        return "<div class='info-text'>감지된 관절 정보가 없습니다.</div>"
    
    key_joints = [
        (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, "왼쪽 어깨"),
        (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, "오른쪽 어깨"),
        (mp.solutions.pose.PoseLandmark.LEFT_HIP, "왼쪽 엉덩이"),
        (mp.solutions.pose.PoseLandmark.RIGHT_HIP, "오른쪽 엉덩이"),
        (mp.solutions.pose.PoseLandmark.LEFT_KNEE, "왼쪽 무릎"),
        (mp.solutions.pose.PoseLandmark.RIGHT_KNEE, "오른쪽 무릎"),
        (mp.solutions.pose.PoseLandmark.LEFT_ANKLE, "왼쪽 발목"),
        (mp.solutions.pose.PoseLandmark.RIGHT_ANKLE, "오른쪽 발목")
    ]
    
    table_html = """
    <table style="width:100%; border-collapse: collapse;">
      <tr style="background-color: #EEF2FF;">
        <th style="text-align: left; padding: 0.5rem;">관절</th>
        <th style="text-align: center; padding: 0.5rem;">X</th>
        <th style="text-align: center; padding: 0.5rem;">Y</th>
        <th style="text-align: center; padding: 0.5rem;">Z</th>
        <th style="text-align: center; padding: 0.5rem;">신뢰도</th>
      </tr>
    """
    
    for idx, (landmark_id, joint_name) in enumerate(key_joints):
        if landmark_id.value in landmarks.landmark:
            landmark = landmarks.landmark[landmark_id.value]
            
            confidence_color = "#10B981" if landmark.visibility > 0.7 else \
                              "#F59E0B" if landmark.visibility > 0.5 else "#EF4444"
            
            bg_color = "#F9FAFB" if idx % 2 == 0 else "white"
            
            table_html += f"""
            <tr style="background-color: {bg_color};">
              <td style="padding: 0.5rem;">{joint_name}</td>
              <td style="text-align: center; padding: 0.5rem;">{landmark.x:.3f}</td>
              <td style="text-align: center; padding: 0.5rem;">{landmark.y:.3f}</td>
              <td style="text-align: center; padding: 0.5rem;">{landmark.z:.3f}</td>
              <td style="text-align: center; padding: 0.5rem; color: {confidence_color}; font-weight: bold;">
                {landmark.visibility:.2f}
              </td>
            </tr>
            """
    
    table_html += "</table>"
    return table_html

# 제목 표시
st.markdown("<div class='title'>🛡️ 지능형 노인 낙상 감지 시스템</div>", unsafe_allow_html=True)

# 상단 상태 표시줄
st.markdown("<div class='subheader'>📊 시스템 상태</div>", unsafe_allow_html=True)
status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    camera_status = "🟢 켜짐" if st.session_state.camera_active else "🔴 꺼짐"
    camera_class = "camera-on" if st.session_state.camera_active else "camera-off"
    st.markdown(f"""
    <div class="status-card">
        <div style="font-weight: bold;">📷 카메라 상태</div>
        <div class="{camera_class}">{camera_status}</div>
    </div>
    """, unsafe_allow_html=True)

with status_col2:
    recent_status = st.session_state.last_status
    status_class = "status-normal"
    if "주의" in recent_status:
        status_class = "status-warning"
    elif "위험" in recent_status or "낙상" in recent_status:
        status_class = "status-danger"
    
    st.markdown(f"""
    <div class="status-card">
        <div style="font-weight: bold;">🔍 현재 상태</div>
        <div class="{status_class}">{recent_status}</div>
    </div>
    """, unsafe_allow_html=True)

with status_col3:
    st.markdown(f"""
    <div class="status-card">
        <div style="font-weight: bold;">⚠️ 낙상 감지</div>
        <div>{st.session_state.fall_count}회</div>
    </div>
    """, unsafe_allow_html=True)

# 메인 화면 2분할
col1, col2 = st.columns([1.5, 1])

with col1:
    st.markdown("<div class='subheader'>📹 실시간 관절 추출</div>", unsafe_allow_html=True)
    
    # 카메라 활성화/비활성화 토글
    camera_toggle = st.checkbox("카메라 활성화", value=st.session_state.camera_active)
    if camera_toggle != st.session_state.camera_active:
        st.session_state.camera_active = camera_toggle
        status_time = datetime.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")
        if camera_toggle:
            st.session_state.history.append(f"[{status_time}]: 카메라 활성화")
            st.session_state.last_status = "카메라 활성화"
        else:
            st.session_state.history.append(f"[{status_time}]: 카메라 비활성화")
            st.session_state.last_status = "카메라 비활성화"
    
    # Streamlit의 내장 카메라 컴포넌트 사용
    if st.session_state.camera_active:
        camera_image = st.camera_input("", key="camera", label_visibility="hidden")
        
        if camera_image is not None:
            # 이미지를 OpenCV 형식으로 변환
            bytes_data = camera_image.getvalue()
            img_array = np.frombuffer(bytes_data, np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            # 이미지 처리 및 관절 추출
            try:
                processed_frame, landmarks = process_frame(frame)
                
                # 결과 이미지 표시
                st.image(processed_frame, channels="BGR", use_column_width=True)
                
                # 낙상 상태 체크
                status, is_fall = detect_fall(landmarks)
                status_time = datetime.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")
                
                if is_fall:
                    st.session_state.fall_count += 1
                
                # 상태가 변경된 경우에만 기록
                if st.session_state.last_status != status:
                    st.session_state.history.append(f"[{status_time}]: {status}")
                    st.session_state.last_status = status
                    
                    # 히스토리 크기 제한
                    if len(st.session_state.history) > 10:
                        st.session_state.history = st.session_state.history[-10:]
                
                # 관절 정보 업데이트
                with col2:
                    st.markdown("<div class='subheader'>🦴 관절 정보</div>", unsafe_allow_html=True)
                    if landmarks:
                        landmark_table = display_landmarks(landmarks)
                        st.markdown(landmark_table, unsafe_allow_html=True)
                    else:
                        st.markdown("<div class='info-text'>감지된 관절 정보가 없습니다.</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"이미지 처리 오류: {str(e)}")
    else:
        # 카메라가 비활성화된 경우 기본 화면 표시
        st.markdown("""
        <div style="
            display: flex;
            justify-content: center;
            align-items: center;
            height: 400px;
            background-color: #F3F4F6;
            border-radius: 10px;
            text-align: center;
        ">
            <div>
                <div style="font-size: 3rem; margin-bottom: 1rem;">📹</div>
                <div>카메라가 활성화되지 않았습니다.<br>
                '카메라 활성화' 체크박스를 선택하여 모니터링을 시작하세요.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    if not st.session_state.camera_active:
        st.markdown("<div class='subheader'>🦴 관절 정보</div>", unsafe_allow_html=True)
        st.markdown("<div class='info-text'>카메라가 비활성화 상태입니다. 관절 정보를 표시할 수 없습니다.</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='subheader'>📜 상태 기록</div>", unsafe_allow_html=True)
    
    # 히스토리 표시
    if st.session_state.history:
        history_html = "<div style='max-height: 300px; overflow-y: auto;'>"
        for item in reversed(st.session_state.history):
            status_class = "status-normal"
            if "주의" in item:
                status_class = "status-warning"
            elif "위험" in item or "낙상" in item:
                status_class = "status-danger"
                
            history_html += f"<div class='log-item'><span class='{status_class}'>{item}</span></div>"
        history_html += "</div>"
        
        st.markdown(history_html, unsafe_allow_html=True)
    else:
        st.markdown("<div class='info-text'>기록된 활동이 없습니다.</div>", unsafe_allow_html=True)

# 사이드바에 시스템 모니터링 정보 표시
st.sidebar.title("시스템 모니터링")
st.sidebar.metric("CPU 사용량", f"{psutil.cpu_percent()}%")
memory = psutil.virtual_memory()
st.sidebar.metric("메모리 사용량", f"{memory.percent}%")

# 푸터
st.markdown(f"""
<div style="text-align: center; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #E5E7EB; color: #9CA3AF; font-size: 0.875rem;">
    © 2025 지능형 노인 낙상 감지 시스템 | 현재 시간: {timestamp}
</div>
""", unsafe_allow_html=True)

