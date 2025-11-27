"""
AI-Based Hands-Free HCI System
Features:
 - Hand-controlled mouse (smooth)
 - Pinch (thumb+index) -> left click
 - Index + middle -> right click
 - Fist -> drag (hold until open)
 - Two-finger vertical -> scroll
 - Eye-controlled drawing with blink toggle
 - Optional voice commands (requires pyaudio)
Controls:
 - m : toggle mode (hand_mouse / eye_draw)
 - c : clear canvas
 - v : toggle voice listening (optional)
 - s : save canvas screenshot
 - q : quit
"""

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import math
import time
import threading
import datetime

# Optional (voice)
try:
    import speech_recognition as sr
    VOICE_AVAILABLE = True
except Exception:
    VOICE_AVAILABLE = False

# ---------- SETTINGS ----------
SCREEN_W, SCREEN_H = pyautogui.size()
CAM_W, CAM_H = 640, 480
SMOOTHING = 10               # larger = smoother but slower
CLICK_THRESHOLD = 40         # pixel distance (camera space) for pinch click
RIGHTCLICK_THRESHOLD = 40
SCROLL_SENSITIVITY = 200     # mapping finger vertical movement to scroll
EAR_BLINK_THRESHOLD = 0.20
BLINK_CONSEC_FRAMES = 2

# ---------- Mediapipe init ----------
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.6,
                       min_tracking_confidence=0.6)

face_mesh = mp_face.FaceMesh(max_num_faces=1,
                            refine_landmarks=True,
                            min_detection_confidence=0.6,
                            min_tracking_confidence=0.6)

# Eye landmark indices (MediaPipe Face Mesh)
R_OUTER, R_INNER, R_TOP, R_BOTTOM = 33, 133, 159, 145
L_OUTER, L_INNER, L_TOP, L_BOTTOM = 362, 263, 386, 374

# ---------- Variables ----------
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0
mode = "hand_mouse"  # or "eye_draw"
pen_on = False
blink_frame_count = 0
draw_canvas = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
dragging = False
voice_listening = False
voice_thread = None
last_click_time = 0

# ---------- Utility functions ----------
def euclid(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def screen_from_cam(norm_x, norm_y):
    """
    Convert normalized landmark (0..1) to screen coordinates (pixels).
    norm_x/norm_y are from MediaPipe (camera mirrored later).
    """
    sx = int(norm_x * SCREEN_W)
    sy = int(norm_y * SCREEN_H)
    return sx, sy

def save_canvas(canvas):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"drawing_{ts}.png"
    cv2.imwrite(filename, canvas)
    print("[INFO] Canvas saved as", filename)

# ---------- Voice command handling (optional) ----------
def voice_listener():
    global voice_listening
    if not VOICE_AVAILABLE:
        print("[VOICE] SpeechRecognition not available.")
        return
    r = sr.Recognizer()
    mic = sr.Microphone()
    print("[VOICE] Listening started. Say commands like 'clear', 'switch mode', 'click', 'quit'.")
    with mic as source:
        r.adjust_for_ambient_noise(source, duration=1)
    while voice_listening:
        try:
            with mic as source:
                audio = r.listen(source, timeout=4, phrase_time_limit=4)
            cmd = r.recognize_google(audio).lower()
            print("[VOICE] Heard:", cmd)
            if "clear" in cmd:
                clear_canvas()
            elif "switch" in cmd or "mode" in cmd:
                toggle_mode()
            elif "click" in cmd:
                pyautogui.click()
            elif "quit" in cmd or "exit" in cmd:
                voice_listening = False
                break
        except Exception as e:
            # timeout or recognition error
            pass
    print("[VOICE] Listening stopped.")

def start_voice_thread():
    global voice_thread, voice_listening
    if VOICE_AVAILABLE:
        voice_listening = True
        voice_thread = threading.Thread(target=voice_listener, daemon=True)
        voice_thread.start()
    else:
        print("[VOICE] Not available (pyaudio/SpeechRecognition missing).")

def stop_voice_thread():
    global voice_listening
    voice_listening = False

# ---------- Helpers ----------
def clear_canvas():
    global draw_canvas
    draw_canvas = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    print("[INFO] Canvas cleared.")

def toggle_mode():
    global mode
    mode = "eye_draw" if mode == "hand_mouse" else "hand_mouse"
    print(f"[INFO] Mode switched to: {mode}")

# ---------- Main loop ----------
cap = cv2.VideoCapture(0)
cap.set(3, CAM_W)
cap.set(4, CAM_H)

print("Controls: m - toggle mode, c - clear canvas, v - toggle voice, s - save canvas, q - quit")
time.sleep(0.5)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Cannot read camera.")
            break

        frame = cv2.flip(frame, 1)  # mirror for natural control
        frame_h, frame_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ---- HAND PROCESSING ----
        hand_results = hands.process(rgb)
        if hand_results.multi_hand_landmarks and mode == "hand_mouse":
            hand_land = hand_results.multi_hand_landmarks[0]
            # tip positions in camera coords
            lm_index = hand_land.landmark[8]   # index fingertip
            lm_thumb = hand_land.landmark[4]   # thumb tip
            lm_middle = hand_land.landmark[12] # middle fingertip
            lm_wrist = hand_land.landmark[0]

            # camera coordinates (pixels) for pinch distance measurement
            ix, iy = int(lm_index.x * frame_w), int(lm_index.y * frame_h)
            tx, ty = int(lm_thumb.x * frame_w), int(lm_thumb.y * frame_h)
            mx, my = int(lm_middle.x * frame_w), int(lm_middle.y * frame_h)

            # map normalized to screen coords and smooth
            screen_x, screen_y = screen_from_cam(lm_index.x, lm_index.y)
            curr_x = prev_x + (screen_x - prev_x) / SMOOTHING
            curr_y = prev_y + (screen_y - prev_y) / SMOOTHING

            # move mouse
            try:
                pyautogui.moveTo(curr_x, curr_y)
            except Exception:
                pass

            # compute distances (camera-space)
            pinch_dist = euclid((ix, iy), (tx, ty))
            idx_mid_dist = euclid((ix, iy), (mx, my))
            # visual feedback
            cv2.circle(frame, (ix, iy), 6, (0, 255, 0), -1)
            cv2.circle(frame, (tx, ty), 6, (0, 0, 255), -1)
            cv2.putText(frame, f"P:{int(pinch_dist)} M:{int(idx_mid_dist)}", (10,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # ----- CLICK (pinch) -----
            # Avoid repeated clicks: small debounce
            now = time.time()
            if pinch_dist < CLICK_THRESHOLD and (now - last_click_time) > 0.3:
                pyautogui.click()
                last_click_time = now
                cv2.putText(frame, "CLICK", (ix+10, iy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            # ----- RIGHT CLICK (index + middle close) -----
            if idx_mid_dist < RIGHTCLICK_THRESHOLD and (now - last_click_time) > 0.5:
                pyautogui.click(button='right')
                last_click_time = now
                cv2.putText(frame, "RCLICK", (ix+10, iy-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2)

            # ----- DRAG MODE: fist detection (all fingers folded)
            # naive method: check distance between index tip and wrist small -> fist
            wrist_x, wrist_y = int(lm_wrist.x * frame_w), int(lm_wrist.y * frame_h)
            idx_wrist_dist = euclid((ix, iy), (wrist_x, wrist_y))
            if idx_wrist_dist < 60 and not dragging:
                # start dragging: mouseDown
                pyautogui.mouseDown()
                dragging = True
                cv2.putText(frame, "DRAG START", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            elif idx_wrist_dist >= 60 and dragging:
                pyautogui.mouseUp()
                dragging = False
                cv2.putText(frame, "DRAG END", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            # ----- SCROLL: two-finger vertical movement
            # Use thumb & middle vertical difference as a simple scroll control
            # If both index and middle are extended and vertical separation changes -> scroll
            # We compute vertical delta of middle vs index and map to scroll
            scroll_amount = int((iy - my) / 10)  # adjust divisor to control sensitivity
            if abs(scroll_amount) > 5:
                try:
                    pyautogui.scroll(-scroll_amount)  # negative to align directions
                    cv2.putText(frame, f"SCROLL {scroll_amount}", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)
                except Exception:
                    pass

            mp_draw.draw_landmarks(frame, hand_land, mp_hands.HAND_CONNECTIONS)
            prev_x, prev_y = curr_x, curr_y

        # ---- FACE / EYE PROCESSING ----
        face_results = face_mesh.process(rgb)
        if face_results.multi_face_landmarks:
            face_land = face_results.multi_face_landmarks[0]

            # get eye landmarks
            r_outer = face_land.landmark[R_OUTER]
            r_inner = face_land.landmark[R_INNER]
            r_top = face_land.landmark[R_TOP]
            r_bottom = face_land.landmark[R_BOTTOM]

            l_outer = face_land.landmark[L_OUTER]
            l_inner = face_land.landmark[L_INNER]
            l_top = face_land.landmark[L_TOP]
            l_bottom = face_land.landmark[L_BOTTOM]

            # compute EAR per eye
            def ear(top, bottom, left, right):
                v = math.hypot((top.x - bottom.x) * frame_w, (top.y - bottom.y) * frame_h)
                h = math.hypot((left.x - right.x) * frame_w, (left.y - right.y) * frame_h)
                if h == 0:
                    return 1.0
                return v / h

            ear_r = ear(r_top, r_bottom, r_outer, r_inner)
            ear_l = ear(l_top, l_bottom, l_outer, l_inner)
            ear_avg = (ear_r + ear_l) / 2.0

            # eye center
            cx = (r_top.x + r_bottom.x + r_outer.x + r_inner.x + l_top.x + l_bottom.x + l_outer.x + l_inner.x) / 8.0
            cy = (r_top.y + r_bottom.y + r_outer.y + r_inner.y + l_top.y + l_bottom.y + l_outer.y + l_inner.y) / 8.0

            # convert to canvas coordinates (camera sized)
            draw_x = int(cx * CAM_W)
            draw_y = int(cy * CAM_H)

            # blink detection to toggle pen
            if ear_avg < EAR_BLINK_THRESHOLD:
                blink_frame_count += 1
            else:
                if blink_frame_count >= BLINK_CONSEC_FRAMES:
                    pen_on = not pen_on
                    print(f"[EYE] Pen toggled: {pen_on}")
                blink_frame_count = 0

            cv2.putText(frame, f"EAR:{ear_avg:.2f} PEN:{'ON' if pen_on else 'OFF'}", (10,130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

            # eye drawing: only when in eye_draw mode
            if mode == "eye_draw":
                if pen_on:
                    # draw on canvas (camera coords)
                    cv2.circle(draw_canvas, (draw_x, draw_y), 4, (0,0,255), -1)
                # show marker on camera feed
                cv2.circle(frame, (int(cx*frame_w), int(cy*frame_h)), 6, (0,0,255), -1)

        # Overlay canvas to the right of camera (show both)
        canvas_resized = cv2.resize(draw_canvas, (frame_w, frame_h))
        combined = np.hstack([frame, canvas_resized])

        cv2.putText(combined, f"MODE: {mode}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(combined, "m:mode, c:clear, v:voice, s:save, q:quit", (10, frame_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        cv2.imshow("AI-HCI System (Left: Camera, Right: Drawing Canvas)", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            toggle_mode()
            time.sleep(0.25)
        elif key == ord('c'):
            clear_canvas()
        elif key == ord('s'):
            save_canvas(draw_canvas)
        elif key == ord('v'):
            # toggle voice
            if VOICE_AVAILABLE:
                if not voice_listening:
                    start_voice_thread()
                    print("[VOICE] On")
                else:
                    stop_voice_thread()
                    print("[VOICE] Off")
            else:
                print("[VOICE] Not available. Install pyaudio + SpeechRecognition for voice features.")

finally:
    # cleanup
    stop_voice_thread()
    cap.release()
    cv2.destroyAllWindows()
