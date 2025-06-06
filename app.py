import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import random
import pygame
from PIL import Image
import time

# Initialize Pygame
pygame.init()
pygame.font.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
PLAYER_WIDTH = 50
PLAYER_HEIGHT = 50
PLAYER_SPEED = 10
OBSTACLE_SPEED = 7
MAX_OBSTACLES = 6
OBSTACLE_SPAWN_CHANCE = 15

FONT = pygame.font.SysFont('Arial', 36)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Streamlit setup
st.set_page_config(layout="wide")
st.title("🕹️ Running Man Game - Hand Gesture Controlled")

# Initialize webcam in session state
def init_webcam():
    if "cap" in st.session_state and st.session_state.cap.isOpened():
        st.session_state.cap.release()
    st.session_state.cap = cv2.VideoCapture(0)
    if not st.session_state.cap.isOpened():
        st.error("Unable to access webcam. Please ensure your webcam is connected and try again.")
        st.stop()

if "cap" not in st.session_state:
    init_webcam()

# Session state initialization
def init_game_state():
    st.session_state.player_x = SCREEN_WIDTH // 2 - PLAYER_WIDTH // 2
    st.session_state.player_y = SCREEN_HEIGHT - PLAYER_HEIGHT - 20
    st.session_state.obstacle_list = []
    st.session_state.score = 0
    st.session_state.game_over_state = False
    st.session_state.last_update_time = time.time()
    st.session_state.hand_position = None
    st.session_state.frame = None
    st.session_state.last_frame_time = 0

if "player_x" not in st.session_state:
    init_game_state()

# Utility: Reset game
def reset_game():
    init_webcam()  # Reinitialize webcam on reset
    init_game_state()

# Utility: Quit game
def quit_game():
    st.session_state.game_over_state = True
    if "cap" in st.session_state and st.session_state.cap.isOpened():
        st.session_state.cap.release()

# UI buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("🔄 Replay"):
        reset_game()
with col2:
    if st.button("❌ Quit"):
        quit_game()

# Obstacle creation
def create_obstacle():
    obstacle_type = random.choice(["square", "circle", "triangle"])
    obstacle_width = random.randint(40, 70)
    obstacle_height = obstacle_width
    obstacle_x = random.randint(0, SCREEN_WIDTH - obstacle_width)
    obstacle_y = -obstacle_height
    return [obstacle_x, obstacle_y, obstacle_width, obstacle_height, obstacle_type]

# Move obstacles
def move_obstacles():
    to_remove = []
    for obs in st.session_state.obstacle_list:
        obs[1] += OBSTACLE_SPEED
        if obs[1] > SCREEN_HEIGHT:
            st.session_state.score += 1
            to_remove.append(obs)
    for obs in to_remove:
        st.session_state.obstacle_list.remove(obs)

# Collision check
def check_collisions():
    px = st.session_state.player_x
    py = st.session_state.player_y
    pw = PLAYER_WIDTH
    ph = PLAYER_HEIGHT
    
    for ox, oy, ow, oh, otype in st.session_state.obstacle_list:
        if (px < ox + ow and px + pw > ox and 
            py < oy + oh and py + ph > oy):
            return True
    return False

# Drawing functions
def draw_player(surface):
    pygame.draw.rect(surface, RED, 
                    (st.session_state.player_x, 
                     st.session_state.player_y, 
                     PLAYER_WIDTH, PLAYER_HEIGHT))

def draw_obstacles(surface):
    for x, y, w, h, t in st.session_state.obstacle_list:
        if t == "square":
            pygame.draw.rect(surface, BLACK, (x, y, w, h))
        elif t == "circle":
            pygame.draw.circle(surface, BLACK, (x + w // 2, y + h // 2), w // 2)
        elif t == "triangle":
            pygame.draw.polygon(surface, BLACK, 
                               [(x + w // 2, y), 
                                (x, y + h), 
                                (x + w, y + h)])

def draw_score(surface):
    text_surface = FONT.render(f"Score: {st.session_state.score}", True, BLACK)
    surface.blit(text_surface, (10, 10))

def draw_hand_position(surface):
    if st.session_state.hand_position:
        pygame.draw.circle(surface, (0, 255, 0), 
                          st.session_state.hand_position, 10)

# Main game function
def run_game_frame():
    # Limit frame processing to 30 FPS for performance
    current_time = time.time()
    if current_time - st.session_state.last_frame_time < 0.033:  # ~30 FPS
        return
    
    st.session_state.last_frame_time = current_time
    
    # Read frame from webcam
    ret, frame = st.session_state.cap.read()
    if not ret:
        st.warning("Webcam disconnected. Trying to reconnect...")
        init_webcam()
        return
    
    # Process frame
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process hand landmarks
    results = hands.process(rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get index finger tip coordinates
            tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # Convert to screen coordinates
            screen_x = int(tip.x * SCREEN_WIDTH)
            screen_y = int(tip.y * SCREEN_HEIGHT)
            
            # Store hand position for visualization
            st.session_state.hand_position = (screen_x, screen_y)
            
            # Smooth movement (weighted average)
            smoothing_factor = 0.3
            new_x = int(st.session_state.player_x * (1 - smoothing_factor) + 
                       screen_x * smoothing_factor)
            
            # Update player position
            st.session_state.player_x = max(0, min(new_x, SCREEN_WIDTH - PLAYER_WIDTH))
    
    # Game logic
    if current_time - st.session_state.last_update_time > 0.5:
        if len(st.session_state.obstacle_list) < MAX_OBSTACLES:
            if random.randint(0, 100) < OBSTACLE_SPAWN_CHANCE:
                st.session_state.obstacle_list.append(create_obstacle())
        st.session_state.last_update_time = current_time
    
    move_obstacles()
    
    if check_collisions():
        st.session_state.game_over_state = True
    
    # Draw game frame
    surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    surface.fill(WHITE)
    draw_player(surface)
    draw_obstacles(surface)
    draw_score(surface)
    draw_hand_position(surface)
    
    # Convert to PIL Image for Streamlit
    img_data = pygame.image.tostring(surface, "RGB")
    img = Image.frombytes("RGB", (SCREEN_WIDTH, SCREEN_HEIGHT), img_data)
    
    # Display webcam feed and game side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(frame, channels="BGR", caption="Webcam Feed (Point with index finger)")
    with col2:
        st.image(img, caption="Game Screen")

# Main game loop
if st.session_state.game_over_state:
    st.markdown("## ❌ Game Over!")
    st.markdown(f"### Final Score: `{st.session_state.score}`")
    st.markdown("Click **Replay** to restart or **Quit** to exit.")
else:
    run_game_frame()
    st.rerun()

# Release resources when the app is closed
if "cap" in st.session_state and st.session_state.cap.isOpened():
    st.session_state.cap.release()