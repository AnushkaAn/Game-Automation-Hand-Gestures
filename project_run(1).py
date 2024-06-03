import cv2
import mediapipe as mp
import pygame
import random
import math

# Initialize Pygames
#this is the game
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
PLAYER_SPEED = 7  # Increased speed for better responsiveness
OBSTACLE_SPEED = 7  # Increased speed for faster gameplay
MAX_OBSTACLES = 6  # Maximum number of obstacles on screen at once
OBSTACLE_SPAWN_CHANCE = 10  # Probability of spawning a new obstacle
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Running Man Game")
FONT = pygame.font.Font(None, 36)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Global variables
player_x = SCREEN_WIDTH // 2 - PLAYER_WIDTH // 2
player_y = SCREEN_HEIGHT - PLAYER_HEIGHT
obstacle_list = []
score = 0
game_over_state = False


def create_obstacle():
    obstacle_type = random.choice(["square", "circle", "triangle"])
    obstacle_x = random.randint(0, SCREEN_WIDTH - 50)
    obstacle_y = -50
    return [obstacle_x, obstacle_y, 50, 50, obstacle_type]


def handle_pygame_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()


def draw_player():
    pygame.draw.rect(SCREEN, RED, (player_x, player_y, PLAYER_WIDTH, PLAYER_HEIGHT))


def draw_obstacles():
    for obstacle in obstacle_list:
        x, y, w, h, obstacle_type = obstacle
        if obstacle_type == "square":
            pygame.draw.rect(SCREEN, BLACK, (x, y, w, h))
        elif obstacle_type == "circle":
            pygame.draw.circle(SCREEN, BLACK, (x + w // 2, y + h // 2), w // 2)
        elif obstacle_type == "triangle":
            pygame.draw.polygon(
                SCREEN,
                BLACK,
                [
                    (x + w // 2, y),
                    (x, y + h),
                    (x + w, y + h),
                ],
            )


def move_obstacles():
    global obstacle_list, score
    to_remove = []
    for obstacle in obstacle_list:
        obstacle[1] += OBSTACLE_SPEED
        if obstacle[1] > SCREEN_HEIGHT:
            score += 1
            to_remove.append(obstacle)

    for obstacle in to_remove:
        obstacle_list.remove(obstacle)


def check_collisions():
    for obstacle in obstacle_list:
        if (
            player_x < obstacle[0] + obstacle[2]
            and player_x + PLAYER_WIDTH > obstacle[0]
            and player_y < obstacle[1] + obstacle[3]
            and player_y + PLAYER_HEIGHT > obstacle[1]
        ):
            return True
    return False


def draw_score():
    score_text = FONT.render(f"Score: {score}", True, BLACK)
    SCREEN.blit(score_text, (10, 10))


def game_over():
    global score, obstacle_list, player_x, player_y, game_over_state

    game_over_state = True

    while game_over_state:
        SCREEN.fill(WHITE)
        game_over_text = FONT.render("Game Over", True, BLACK)
        score_text = FONT.render(f"Score: {score}", True, BLACK)
        replay_text = FONT.render("Replay", True, BLACK)
        quit_text = FONT.render("Quit", True, BLACK)

        game_over_pos = (
            SCREEN_WIDTH // 2 - game_over_text.get_width() // 2,
            SCREEN_HEIGHT // 2,
        )
        score_pos = (
            SCREEN_WIDTH // 2 - score_text.get_width() // 2,
            SCREEN_HEIGHT // 2 - 50,
        )
        replay_pos = (
            SCREEN_WIDTH // 2 - replay_text.get_width() // 2,
            SCREEN_HEIGHT // 2 + 50,
        )
        quit_pos = (
            SCREEN_WIDTH // 2 - quit_text.get_width() // 2,
            SCREEN_HEIGHT // 2 + 100,
        )

        SCREEN.blit(game_over_text, game_over_pos)
        SCREEN.blit(score_text, score_pos)
        SCREEN.blit(replay_text, replay_pos)
        SCREEN.blit(quit_text, quit_pos)

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()

                if (
                    replay_pos[0] <= mouse_x <= replay_pos[0] + replay_text.get_width()
                    and replay_pos[1] <= mouse_y <= replay_pos[1] + replay_text.get_height()
                ):
                    score = 0
                    obstacle_list = []
                    player_x = SCREEN_WIDTH // 2 - PLAYER_WIDTH // 2
                    game_over_state = False  # Restart the game

                elif (
                    quit_pos[0] <= mouse_x <= quit_pos[0] + quit_text.get_width()
                    and quit_pos[1] <= mouse_y <= quit_pos[1] + quit_text.get_height()
                ):
                    pygame.quit()
                    quit()


def main_game():
    global player_x, player_y, game_over_state

    # Initialize camera and MediaPipe
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        average_position = (player_x, player_y)  # Starting position

        while cap.isOpened() and not game_over_state:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Flip the frame to mirror the player's perspective
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get the index finger tip position
                    index_finger_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                    height, width, _ = frame.shape
                    curr_index_finger_position = (
                        int(index_finger_landmark.x * width),
                        int(index_finger_landmark.y * height),
                    )

                    # Apply smoothing for a more stable player movement
                    smoothing_factor = 0.3  # Adjusted for faster response
                    average_position = (
                        int(average_position[0] * (1 - smoothing_factor) + curr_index_finger_position[0] * smoothing_factor),
                        int(average_position[1] * (1 - smoothing_factor) + curr_index_finger_position[1] * smoothing_factor),
                    )

                    # Determine player movement based on finger position
                    if average_position[0] < SCREEN_WIDTH // 2:
                        player_x -= PLAYER_SPEED
                    else:
                        player_x += PLAYER_SPEED

                    player_x = max(0, min(player_x, SCREEN_WIDTH - PLAYER_WIDTH))

            # Game logic
            SCREEN.fill(WHITE)  # Clear the screen
            handle_pygame_events()  # Check for Pygame events

            # Limit the number of obstacles on screen
            if len(obstacle_list) < MAX_OBSTACLES:
                if random.randint(0, 100) < OBSTACLE_SPAWN_CHANCE:
                    obstacle_list.append(create_obstacle())

            # Move obstacles and check for collisions
            move_obstacles()
            if check_collisions():
                game_over()

            draw_player()  # Draw player
            draw_obstacles()  # Draw obstacles
            draw_score()  # Draw the score

            pygame.display.update()  # Update the game screen

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main_game()
    pygame.quit()
