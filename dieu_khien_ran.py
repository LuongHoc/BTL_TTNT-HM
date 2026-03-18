import cv2
import mediapipe as mp
import numpy as np
import pygame
import joblib
import random

# ===== Khoi tao model AI =====
model = joblib.load("mo_hinh/svm_tay.pkl")

# ===== Khoi tao Mediapipe =====
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# ===== Khoi tao webcam =====
cap = cv2.VideoCapture(0)

# ===== Khoi tao Pygame =====
pygame.init()

WIDTH = 600
HEIGHT = 600

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake AI Control")

font = pygame.font.SysFont("Segoe UI", 24, bold=True)
small_font = pygame.font.SysFont("Segoe UI", 20)
big_font = pygame.font.SysFont("Segoe UI", 58, bold=True)

# ===== Bang mau giao dien =====
BG_TOP = (18, 24, 48)
BG_BOTTOM = (9, 12, 24)
GRID = (30, 38, 66)
SNAKE_HEAD = (105, 255, 170)
SNAKE_BODY = (42, 210, 140)
FOOD = (255, 98, 98)
FOOD_HIGHLIGHT = (255, 175, 175)
TEXT = (255,255,255)
TEXT_SOFT = (190, 205, 235)
BUTTON = (79, 142, 232)
BUTTON_HOVER = (109, 167, 245)
PANEL_BG = (13, 18, 38)
PANEL_BORDER = (70, 92, 150)

snake_size = 20
TARGET_FPS = 60
GAME_SPEED = 7  # so buoc di chuyen moi giay cua ran
MOVE_INTERVAL_MS = 1000.0 / GAME_SPEED
clock = pygame.time.Clock()

gesture_text = "None"
fps_value = 0.0
frame_time_ms = 0.0
movement_accumulator_ms = 0.0

GESTURE_TO_DIRECTION = {
    "trai": "LEFT",
    "phai": "RIGHT",
    "len": "UP",
    "xuong": "DOWN"
}


# ===== Ham khoi tao lai trang thai game =====
def reset_game():

    snake = [[300,300]]
    direction = "RIGHT"
    food = [random.randrange(0,WIDTH,20), random.randrange(0,HEIGHT,20)]
    score = 0

    return snake, direction, food, score


snake, direction, food, score = reset_game()
game_over = False

OPPOSITE_DIRECTION = {
    "LEFT": "RIGHT",
    "RIGHT": "LEFT",
    "UP": "DOWN",
    "DOWN": "UP"
}


# ===== Ham su dung model de du doan cu chi =====
def predict_gesture(landmarks):

    data = []

    for lm in landmarks:
        data.append(lm.x)
        data.append(lm.y)

    data = np.array(data).reshape(1,-1)

    pred = model.predict(data)

    return pred[0]


def cap_nhat_huong_tu_gesture(gesture, current_direction):
    # Chuyen nhan cu chi sang huong va chan quay dau 180 do
    new_direction = GESTURE_TO_DIRECTION.get(gesture)

    if new_direction and OPPOSITE_DIRECTION[current_direction] != new_direction:
        return new_direction

    return current_direction


# ===== Ham ve cac thanh phan giao dien =====
def draw_snake():

    for i, block in enumerate(snake):
        color = SNAKE_HEAD if i == 0 else SNAKE_BODY
        rect = pygame.Rect(block[0], block[1], snake_size, snake_size)
        pygame.draw.rect(screen, color, rect, border_radius=6)

    # mắt cho đầu rắn để tạo điểm nhấn
    head = snake[0]
    eye_color = (15, 30, 20)
    if direction in ["RIGHT", "LEFT"]:
        y_eye = head[1] + 6
        x1 = head[0] + (14 if direction == "RIGHT" else 6)
        x2 = head[0] + (14 if direction == "RIGHT" else 6)
        pygame.draw.circle(screen, eye_color, (x1, y_eye), 2)
        pygame.draw.circle(screen, eye_color, (x2, y_eye + 7), 2)
    else:
        x_eye = head[0] + 6
        y1 = head[1] + (6 if direction == "UP" else 14)
        y2 = head[1] + (6 if direction == "UP" else 14)
        pygame.draw.circle(screen, eye_color, (x_eye, y1), 2)
        pygame.draw.circle(screen, eye_color, (x_eye + 7, y2), 2)


def draw_food():

    center = (food[0] + snake_size // 2, food[1] + snake_size // 2)
    pygame.draw.circle(screen, FOOD, center, snake_size // 2 - 1)
    pygame.draw.circle(screen, FOOD_HIGHLIGHT, (center[0] - 4, center[1] - 4), 3)


def draw_score():

    text = font.render("Score: " + str(score), True, TEXT)
    screen.blit(text,(18,16))


def draw_gesture():

    text = small_font.render("Gesture: " + gesture_text, True, TEXT_SOFT)
    screen.blit(text,(18,48))


def draw_performance():

    perf_text = small_font.render(
        f"FPS: {fps_value:4.1f} | Frame: {frame_time_ms:5.1f} ms | Speed: {GAME_SPEED} step/s",
        True,
        TEXT_SOFT
    )
    screen.blit(perf_text, (18, 72))


def draw_background():

    # nền gradient theo chiều dọc
    for y in range(HEIGHT):
        ratio = y / HEIGHT
        r = int(BG_TOP[0] * (1 - ratio) + BG_BOTTOM[0] * ratio)
        g = int(BG_TOP[1] * (1 - ratio) + BG_BOTTOM[1] * ratio)
        b = int(BG_TOP[2] * (1 - ratio) + BG_BOTTOM[2] * ratio)
        pygame.draw.line(screen, (r, g, b), (0, y), (WIDTH, y))

    # lưới nhẹ để tạo cảm giác board game
    for x in range(0, WIDTH, snake_size):
        pygame.draw.line(screen, GRID, (x, 0), (x, HEIGHT), 1)
    for y in range(0, HEIGHT, snake_size):
        pygame.draw.line(screen, GRID, (0, y), (WIDTH, y), 1)


def draw_hud_panel():

    panel = pygame.Rect(10, 8, 320, 102)
    pygame.draw.rect(screen, PANEL_BG, panel, border_radius=12)
    pygame.draw.rect(screen, PANEL_BORDER, panel, width=2, border_radius=12)


def draw_restart_button():

    button_rect = pygame.Rect(WIDTH//2-90, HEIGHT//2+44, 180, 52)
    is_hover = button_rect.collidepoint(pygame.mouse.get_pos())

    pygame.draw.rect(screen, (5, 8, 16), button_rect.move(0, 4), border_radius=14)
    pygame.draw.rect(screen, BUTTON_HOVER if is_hover else BUTTON, button_rect, border_radius=14)
    pygame.draw.rect(screen, (190, 220, 255), button_rect, width=2, border_radius=14)

    text = font.render("Choi lai", True, (255,255,255))
    text_rect = text.get_rect(center=button_rect.center)
    screen.blit(text, text_rect)


def cap_nhat_logic_game():
    # Cap nhat vi tri ran theo GAME_SPEED va xu ly va cham
    global snake, food, score, game_over, movement_accumulator_ms

    if game_over or movement_accumulator_ms < MOVE_INTERVAL_MS:
        return

    movement_accumulator_ms -= MOVE_INTERVAL_MS
    head = snake[0].copy()

    if direction == "LEFT":
        head[0] -= snake_size
    if direction == "RIGHT":
        head[0] += snake_size
    if direction == "UP":
        head[1] -= snake_size
    if direction == "DOWN":
        head[1] += snake_size

    snake.insert(0, head)

    # Neu an duoc food thi tang diem, nguoc lai cat duoi
    if head == food:
        score += 1
        food = [random.randrange(0, WIDTH, snake_size), random.randrange(0, HEIGHT, snake_size)]
    else:
        snake.pop()

    # Kiem tra va cham tuong/than
    if head[0] < 0 or head[0] >= WIDTH or head[1] < 0 or head[1] >= HEIGHT:
        game_over = True
    if head in snake[1:]:
        game_over = True


def ve_game():
    # Ve toan bo scene game va man hinh game over neu co
    draw_background()
    draw_hud_panel()
    draw_snake()
    draw_food()
    draw_score()
    draw_gesture()
    draw_performance()

    if game_over:
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((3, 5, 10, 135))
        screen.blit(overlay, (0, 0))

        text = big_font.render("GAME OVER", True, (255,95,95))
        screen.blit(text,(WIDTH//2-165, HEIGHT//2-65))

        sub_text = small_font.render("Nhan vao nut de choi tiep", True, TEXT_SOFT)
        screen.blit(sub_text, (WIDTH//2-105, HEIGHT//2+5))

        draw_restart_button()


def xu_ly_su_kien():
    # Xu ly su kien thoat game va bam nut choi lai
    global running, snake, direction, food, score, game_over, movement_accumulator_ms

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN and game_over:
            mouse = pygame.mouse.get_pos()

            if WIDTH//2-90 < mouse[0] < WIDTH//2+90 and HEIGHT//2+44 < mouse[1] < HEIGHT//2+96:
                snake, direction, food, score = reset_game()
                game_over = False
                movement_accumulator_ms = 0.0


running = True


while running:

    # Nhip render/doc webcam va tinh thoi gian frame hien tai
    delta_ms = clock.tick(TARGET_FPS)
    frame_time_ms = float(delta_ms)
    fps_value = clock.get_fps()
    movement_accumulator_ms += delta_ms

    # ===== 1) Lay frame webcam va nhan dien cu chi =====

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)


    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # nhận diện tay
    if results.multi_hand_landmarks and not game_over:

        for hand_landmarks in results.multi_hand_landmarks:

            gesture = predict_gesture(hand_landmarks.landmark)
            gesture_text = gesture

            direction = cap_nhat_huong_tu_gesture(gesture, direction)


    # Hien thi ket qua cu chi tren cua so webcam
    cv2.putText(frame,
                "Gesture: " + gesture_text,
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)

    cv2.imshow("AI Hand Control", frame)
    cv2.waitKey(1)


    # ===== 2) Cap nhat logic game =====
    cap_nhat_logic_game()

    # ===== 3) Ve toan bo man hinh game =====
    ve_game()


    pygame.display.update()


    # ===== 4) Xu ly su kien cua so =====
    xu_ly_su_kien()


pygame.quit()

cap.release()
cv2.destroyAllWindows()