import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font(None, 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

BLOCK_SIZE = 20
SPEED = 20

# Snake Environment
class SnakeGameAI:
    
    def __init__(self, w=320, h=240):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.reset()
        self.clock = pygame.time.Clock()
        self.frame_iteration = 0

#khởi tạo thân rắn
    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [self.head, 
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

#Khởi tạo thức ăn
    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

#các bước tiếp theo
    def play_step(self, action):
        self.frame_iteration += 1
    
    # Kiểm tra nếu có sự kiện thoát game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
               pygame.quit()
               quit()

    # Thực hiện các hành động và logic của game
        self._move(action) 
        self.snake.insert(0, self.head)

        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
           game_over = True
           reward = -10
           return reward, game_over, self.score

        if self.head == self.food:
           self.score += 1
           reward = 10
           self._place_food()
        else:
           self.snake.pop()

        self._update_ui()
        self.clock.tick(SPEED)

        return reward, game_over, self.score


#cập nhật giao diện
    def _update_ui(self):
        self.display.fill((0, 0, 0))

        for pt in self.snake:
            pygame.draw.rect(self.display, (0, 255, 0), pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))

        pygame.draw.rect(self.display, (255, 0, 0), pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, (255, 255, 255))
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if action == 0:  # move straight
            new_dir = clock_wise[idx]
        elif action == 1:  # move right
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:  # move left
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE

        self.head = Point(x, y)

#kiểm tra va chạm, khi gọi hàm mà ko truyền tham số, hàm sẽ lấy vị trí đầu rắn
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Va chạm tường
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Va chạm thân rắn
        if pt in self.snake[1:]:
            return True

        return False

    def get_state(self):
        head = self.snake[0]
        
        # Các vị trí xung quanh đầu rắn
        point_l = Point(head.x - BLOCK_SIZE, head.y)  # Trái
        point_r = Point(head.x + BLOCK_SIZE, head.y)  # Phải
        point_u = Point(head.x, head.y - BLOCK_SIZE)  # Trên
        point_d = Point(head.x, head.y + BLOCK_SIZE)  # Dưới
        
        # Kiểm tra va chạm với tường hoặc thân rắn
        danger_straight = (self.direction == Direction.RIGHT and self.is_collision(point_r)) or \
                          (self.direction == Direction.LEFT and self.is_collision(point_l)) or \
                          (self.direction == Direction.UP and self.is_collision(point_u)) or \
                          (self.direction == Direction.DOWN and self.is_collision(point_d))
        
        danger_right = (self.direction == Direction.UP and self.is_collision(point_r)) or \
                       (self.direction == Direction.DOWN and self.is_collision(point_l)) or \
                       (self.direction == Direction.LEFT and self.is_collision(point_u)) or \
                       (self.direction == Direction.RIGHT and self.is_collision(point_d))
        
        danger_left = (self.direction == Direction.DOWN and self.is_collision(point_r)) or \
                      (self.direction == Direction.UP and self.is_collision(point_l)) or \
                      (self.direction == Direction.RIGHT and self.is_collision(point_u)) or \
                      (self.direction == Direction.LEFT and self.is_collision(point_d))
        
        # Vị trí của thức ăn 
        food_left = self.food.x < self.head.x
        food_right = self.food.x > self.head.x
        food_up = self.food.y < self.head.y
        food_down = self.food.y > self.head.y
        
        # Trả về trạng thái dưới dạng mảng nhị phân
        state = [
            # Nguy hiểm xung quanh
            danger_straight,
            danger_right,
            danger_left,

            # Hướng hiện tại
            self.direction == Direction.LEFT,
            self.direction == Direction.RIGHT,
            self.direction == Direction.UP,
            self.direction == Direction.DOWN,

            # Vị trí thức ăn
            food_left,
            food_right,
            food_up,
            food_down
        ]

        return np.array(state, dtype=int)