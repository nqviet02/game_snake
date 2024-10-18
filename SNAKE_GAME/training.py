import numpy as np
import random
import matplotlib.pyplot as plt
from snake_game import SnakeGameAI, Direction, Point, BLOCK_SIZE  
import os
import json  # Nhập thư viện json

class Training:
    def __init__(self, num_episodes=200, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 alpha=0.1, gamma=0.9):
        self.num_episodes = num_episodes
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.alpha = alpha
        self.gamma = gamma
        self.scores = []  # Danh sách lưu điểm số
        self.Q_table = np.zeros((2048, 3))  # Bảng Q

    def get_action(self, state_idx):
        """Chọn hành động dựa trên chính sách 𝜖-greedy."""
        if random.random() < self.epsilon:
            return random.randint(0, 2)  # Chọn hành động ngẫu nhiên
        else:
            return np.argmax(self.Q_table[state_idx])  # Chọn hành động tốt nhất từ bảng Q

    def update_Q_table(self, state_idx, action, reward, new_state_idx):
        """Cập nhật bảng Q dựa trên kinh nghiệm."""
        best_future_q = np.max(self.Q_table[new_state_idx])  # Giá trị Q tốt nhất cho trạng thái mới
        current_q = self.Q_table[state_idx, action]
        self.Q_table[state_idx, action] += self.alpha * (reward + self.gamma * best_future_q - current_q)

    def state_to_index(self, state):
        """Chuyển đổi trạng thái thành chỉ số để sử dụng trong bảng Q."""
        index = 0
        for i, val in enumerate(state):
            index += val * (2 ** i)
        return index

    def save_q_table(self, filename='q_table.json'):
        """Lưu bảng Q vào tệp JSON."""
        with open(filename, 'w') as f:
            json.dump(self.Q_table.tolist(), f)  # Chuyển đổi mảng NumPy thành list

    def load_q_table(self, filename='q_table.json'):
        """Tải bảng Q từ tệp JSON."""
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                self.Q_table = np.array(json.load(f))  # Chuyển đổi list thành mảng NumPy
            print(f"Kích thước của bảng Q đã tải: {self.Q_table.shape}")
            print("Một số giá trị trong bảng Q:")
            print(self.Q_table[:5])  # Hiển thị một phần của bảng Q để kiểm tra
        else:
            print("Không tìm thấy tệp 'q_table.json'.")

    def train(self):
        """Chạy quá trình huấn luyện AI."""
        game = SnakeGameAI()  # Khởi tạo game

        # Tải bảng Q nếu đã tồn tại
        self.load_q_table()

        for episode in range(self.num_episodes):
            game.reset()  # Đặt lại trò chơi
            state = game.get_state()  # Lấy trạng thái ban đầu
            state_idx = self.state_to_index(state)  # Chuyển đổi trạng thái thành chỉ số
            
            while True:
                action_idx = self.get_action(state_idx)  # Lấy hành động
                reward, game_over, score = game.play_step(action_idx)  # Thực hiện hành động và lấy phần thưởng
                new_state = game.get_state()  # Lấy trạng thái mới
                new_state_idx = self.state_to_index(new_state)  # Chuyển đổi trạng thái mới thành chỉ số
                
                # Cập nhật bảng Q
                self.update_Q_table(state_idx, action_idx, reward, new_state_idx)

                # Chuyển sang trạng thái mới
                state_idx = new_state_idx

                if game_over:
                    self.scores.append(score)  # Lưu điểm số
                    break
            
            # Giảm epsilon để giảm tần suất thăm dò
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Vẽ biểu đồ điểm số
            if episode % 10 == 0:  # Vẽ mỗi 10 tập
                plt.plot(self.scores)
                plt.xlabel('Episodes')
                plt.ylabel('Scores')
                plt.title('Training Progress')
                plt.pause(0.001)  # Thêm khoảng dừng để cập nhật biểu đồ
                plt.savefig('training_progress2.png') 
                print(f"Thư mục hiện tại: {os.getcwd()}")

        # Lưu Q-table sau khi huấn luyện
        self.save_q_table()
        print("Q-table đã được lưu vào tệp 'q_table.json'.")

if __name__ == "__main__":
    training = Training()
    training.train()  # Bắt đầu huấn luyện