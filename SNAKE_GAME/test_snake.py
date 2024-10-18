import numpy as np
from snake_game import SnakeGameAI  # Đảm bảo bạn đã import trò chơi

class Play:
    def __init__(self, q_table_file='q_table.json'):
        self.Q_table = self.load_q_table(q_table_file)
        self.game = SnakeGameAI()  # Khởi tạo trò chơi

    def load_q_table(self, filename):
        """Tải bảng Q từ tệp JSON."""
        import json
        with open(filename, 'r') as f:
            q_table = json.load(f)
        return np.array(q_table)

    def state_to_index(self, state):
        """Chuyển đổi trạng thái thành chỉ số để sử dụng trong bảng Q."""
        index = 0
        for i, val in enumerate(state):
            index += val * (2 ** i)
        return index

    def play_one_game(self):
        """Chạy một trò chơi sử dụng bảng Q đã tải."""
        self.game.reset()  # Đặt lại trò chơi
        state = self.game.get_state()  # Lấy trạng thái ban đầu
        state_idx = self.state_to_index(state)  # Chuyển đổi trạng thái thành chỉ số

        while True:
            # Lấy hành động tốt nhất từ bảng Q
            action_idx = np.argmax(self.Q_table[state_idx])  # Chọn hành động tốt nhất
            reward, game_over, score = self.game.play_step(action_idx)  # Thực hiện hành động
            
            # Lấy trạng thái mới
            new_state = self.game.get_state()  # Lấy trạng thái mới
            new_state_idx = self.state_to_index(new_state)  # Chuyển đổi trạng thái mới thành chỉ số

            # Cập nhật trạng thái
            state_idx = new_state_idx
            
            if game_over:
                print("Trò chơi kết thúc với điểm số:", score)
                break

    def play_multiple_games(self, num_games=5):
        """Chạy nhiều trò chơi liên tiếp."""
        for i in range(num_games):
            print(f"Bắt đầu trò chơi thứ {i + 1}...")
            self.play_one_game()  # Chơi một trò chơi
            print()  # Dòng trống giữa các trò chơi

if __name__ == "__main__":
    game_play = Play()
    game_play.play_multiple_games(5)  # Bắt đầu chơi 5 trò chơi
