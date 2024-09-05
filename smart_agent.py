import numpy as np

class GomokuAgent:
    def __init__(self, agent_symbol, blank_symbol, opponent_symbol):
        self.name = "Smart Agent"
        self.agent_symbol = agent_symbol
        self.blank_symbol = blank_symbol
        self.opponent_symbol = opponent_symbol
        self.first_move = True  # Track if it's the agent's first move

    def play(self, board):
        if self.first_move:
            self.first_move = False
            # Check if the board is empty
            if np.count_nonzero(board) == 0:
                # Place the first move in the center
                center = (board.shape[0] // 2, board.shape[1] // 2)
                print("Smart Agent starts by placing the first move in the center:", center)
                return center
        
        empty_positions = [(i, j) for i in range(board.shape[0]) for j in range(board.shape[1]) if board[i, j] == self.blank_symbol]

        if not empty_positions:
            print("No empty positions left")
            return (-1, -1)

        # Check for immediate blocking or winning move first
        for move in empty_positions:
            # Check if we can win
            if self.is_winning_move(board, move, self.agent_symbol):
                print("Winning move found:", move)
                return move
            # Check if we need to block opponent
            if self.is_winning_move(board, move, self.opponent_symbol):
                print("Blocking opponent move:", move)
                return move

        # If no immediate win or block, use minimax
        best_move = None
        best_score = -np.inf

        for move in empty_positions:
            board_copy = board.copy()
            board_copy[move] = self.agent_symbol
            score = self.minimax(board_copy, depth=3, alpha=-np.inf, beta=np.inf, is_maximizing=False)
            print(f"Evaluating move {move}, score: {score}")

            if score > best_score:
                best_score = score
                best_move = move

        print("Smart Agent chose move:", best_move, "with score:", best_score)
        return best_move

    def minimax(self, board, depth, alpha, beta, is_maximizing):
        if depth == 0 or self.is_game_over(board):
            return self.evaluate_board(board)

        empty_positions = [(i, j) for i in range(board.shape[0]) for j in range(board.shape[1]) if board[i, j] == self.blank_symbol]

        if is_maximizing:
            max_eval = -np.inf
            for move in empty_positions:
                board_copy = board.copy()
                board_copy[move] = self.agent_symbol
                eval = self.minimax(board_copy, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = np.inf
            for move in empty_positions:
                board_copy = board.copy()
                board_copy[move] = self.opponent_symbol
                eval = self.minimax(board_copy, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def is_winning_move(self, board, move, player):
        # Check if placing a move results in a win for the given player
        board_copy = board.copy()
        board_copy[move] = player
        return self.is_game_over(board_copy)

    def is_game_over(self, board):
        # Check if the game is over by checking for any winning line of length 5
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                if board[i, j] != self.blank_symbol:
                    if self.evaluate_line(board, i, j, self.agent_symbol) >= 1000 or self.evaluate_line(board, i, j, self.opponent_symbol) >= 1000:
                        return True
        return False

    def evaluate_board(self, board):
        # Enhanced evaluation function
        score = 0

        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                if board[i, j] == self.agent_symbol:
                    score += self.evaluate_line(board, i, j, self.agent_symbol)
                elif board[i, j] == self.opponent_symbol:
                    score -= self.evaluate_line(board, i, j, self.opponent_symbol)

        return score

    def evaluate_line(self, board, x, y, player):
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        max_score = 0

        for direction in directions:
            score = self.score_line(board, x, y, direction, player)
            if score > max_score:
                max_score = score

        return max_score

    def score_line(self, board, x, y, direction, player):
        dx, dy = direction
        length = 1
        score = 0

        for step in range(1, 5):
            nx = x + step * dx
            ny = y + step * dy

            if 0 <= nx < board.shape[0] and 0 <= ny < board.shape[1]:
                if board[nx, ny] == player:
                    length += 1
                elif board[nx, ny] == self.blank_symbol:
                    break
                else:
                    break
            else:
                break

        for step in range(1, 5):
            nx = x - step * dx
            ny = y - step * dy

            if 0 <= nx < board.shape[0] and 0 <= ny < board.shape[1]:
                if board[nx, ny] == player:
                    length += 1
                elif board[nx, ny] == self.blank_symbol:
                    break
                else:
                    break
            else:
                break

        # Enhanced heuristic scoring
        if length >= 5:
            score += 1000
        elif length == 4:
            score += 100
        elif length == 3:
            score += 10
        elif length == 2:
            score += 1

        return score
