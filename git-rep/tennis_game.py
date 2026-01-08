import random
import time
import os

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

class TennisGame:
    def __init__(self):
        self.player1_score = 0
        self.player2_score = 0
        self.player1_games = 0
        self.player2_games = 0
        self.ball_position = 50
        self.ball_direction = 1

    def draw_court(self):
        print("=" * 60)
        print("       8-BIT TENNIS GAME")
        print("=" * 60)
        print(f"\nPlayer 1: {self.player1_games} games | Score: {self.get_tennis_score(self.player1_score, self.player2_score)}")
        print(f"Player 2: {self.player2_games} games | Score: {self.get_tennis_score(self.player2_score, self.player1_score)}")
        print("\n" + "-" * 60)

        # Draw the court
        court = [" "] * 60
        court[29] = "|"
        court[30] = "|"

        # Draw ball position
        if 0 <= self.ball_position < 60:
            court[self.ball_position] = "o"

        # Draw players
        court[5] = "P1"
        court[54] = "P2"

        print("".join(court))
        print("-" * 60 + "\n")

    def get_tennis_score(self, score1, score2):
        if score1 == score2 and score1 >= 3:
            return "DEUCE"
        elif score1 >= 4 and score1 - score2 >= 2:
            return "WIN"
        elif score2 >= 4 and score2 - score1 >= 2:
            return "LOSE"
        elif score1 >= 3 and score1 - score2 == 1:
            return "ADV"
        elif score2 >= 3 and score2 - score1 == 1:
            return "40"
        else:
            scores = ["0", "15", "30", "40"]
            return scores[min(score1, 3)]

    def play_rally(self):
        self.ball_position = 30
        self.ball_direction = random.choice([-1, 1])

        rally_length = random.randint(3, 10)

        for _ in range(rally_length):
            self.ball_position += self.ball_direction * random.randint(2, 5)
            self.draw_court()
            time.sleep(0.3)

        # Determine winner of rally
        if random.random() > 0.5:
            self.player1_score += 1
            winner = "Player 1"
        else:
            self.player2_score += 1
            winner = "Player 2"

        print(f">>> {winner} wins the point! <<<\n")
        time.sleep(1)

        # Check if game is won
        if self.player1_score >= 4 and self.player1_score - self.player2_score >= 2:
            self.player1_games += 1
            print(f">>> Player 1 wins the game! <<<")
            self.player1_score = 0
            self.player2_score = 0
            time.sleep(2)
        elif self.player2_score >= 4 and self.player2_score - self.player1_score >= 2:
            self.player2_games += 1
            print(f">>> Player 2 wins the game! <<<")
            self.player1_score = 0
            self.player2_score = 0
            time.sleep(2)

    def play_match(self):
        print("\n*** STARTING MATCH ***\n")
        print("First to 3 games wins!\n")
        time.sleep(2)

        while self.player1_games < 3 and self.player2_games < 3:
            clear_screen()
            self.play_rally()

        clear_screen()
        self.draw_court()

        if self.player1_games == 3:
            print("\n" + "=" * 60)
            print("    PLAYER 1 WINS THE MATCH!")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("    PLAYER 2 WINS THE MATCH!")
            print("=" * 60)

if __name__ == "__main__":
    game = TennisGame()
    game.play_match()
