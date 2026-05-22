import copy
import time

import numpy as np
import pygame
from pygame import font
import sys

from util import load_config
from experiment import setup_mcts
from games.game_maker import GameMaker
from models.models import QLearningNet, PolicyValueNet
from monte_carlo import MCTSPlayer
from games.game_env import BoardGameEnv
from algorithms import random_policy, create_epsilon_greedy_policy
from trainer import QLearningTrainer, PolicyValueTrainer

CONFIG = load_config()

# Constants
WIDTH = 600
HEIGHT = 600
ROWS = 8
COLS = 8
SQUARE_SIZE = WIDTH // COLS

# Colors
WHITE = (255, 255, 255)
GRID_COLOR = TEXT_COLOR = BLACK = (0, 0, 0)
GREEN = (0, 128, 0)
GRAY = (128, 128, 128)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# test_state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, -1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1])
test_state = np.array(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, -1, 1,
     0, 0, 0, 0, 1, -1, 1, 1, 0, 0, 0, 0, -1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
test_state_2 = np.array([0, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1, 1, 1, 1, 0, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 0, 0, 1, 1, -1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, -1]
)

test_state_3 = np.array([0, 0, 0, -1, -1, -1, -1, -1,
                         -1, 0, -1, -1, -1, 1, 1, 1,
                         -1, -1, -1, 1, 1, -1, 1, 1,
                         -1, 1, -1, -1, 1, 1, 1, -1,
                         -1, 0, -1, -1, -1, 1, 1, -1,
                         0, 0, -1, -1, 1, -1, 1, -1,
                         0, 0, -1, -1, -1, 1, -1, 0,
                         0, 0, 1, 1, 1, 1, 1, -1, 1]
)

print(list(test_state.flatten()))


def draw_board(mode, screen, env, draw_hints=False):
    screen.fill(GREEN)
    # Get board size and square size
    board_size = BoardGameEnv.calc_board_size(env.get_state())
    square_size = WIDTH // board_size

    # Draw grid lines
    for i in range(1, board_size + 1):
        pygame.draw.line(screen, GRID_COLOR, (0, i * square_size), (WIDTH, i * square_size), 2)
        pygame.draw.line(screen, GRID_COLOR, (i * square_size, 0), (i * square_size, HEIGHT), 2)

    # Draw pieces on the board
    black_count = 0
    white_count = 0
    for row in range(board_size):
        for col in range(board_size):
            # pygame.draw.rect(screen, BLACK, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 1)
            idx = row * board_size + col
            if env.get_state()[idx] == BoardGameEnv.PLAYER_1:
                black_count += 1
                pygame.draw.circle(screen, (0, 0, 0),
                                   (col * square_size + square_size // 2, row * square_size + square_size // 2),
                                   square_size // 2 - 2)
            elif env.get_state()[idx] == BoardGameEnv.PLAYER_2:
                white_count += 1
                pygame.draw.circle(screen, (255, 255, 255),
                                   (col * square_size + square_size // 2, row * square_size + square_size // 2),
                                   square_size // 2 - 2)

    # Draw hints for possible moves
    possible_actions = env.get_possible_actions(env.get_current_player())
    for action in possible_actions:
        row = action // board_size
        col = action % board_size
        # Give hint in correct color
        if (env.get_current_player() == BoardGameEnv.PLAYER_1):
            pygame.draw.circle(screen, BLACK,
                               (col * square_size + square_size // 2, row * square_size + square_size // 2), 5)
        else:
            pygame.draw.circle(screen, WHITE,
                               (col * square_size + square_size // 2, row * square_size + square_size // 2), 5)

    # Display game mode
    font = pygame.font.Font(None, 36)
    mode_text = font.render(f"Mode: {'Human vs AI' if mode == 0 else 'Human vs Human'}", True, TEXT_COLOR)
    screen.blit(mode_text, (10, HEIGHT + 20))

    # Display whose turn it is
    turn_text = font.render(f"Turn: {'Black' if env.get_current_player() == BoardGameEnv.PLAYER_1 else 'White'}", True,
                            TEXT_COLOR)
    screen.blit(turn_text, (10, HEIGHT + 60))

    # Count number of black and white pieces
    count_text = font.render(f"Black: {black_count}  White: {white_count}", True, TEXT_COLOR)
    screen.blit(count_text, (10, HEIGHT + 100))

    # Undo text
    undo_text = font.render(f"Press 'U' on your keyboard to undo, 'H' for hint.", True, TEXT_COLOR)
    screen.blit(undo_text, (10, HEIGHT + 140))
    
    # Restart and quit text
    restart_quit_text = font.render(f"At the end, press 'R'  to restart, 'Q' to quit.", True, TEXT_COLOR)
    screen.blit(restart_quit_text, (10, HEIGHT + 180))


def get_click_pos(pos, board_size):
    x, y = pos
    row = y // (WIDTH // board_size)
    col = x // (WIDTH // board_size)
    return row, col


def show_message(screen, message, time):
    font.init()
    text_font = font.SysFont("Arial", 24)
    text_surface = text_font.render(message, True, RED)
    text_rect = text_surface.get_rect()
    text_rect.center = (screen.get_width() // 2, screen.get_height() // 2)
    pygame.draw.rect(screen, BLACK, (text_rect.left, text_rect.top, text_rect.width, text_rect.height), 1)
    screen.blit(text_surface, text_rect)
    pygame.display.update()
    pygame.time.delay(time)


def draw_text(screen, text, position, font_size, color):
    font = pygame.font.Font(None, font_size)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center=position)
    screen.blit(text_surface, text_rect)


def draw_button(screen, text, position, size, color, text_color):
    rect = pygame.Rect(position[0], position[1], size[0], size[1])
    pygame.draw.rect(screen, color, rect)
    draw_text(screen, text, rect.center, 30, text_color)
    return rect


def show_start_menu(screen):
    running = True
    text_lines = ["Welcome to Reversi.",
                  "",
                  "This is course project from CS5180",
                  "Reinforcement Learning and Sequential Decision Making",
                  "Author: Shoukang Hong, Jingming Cheng",
                  "",
                  "To win, you need to have more pieces on the board",
                  "than your opponent by the end of the game.",
                  "Each piece played must be laid adjacent to an opponent's piece",
                  "so that the opponent's piece or a row of opponent's pieces is",
                  "flanked by the new piece and another piece of the player's color.",
                  "All of the opponent's pieces between these two pieces",
                  "are 'captured' and turned over to match the player's color.",
                  "",
                  "Press 'U' to undo, 'H' for hint when you play this game.",
                  "Hint format: Q, n_visits"]
    while running:
        screen.fill(GREEN)
        font = pygame.font.Font(None, 28)
        text_surfaces = [font.render(line, True, TEXT_COLOR) for line in text_lines]
        text_positions = [(0, 0 + i * (text_surfaces[0].get_height() + 10)) for i, _ in enumerate(text_lines)]
        for surface, pos in zip(text_surfaces, text_positions):
            screen.blit(surface, pos)
        start_btn = draw_button(screen, 'Start Game', (200, 600), (200, 50), GRAY, WHITE)
        quit_btn = draw_button(screen, 'Quit Game', (200, 700), (200, 50), GRAY, WHITE)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                if start_btn.collidepoint(mouse_pos):
                    return
                elif quit_btn.collidepoint(mouse_pos):
                    event.type = pygame.QUIT

        pygame.display.flip()


def show_mode_menu(screen):
    running = True
    while running:
        screen.fill(GREEN)
        mode_btn_human_ai = draw_button(screen, 'Human vs AI', (200, 150), (200, 50), GRAY, WHITE)
        mode_btn_human_human = draw_button(screen, 'Human vs Human', (200, 250), (200, 50), GRAY, WHITE)
        # mode_btn_ai_ai = draw_button(screen, 'AI vs AI', (200, 350), (200, 50), GRAY, WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                if mode_btn_human_ai.collidepoint(mouse_pos):
                    return 0  # Human vs AI
                elif mode_btn_human_human.collidepoint(mouse_pos):
                    return 1  # Human vs Human
                # elif mode_btn_ai_ai.collidepoint(mouse_pos):
                #     return 2  # AI vs AI

        pygame.display.flip()


# def show_board_size_menu(screen):
#     running = True
#     while running:
#         screen.fill(GREEN)
#         btn_6x6 = draw_button(screen, '6x6 Board', (200, 150), (200, 50), GRAY, WHITE)
#         btn_8x8 = draw_button(screen, '8x8 Board', (200, 250), (200, 50), GRAY, WHITE)
#         btn_10x10 = draw_button(screen, '10x10 Board', (200, 350), (200, 50), GRAY, WHITE)

#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 pygame.quit()
#                 sys.exit()
#             elif event.type == pygame.MOUSEBUTTONDOWN:
#                 mouse_pos = event.pos
#                 if btn_6x6.collidepoint(mouse_pos):
#                     return 6  # Board size 6x6
#                 elif btn_8x8.collidepoint(mouse_pos):
#                     return 8  # Board size 8x8
#                 elif btn_10x10.collidepoint(mouse_pos):
#                     return 10  # Board size 10x10

#         pygame.display.flip()


def show_color_selection_menu(screen):
    running = True
    while running:
        screen.fill(GREEN)
        black_btn = draw_button(screen, 'Play as Black', (200, 200), (200, 50), GRAY, WHITE)
        white_btn = draw_button(screen, 'Play as White', (200, 300), (200, 50), GRAY, WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                if black_btn.collidepoint(mouse_pos):
                    return BoardGameEnv.PLAYER_1  # Player chooses to play as black
                elif white_btn.collidepoint(mouse_pos):
                    return BoardGameEnv.PLAYER_2  # Player chooses to play as white

        pygame.display.flip()


def show_difficulty_selection_menu(screen):
    running = True
    while running:
        screen.fill(GREEN)
        easy_btn = draw_button(screen, 'Easy (random policy)', (100, 150), (400, 50), GRAY, WHITE)
        medium_btn = draw_button(screen, 'Medium(Qlearning model policy)', (100, 250), (400, 50), GRAY, WHITE)
        hard_btn = draw_button(screen, 'Hard(MCTS model policy)', (100, 350), (400, 50), GRAY, WHITE)
        lunatic_btn = draw_button(screen, 'Lunatic(MCTS model 100 playout)', (100, 450), (400, 50), GRAY, WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                if easy_btn.collidepoint(mouse_pos):
                    return 0  # Player chooses the Easy difficulty
                elif medium_btn.collidepoint(mouse_pos):
                    return 1  # Player chooses the Medium difficulty
                elif hard_btn.collidepoint(mouse_pos):
                    return 2  # Player chooses the Hard difficulty
                elif lunatic_btn.collidepoint(mouse_pos):
                    return 3  # Player chooses the Lunatic difficulty

        pygame.display.flip()

def main():
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT + 220))  # Give more space for text
    pygame.display.set_caption('Reversi')

    # start_menu = show_start_menu(screen)
    # Show the menu and get the selected mode
    mode = show_mode_menu(screen)
    # mode = 0  # Human vs AI
    # mode = 1  # Human vs human
    # mode = 2  # AI vs AI

    # Show the board size menu and get the selected size
    # board_size = show_board_size_menu(screen)  # You can adjust the board size as needed
    board_size = 8 # 2024.4.20, show_board_size_menu is disabled.
    env: BoardGameEnv = GameMaker.make('reversi')
    env.reset()
    # env.set_state(test_state_3)

    player_color = show_color_selection_menu(screen) if mode != 1 else BoardGameEnv.PLAYER_1

    # _, trainer, _, _ = setup_mcts()
    model = PolicyValueNet(board_size=board_size, num_filters=80, num_res_blocks=12,
                           in_channels=3, num_value_out_channel=2,
                           num_policy_out_channel=2, num_value_fc1=256, num_actions=BoardGameEnv.action_space_n())
    trainer = PolicyValueTrainer(model, f"./temp/mcts_ss_8_{board_size}_{model.__class__.__name__}",
                                 has_eval_model=False)
    trainer.load_model("benchmark_model", load_eval_model=False)
    # symmetrize_policy_value_net(trainer.train_model)
    if mode == 0 or mode == 2:
        # Show the difficulty selection menu and get the selected difficulty
        difficulty = show_difficulty_selection_menu(screen)
        policy = random_policy
        if difficulty == 0:
            policy = random_policy
        elif difficulty == 1:
            model_q = QLearningNet(board_size=board_size, num_filters=128, num_res_blocks=10, fc1_size=256)
            trainer_q = QLearningTrainer(model_q, f"./temp/qlearning_{board_size}_{model_q.__class__.__name__}")
            trainer_q.load_model("benchmark_model")
            policy = create_epsilon_greedy_policy(0, trainer_q.train_model_eval_q, env.action_space.n, temp=0.01)
        elif difficulty == 2 or difficulty == 3:
            n_playout = 1 if difficulty == 2 else 100
            mcts_player = MCTSPlayer(trainer.policy_value_fn, copy.deepcopy(env), n_playout=n_playout, is_self_play=False)
            mcts_player.mcts.select_mode = 1
            def func(s, possible_actions):
                probs, v = trainer.policy_value_fn(s, possible_actions)
                env.render()
                acts, probs = zip(*probs)
                prob_array = np.array([0] * 64, dtype=int)
                prob_array[np.array(acts)] = np.round(np.array(probs) * 1000)
                print(prob_array.reshape((8, 8)), v)
                action, move_probs, v = mcts_player.get_action(s, board_size ** 2, return_prob=True, temp=0.1)
                print(np.array(move_probs.reshape((8, 8)) * 1000, dtype=int), action, v)
                return action
            policy = func

    mcts_player_human = MCTSPlayer(trainer.policy_value_fn, copy.deepcopy(env), n_playout=6, is_self_play=False)
    mcts_player_human.mcts.select_mode = 1
    def ref_func(s, possible_actions):
        probs, v = trainer.policy_value_fn(s, possible_actions)
        print(list(env.get_state()))
        env.render()
        acts, probs = zip(*probs)
        prob_array = np.array([0] * 64, dtype=int)
        if mcts_player_human.mcts._root and mcts_player_human.mcts._root._P_arr is not None:
            prob_array[np.array(acts)] = np.round(mcts_player_human.mcts._root._P_arr * 1000)
        else:
            prob_array[np.array(acts)] = np.round(np.array(probs) * 1000)
        print(prob_array.reshape((8, 8)), v)
        if mcts_player_human.mcts._root and mcts_player_human.mcts._root._P_arr is not None:
            node = mcts_player_human.mcts._root
            print(node._done, node._done_arr, node._Q, node.get_child_player())
            acts, probs, v = mcts_player_human.mcts.get_move_probs(s, temp=1)
            print(mcts_player_human.mcts._root._n_visit_arr)
            prob_array = np.array([0] * 64, dtype=int)
            prob_array[np.array(acts)] = np.round(probs * 1000)
            print(prob_array.reshape((8, 8)), v)
        return action

    # print(env.get_current_player())
    
    def get_hint(state, possible_actions, mode):
        action, move_probs, v = mcts_player_human.mcts.get_move_probs(state, temp=1, max_playout=100000)
        root_node = mcts_player_human.mcts._root
        # Convert move_probs to a more visually friendly format, e.g., a list of probabilities
        hints = [(action, child._Q, child._n_visits) for action, child in mcts_player_human.mcts._root._children.items()]
        return hints, v

    def draw_hints(screen, env, hints, win_rate):
        draw_board(mode, screen, env, draw_hints=True)  # Reuse the draw_board function with an additional parameter to indicate hint drawing
        player = env.get_current_player()
        # Draw move probabilities on the board
        board_size = BoardGameEnv.calc_board_size(env.get_state())
        square_size = WIDTH // board_size
        prob_font = pygame.font.Font(None, 32)  # Font size of probability
        rate_font = pygame.font.Font(None, 32)  # Font size of win_rate
        
        action_list = []
        Q_list = []
        n_visits_list = []
        for action, Q, n_visits in hints:
            action_list.append(action)
            Q_list.append(Q)
            n_visits_list.append(n_visits)
        best_Q = max(Q_list) if env.get_current_player() == 1 else min(Q_list)
        best_n_visits = max(n_visits_list)
        for action, Q, n_visits in hints:
            row, col = action // board_size, action % board_size
            if n_visits == best_n_visits and best_n_visits > 1:
                color = RED
            elif Q == best_Q:
                color = BLUE
            else:
                color = BLACK if env.get_current_player() == BoardGameEnv.PLAYER_1 else WHITE
            # idx_text = prob_font.render(f"{action}", True, BLACK if env.get_current_player() == ReversiEnv.BLACK else WHITE, GREEN)
            hint_text = prob_font.render(f"{(1 +Q * player) * 50:.1f}%", True, color, GREEN)
            visit_text = prob_font.render(f"{n_visits}", True, color, GREEN)
            # idx_rect = hint_text.get_rect(center=(col * square_size + square_size // 2, row * square_size + square_size // 2 - 15))
            text_rect = hint_text.get_rect(center=(col * square_size + square_size // 2, row * square_size + square_size // 2 - 18))
            # screen.blit(idx_text, idx_rect)
            screen.blit(hint_text, text_rect)
            text_rect = visit_text.get_rect(center=(col * square_size + square_size // 2, row * square_size + square_size // 2 + 22))
            screen.blit(visit_text, text_rect)

        # Display the win rate at the bottom of the board
        win_rate_text = rate_font.render(f"Win Rate: { (1+win_rate * player)*50:.1f}%", True, BLACK)
        screen.blit(win_rate_text, (WIDTH/2, HEIGHT + 60))  # Adjust positioning as needed

        pygame.display.flip()  # Make sure to update the display to show the hints

    showing_hints = False
    st = time.time()
    sv = 0
    while not env.done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            else:
                if mode == 2 or mode == 0 and env.get_current_player() != player_color:
                    action = policy(np.copy(env.get_state()), env.get_possible_actions(env.get_current_player()))
                    if action in env.get_possible_actions(env.get_current_player()):
                        state, reward, done, info = env.step(action,True)
                        mcts_player_human.mcts.update_with_move(action)
                        # print("History: " + str(env.state_history))
                        # pygame.time.delay(1000)
                
                else: # Human
                    # print(env.get_current_player())
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        pos = pygame.mouse.get_pos()
                        row, col = get_click_pos(pos, board_size)
                        action = row * board_size + col
                        if action in env.get_possible_actions(env.get_current_player()):
                            ref_func(np.copy(env.get_state()), env.get_possible_actions(env.get_current_player()))
                            state, reward, done, info = env.step(action,True)
                            mcts_player_human.mcts.update_with_move(action)
                            # print("History: " + str(env.state_history))
                    # Implement undo functionality
                    elif event.type == pygame.KEYDOWN:
                        mcts_player_human.mcts.update_with_move(-1)
                        if event.key == pygame.K_u:  # Assuming 'u' key triggers undo
                            # If unable to undo, please see "undo" in game_env.py
                            undo_result, new_state = env.undo(mode)
                            if undo_result == True:
                                show_message(screen, "Undo", 1000)
                            else:
                                show_message(screen, "No previous moves to undo.", 1000)
                        elif event.key == pygame.K_h: # Hint
                            showing_hints = not showing_hints  # Toggle the hint display

        if env.done:
            showing_hints = False
        if showing_hints and (mode == 1 or (mode == 0 and env.get_current_player() == player_color)):
            hints, win_rate = get_hint(env.get_state(), env.get_possible_actions(env.get_current_player()), mode)
            dn = mcts_player_human.mcts._root._n_visits - sv
            if dn < 0:
                sv = mcts_player_human.mcts._root._n_visits
                st = time.time()
            next_time = time.time()
            if dn > 0 and next_time - st > 1:
                print(f"{dn/(next_time - st)} playout per second")
                st = next_time
                sv = mcts_player_human.mcts._root._n_visits
            draw_hints(screen, env, hints, win_rate)
        else:
            draw_board(mode, screen, env)
        pygame.display.flip()

    # Check which one wins
    black_count = 0
    white_count = 0
    for row in range(board_size):
        for col in range(board_size):
            # pygame.draw.rect(screen, BLACK, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 1)
            idx = row * board_size + col
            if env.get_state()[idx] == BoardGameEnv.PLAYER_1:
                black_count += 1
            elif env.get_state()[idx] == BoardGameEnv.PLAYER_2:
                white_count += 1
    if black_count == white_count:
        show_message(screen, "Tie", 2000)
    elif black_count < white_count:
        show_message(screen, "White Wins!", 2000)
    else:
        show_message(screen, "Black Wins!", 2000)
    # print(env.done)
    if env.done:
        while env.done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pass
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r: # Restart the game
                        main()
                    elif event.key == pygame.K_q: # Quit the game:
                        env.done = False # Just for end the while loop
    env.done = True # Resume game quit

if __name__ == '__main__':
    np.set_printoptions(precision=2, floatmode="fixed", linewidth=300)
    main()