# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 19:36:42 2020

@author: simon, tim, florian
"""

from itertools import groupby, chain
from dqn import Agent
import numpy as np
import math
import random
import copy
import pygame
import sys
import matplotlib.pyplot as plt

# colors for GUI
blue = (0, 0, 255)
black = (0, 0, 0)
red = (255, 0, 0)
yellow = (255, 255, 0)

# AI = Minimax, Player = Our reinforcement model or us when playing in GUI
PLAYER = 0
AI = 1
PLAYER_PIECE = 1
AI_PIECE = 2

NONE = '.'
Tie = 0

done = False

# global variables in PEP format
WINDOW_LENGTH = 4
ROW_COUNT = 6
COLUMN_COUNT = 7
SQUARESIZE = 100
WIDTH = COLUMN_COUNT * SQUARESIZE
HEIGHT = (ROW_COUNT + 1) * SQUARESIZE
SIZE = (WIDTH, HEIGHT)
RADIUS = int(SQUARESIZE / 2 - 5)
EMPTY = 0


def create_board():
    """
        INPUT: Nothing

        OUTPUT: default 6x7 Matrix filled with zeros
    """
    board = np.zeros((ROW_COUNT, COLUMN_COUNT))
    return board


def print_board(board):
    """
            INPUT:
                -board: The board from create_board()

            OUTPUT:
                    no return, just print the flipped board, because we fill it top to bottom, but in reality it needs
                    to be shown bottom to top, so flip mirrors it horizontally.
        """
    print(np.flip(board, 0))


def draw_board(board, screen):
    """
        INPUT:
            - board: the board from create_board()
            - screen: the pygame initialized screen, for rendering the GUI

        OUTPUT:
            and updated GUI with the latest turn
    """
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, blue, (c * SQUARESIZE, r * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, black, (
                int(c * SQUARESIZE + SQUARESIZE / 2), int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)), RADIUS)

    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            if board[r][c] == PLAYER_PIECE:
                pygame.draw.circle(screen, red, (
                    int(c * SQUARESIZE + SQUARESIZE / 2), HEIGHT - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
            elif board[r][c] == AI_PIECE:
                pygame.draw.circle(screen, yellow, (
                    int(c * SQUARESIZE + SQUARESIZE / 2), HEIGHT - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
    pygame.display.update()


def drop_piece(board, row, col, piece):
    """
        INPUT:
            - board: the board from create_board()
            - row (int): the top row where the piece can be placed on
            - col (int): the to dropped col
            - piece (int): either the PLAYER_PIECE (1) or AI_PIECE (2)

        OUTPUT:
            update the board with the dropped piece
    """
    board[row][col] = piece


def is_valid_location(board, col):
    """
        INPUT:
            - board: the board from create_board()
            - col (int): the to dropped col

        OUTPUT:
            True if the piece can be dropped on this column, False else
    """
    return board[ROW_COUNT - 1][col] == 0


def get_valid_locations(board):
    """
        INPUT:
            - board: the board from create_board()

        OUTPUT:
            check for not filled up columns. This is function is used by the minimax
            to check for future steps
    """
    valid_locations = []

    for c in range(COLUMN_COUNT):
        if is_valid_location(board, c):
            valid_locations.append(c)

    return valid_locations


def get_next_open_row(board, col):
    """
           INPUT:
               - board: the board from create_board()
               - col (int): the to dropped col

           OUTPUT:
               check for the next valid location of the row. Something between 0 and 5.
       """
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r


def winning_move(board, piece):
    """
        INPUT:
            - board: the board from create_board()
            - piece (int): either the PLAYER_PIECE (1) or AI_PIECE (2)

        OUTPUT:
            True if there is either a horizontal, vertical or diagonal victory. Victory = 4 pieces together.
            False else
    """
    # Check horizontal locations for win
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c + 1] == piece and board[r][c + 2] == piece and board[r][
                c + 3] == piece:
                return True

    # Check vertical locations for win
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r + 1][c] == piece and board[r + 2][c] == piece and board[r + 3][
                c] == piece:
                return True

    # Check positively sloped diaganols
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r + 1][c + 1] == piece and board[r + 2][c + 2] == piece and board[r + 3][
                c + 3] == piece:
                return True

    # Check negatively sloped diaganols
    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == piece and board[r - 1][c + 1] == piece and board[r - 2][c + 2] == piece and board[r - 3][
                c + 3] == piece:
                return True


def score_position(board, piece):
    """
        INPUT:
            - board: the board from create_board()
            - piece (int): either the PLAYER_PIECE (1) or AI_PIECE (2)

        OUTPUT:
            This function uses the evaluate_window function to calculate the total score for
            the steps into the future (depth). The highest score returned by this function will
            be used by the minimax to take this next highest scored turn.
    """
    score = 0

    ## score center
    center_arr = [int(i) for i in list(board[:, COLUMN_COUNT // 2])]
    center_count = center_arr.count(piece)
    score += 3 * center_count

    ## horizontal score
    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r, :])]

        for c in range(COLUMN_COUNT - 3):
            window = row_array[c: c + 6]
            score += evaluate_window(window, piece)

    ## vertical score
    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:, c])]

        for r in range(ROW_COUNT - 3):
            window = col_array[r: r + 5]
            score += evaluate_window(window, piece)

    ## diagonal positive score
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    ## diagonal negative score
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + 3 - i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    return score


def is_terminal_node(board):
    """
        INPUT:
            - board: the board from create_board()

        OUTPUT:
            return True if the game is either won for either player or there is a tie. False else
    """
    return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(get_valid_locations(board)) == 0


def evaluate_windows(window, piece):
    """
        INPUT:
            - board: the board from create_board()
            - piece (int): either the PLAYER_PIECE (1) or AI_PIECE (2)

        OUTPUT:
            returns the score for the evaluated window provided by the score_position function.
    """
    score = 0

    opponent_piece = PLAYER_PIECE

    if piece == AI_PIECE:
        opponent_piece = PLAYER_PIECE

    if window.count(piece) == 3 and window.count(0) <= 1:
        score += 20

    elif window.count(piece) == 3 and window.count(0) >= 1 and window.count(opponent_piece) >= 1:
        score -= 20

    elif window.count(piece) == 3 and window.count(0) >= 0 and window.count(opponent_piece) == 0:
        score += 50

    elif window.count(piece) == 2 and window.count(0) <= 2:
        score += 10

    elif window.count(piece) == 2 and window.count(0) > 2:
        score += 5

    if window.count(opponent_piece) == 3 and window.count(0) <= 1:
        score -= 50

    elif window.count(opponent_piece) == 3 and window.count(0) > 1:
        score -= 3

    elif window.count(opponent_piece) == 2 and window.count(0) == 2:
        score -= 5

    elif window.count(opponent_piece) == 2 and window.count(0) <= 2:
        score -= 20

    elif window.count(opponent_piece) == 2 and window.count(0) > 2:
        score -= 10

    return score


def evaluate_window(window, piece):
    score = 0

    #PLAYER = 0
    #AI = 1
    #PLAYER_PIECE = 1
    #AI_PIECE = 2
    opp_piece = PLAYER_PIECE
    if piece == AI_PIECE:
        opp_piece = PLAYER_PIECE
    print(f"Count: {window.count(piece)} | Opponent Count: {window.count(opp_piece)} | Window = {window} | Piece = {piece}")
    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 5
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 2

    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= 4

    return score


def minimax(board, depth, alpha, beta, maximizingPlayer):
    """
        This is the key AI function of Minimax. Recursive calls and scorings for the future turns

        INPUT:
            - board: the board from create_board()
            - depth (int): indicating the steps looked into the future
            - alpha (int): used for the alpha-beta pruning. More efficient
            - beta (int): used for the alpha-beta pruning. More efficient
            - maximizingPlayer (bool): used for the recursive max call on the minimax and min calls on the opponent (us)

        OUTPUT:
            based on the highest score for all evaluated possible moves and with respect to the depth,
            the return is the best predicted column
    """
    valid_locations = get_valid_locations(board)

    if is_terminal_node(board) or depth == 0:
        if is_terminal_node(board):
            if winning_move(board, AI_PIECE):
                return None, 10000000
            elif winning_move(board, PLAYER_PIECE):
                return None, -10000000
            else:
                return None, 0

        else:
            return None, score_position(board, AI_PIECE)

    if maximizingPlayer:
        value = -math.inf
        column = random.choice(valid_locations)

        for col in valid_locations:
            row = get_next_open_row(board, col)
            bord_p = copy.deepcopy(board)
            drop_piece(bord_p, row, col, AI_PIECE)
            new_score = minimax(bord_p, depth - 1, alpha, beta, False)[1]

            if new_score > value:
                value = new_score
                column = col

            alpha = max(alpha, value)
            if alpha >= beta:
                break

        return column, value

    else:
        value = math.inf

        for col in valid_locations:
            row = get_next_open_row(board, col)
            bord_p = copy.deepcopy(board)
            drop_piece(bord_p, row, col, PLAYER_PIECE)
            new_score = minimax(bord_p, depth - 1, alpha, beta, True)[1]

            if new_score < value:
                value = new_score
                column = col

            beta = min(beta, value)
            if alpha >= beta:
                break

        return column, value


def diagonalsPos(board, cols, rows):
    """
        Get positive diagonals, going from bottom-left to top-right.

        INPUT:
            - board: the board from create_board()
            - row (int): the top row where the piece can be placed on
            - col (int): the to dropped col

        OUTPUT:
            return the positive diagonals, going from bottom-left to top-right.
    """
    for di in ([(j, i - j) for j in range(cols)] for i in range(cols + rows - 1)):
        yield [board[i][j] for i, j in di if i >= 0 and j >= 0 and i < cols and j < rows]


def diagonalsNeg(board, cols, rows):
    """
        INPUT:
            - board: the board from create_board()
            - row (int): the top row where the piece can be placed on
            - col (int): the to dropped col

        OUTPUT:
            return the negative diagonals, going from top-left to bottom-right.
    """
    for di in ([(j, i - cols + j + 1) for j in range(cols)] for i in range(cols + rows - 1)):
        yield [board[i][j] for i, j in di if i >= 0 and j >= 0 and i < cols and j < rows]


class Game:
    """
        Game class will be initialized for each training function like play_minimax.
        This class creates the board and offers print and win checking function.
    """

    def __init__(self, cols=7, rows=6, requiredToWin=4):
        self.cols = cols
        self.rows = rows
        self.win = requiredToWin
        self.board = [[0] * rows for _ in range(cols)]

    def insert(self, column, color):
        """Insert the color in the given column."""
        c = self.board[column]
        if c[0] != 0:
            raise Exception('Column is full')

        i = -1
        while c[i] != 0:
            i -= 1
        c[i] = color

        self.checkForWin()

    def checkForWin(self):
        """Check the current board for a winner."""
        w = self.getWinner()
        if w == PLAYER or w == AI:
            # self.printBoard()
            # print('%d'%w + ' won!')
            return
        if w == Tie:
            # print('Tie')
            return

    def getWinner(self):
        """Get the winner on the current board."""
        global horizontal_win
        global done
        lines = (
            self.board,  # columns
            zip(*self.board),  # rows
            diagonalsPos(self.board, self.cols, self.rows),  # positive diagonals
            diagonalsNeg(self.board, self.cols, self.rows)  # negative diagonals
        )

        for sublist in self.board:
            if sublist[0] == sublist[1] == sublist[2] == sublist[3] or sublist[1] == sublist[2] == sublist[3] == \
                    sublist[4] or sublist[2] == sublist[3] == sublist[4] == sublist[5]:
                horizontal_win = True

        for line in chain(*lines):
            for color, group in groupby(line):
                if color != 0 and len(list(group)) >= self.win:
                    done = True
                    return color
        counter = 0
        for sublist in self.board:
            for i in sublist:
                if i != 0:
                    counter += 1
        if counter == 42:
            done = True
            return Tie


    def printBoard(self):
        """Print the board."""
        print('  '.join(map(str, range(self.cols))))
        for y in range(self.rows):
            print('  '.join(str(self.board[x][y]) for x in range(self.cols)))
        print()

    def check_if_action_valid(self, column):
        valid = True
        c = self.board[column]
        if c[0] != 0:
            valid = False
        return valid


def play():
    """
        legacy function for our first try to train reinforcement learning
    """
    global done
    done = False
    g = Game()
    turn = random.choice([PLAYER, AI])
    transitions_agent = []
    agent.epsilon = agent.eps_min
    while done == False:
        g.printBoard()
        if turn == PLAYER:
            row = input('{}\'s turn:'.format('Red'))
            g.insert(int(row), turn)
        else:
            observation = []
            for sublist in g.board:
                for i in sublist:
                    observation.append(i)
            observation = np.asarray(observation)
            action = agent.choose_action(observation)
            if g.check_if_action_valid(action):
                print('{}\'s turn: %d'.format('Yellow') % action)
                g.insert(action, turn)
            else:
                while g.check_if_action_valid(action) == False:
                    agent.store_transition(observation, action, -100, observation, done)
                    action = agent.choose_action(observation)
                print('{}\'s turn: %d'.format('Yellow') % action)
                g.insert(action, turn)
            observation_ = []
            for sublist in g.board:
                for i in sublist:
                    observation_.append(i)
            observation_ = np.asarray(observation_)
            transitions_agent += [(observation, action, observation_, done)]
        turn = AI if turn == PLAYER else PLAYER
    winner = AI if turn == PLAYER else PLAYER
    if winner == AI:
        reward = 20
    else:
        reward = -20
    for i in range(len(transitions_agent)):
        agent.store_transition(transitions_agent[i][0], transitions_agent[i][1], reward, transitions_agent[i][2],
                               transitions_agent[i][3])
    agent.learn()
    return


def selfplay():
    """
        legacy function for trying to implement self-play reinforcement learning like alpha-zero Go
    """
    agent2 = Agent(0.99, 0.1, 0.003, [42], train_games, 7, eps_dec)
    agent2.load_checkpoint()
    global win_cntr
    global done
    g = Game()
    turn = random.choice([PLAYER, AI])
    done = False
    transitions_agent = []
    transitions_agent2 = []
    while done == False:
        g.printBoard()
        if turn == PLAYER:
            # row = input('{}\'s turn: '.format('Red'))
            # g.insert(int(row), turn)
            observation = []
            for sublist in g.board:
                for i in sublist:
                    observation.append(i)
            observation = np.asarray(observation)
            action = agent2.choose_action(observation)
            if g.check_if_action_valid(action):
                print('{}\'s turn: %d'.format('Red') % action)
                g.insert(action, turn)
            else:
                while g.check_if_action_valid(action) == False:
                    agent.store_transition(observation, action, -100, observation, done)
                    action = np.random.randint(7)
                print('{}\'s turn: %d'.format('Red') % action)
                g.insert(action, turn)
            observation_ = []
            for sublist in g.board:
                for i in sublist:
                    observation_.append(i)
            observation_ = np.asarray(observation_)
            transitions_agent2 += [(observation, action, observation_, done)]
        else:
            observation = []
            for sublist in g.board:
                for i in sublist:
                    observation.append(i)
            observation = np.asarray(observation)
            action = agent.choose_action(observation)
            if g.check_if_action_valid(action):
                print('{}\'s turn: %d'.format('Yellow') % action)
                g.insert(action, turn)
            else:
                while g.check_if_action_valid(action) == False:
                    agent.store_transition(observation, action, -100, observation, done)
                    action = agent.choose_action(observation)
                print('{}\'s turn: %d'.format('Yellow') % action)
                g.insert(action, turn)
            observation_ = []
            for sublist in g.board:
                for i in sublist:
                    observation_.append(i)
            observation_ = np.asarray(observation_)
            transitions_agent += [(observation, action, observation_, done)]
        turn = AI if turn == PLAYER else PLAYER
    if g.getWinner() == Tie:
        reward_agent = 0
    else:
        winner = AI if turn == PLAYER else PLAYER
        if winner == AI:
            win_cntr += 1
            if horizontal_win:
                reward_agent = 5
            else:
                reward_agent = 20
                reward_agent2 = -20
        else:
            reward_agent = -20
            reward_agent2 = 20
    for i in range(len(transitions_agent)):
        agent.store_transition(transitions_agent[i][0], transitions_agent[i][1], reward_agent, transitions_agent[i][2],
                               transitions_agent[i][3])
    agent.learn()
    return


def play_minimax(depth):
    """
        Our primary function for training our reinforcement model. The depth equals the strength and looks into
        the future of the minimax. This function executed plays against the minimax from the top.
    """
    global win_cntr
    global lose_cntr
    global done
    global horizontal_win
    horizontal_win = False
    g = Game()
    turn = np.random.randint(2)
    done = False
    transitions_agent = []
    transitions_agent2 = []
    while done == False:
        g.printBoard()

        # Player 1 dqn agent
        if turn == PLAYER:
            observation = []
            for row, sublist in enumerate(g.board):
                for col, obs in enumerate(sublist):
                    observation.append(obs)

            observation = np.asarray(observation)
            action = agent.choose_action(observation)
            if g.check_if_action_valid(action):
                g.insert(action, PLAYER_PIECE)
            else:
                while g.check_if_action_valid(action) == False:
                    agent.store_transition(observation, action, -100, observation, done)
                    action = agent.choose_action(observation)
                g.insert(action, PLAYER_PIECE)
            observation_ = []
            for sublist in g.board:
                for obs in sublist:
                    observation_.append(obs)
            observation_ = np.asarray(observation_)
            transitions_agent2 += [(observation, action, observation_, done)]

        # Player 2 Minimax
        else:
            observation = []
            obs = np.zeros((6, 7))
            for row, sublist in enumerate(g.board):
                for col, sub in enumerate(sublist):
                    observation.append(sub)
                    obs[col, row] = sub

            observation = np.asarray(observation)

            if np.random.rand(1) > decay:
                action, _ = minimax(np.flipud(obs), depth, -math.inf, math.inf, True)
            else:
                action = np.random.randint(7)

            if g.check_if_action_valid(action):
                g.insert(action, AI_PIECE)
            else:
                while not g.check_if_action_valid(action):
                    agent.store_transition(observation, action, -100, observation, done)
                    action = np.random.randint(7)

                g.insert(action, AI_PIECE)

            observation_ = []
            for sublist in g.board:
                for sub in sublist:
                    observation_.append(sub)
            observation_ = np.asarray(observation_)
            transitions_agent += [(observation, action, observation_, done)]
        turn = (turn + 1) % 2

    if g.getWinner() == Tie:
        reward_minimax = -5
        reward_nn = -5
    else:
        winner = AI if turn == PLAYER else PLAYER

        if winner == AI:
            lose_cntr += 1
            reward_minimax = 200
            reward_nn = -200
        else:
            win_cntr += 1
            reward_minimax = -200
            reward_nn = 200

    for obs in range(len(transitions_agent)):
        agent.store_transition(transitions_agent[obs][0], transitions_agent[obs][1], reward_minimax,
                               transitions_agent[obs][2], transitions_agent[obs][3])
    for i in range(len(transitions_agent2)):
        agent.store_transition(transitions_agent2[i][0], transitions_agent2[i][1], reward_nn, transitions_agent2[i][2],
                               transitions_agent2[i][3])
    agent.learn()
    return


def play_against_minimax():
    """
        Here we can play against the minimax in the terminal
    """
    global FIRST_MOVE
    global done
    done = False
    g = Game()
    turn = np.random.randint(2)
    # if turn == RED:
    #    FIRST_MOVE = False
    transitions_agent = []
    agent.epsilon = agent.eps_min
    while done == False:
        g.printBoard()
        # print(g.board)
        if turn == PLAYER:
            row = input('{}\'s turn:'.format('Red'))
            g.insert(int(row), PLAYER_PIECE)
        else:
            observation = []
            obs = np.zeros((6, 7))
            for row, sublist in enumerate(g.board):
                for col, i in enumerate(sublist):
                    observation.append(i)
                    obs[col, row] = i

            observation = np.asarray(observation)
            action, _ = minimax(np.flipud(obs), 1, -math.inf, math.inf, True)
            if g.check_if_action_valid(action):
                print('{}\'s turn: %d'.format('Yellow') % action)
                g.insert(action, AI_PIECE)
            else:
                while g.check_if_action_valid(action) == False:
                    agent.store_transition(observation, action, -100, observation, done)
                    action = np.random.randint(7)
                print('{}\'s turn: %d'.format('Yellow') % action)
                g.insert(action, AI_PIECE)
            observation_ = []
            for sublist in g.board:
                for i in sublist:
                    observation_.append(i)
            observation_ = np.asarray(observation_)
            transitions_agent += [(observation, action, observation_, done)]
        turn = (turn + 1) % 2
    return


def play_random_agent():
    global horizontal_win
    global win_cntr
    global done
    done = False
    horizontal_win = False
    g = Game()
    turn = random.choice([PLAYER, AI])
    transitions_agent = []
    while done == False:
        #g.printBoard()
        if turn == PLAYER:
            action = np.random.randint(7)
            if g.check_if_action_valid(action):
                #print('{}\'s turn: %d'.format('Red')%action)
                g.insert(action, PLAYER_PIECE)
            else:
                while g.check_if_action_valid(action) == False:
                    action = np.random.randint(7)
                g.insert(action,PLAYER_PIECE)
        else:
            observation = []
            for sublist in g.board:
                for i in sublist:
                    observation.append(i)
            observation = np.asarray(observation)
            action = agent.choose_action(observation)
            if g.check_if_action_valid(action):
                #print('{}\'s turn: %d'.format('Yellow')%action)
                g.insert(action,AI_PIECE)
            else:
                while g.check_if_action_valid(action) == False:
                    agent.store_transition(observation, action, -100, observation, done)
                    action = agent.choose_action(observation)
                #print('{}\'s turn: %d'.format('Yellow')%action)
                g.insert(action,AI_PIECE)
            observation_ = []
            for sublist in g.board:
                for i in sublist:
                    observation_.append(i)
            observation_ = np.asarray(observation_)
            transitions_agent += [(observation, action, observation_, done)]
        turn = AI if turn == PLAYER else PLAYER
    if g.getWinner() == Tie:
        reward = 0
    winner = AI if turn == PLAYER else PLAYER
    if winner == AI:
        win_cntr += 1
        if horizontal_win:
            reward = 5
        else:
            reward = 60
    else:
        reward = -20
    for i in range(len(transitions_agent)):
        agent.store_transition(transitions_agent[i][0], transitions_agent[i][1], reward, transitions_agent[i][2], transitions_agent[i][3])
    agent.learn()
    return


def play_gui():
    """
        Here we can play against the minimax in the GUI
    """
    global done
    GAME_OVER = False
    pygame.init()
    board = create_board()

    screen = pygame.display.set_mode(SIZE)
    draw_board(board, screen)
    pygame.display.update()

    myfont = pygame.font.SysFont("monospace", 75)
    turn = np.random.randint(0, 2)

    while not GAME_OVER:
        g = Game()
        done = False
        transitions_agent = []

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

            if event.type == pygame.MOUSEMOTION:
                pygame.draw.rect(screen, black, (0, 0, WIDTH, SQUARESIZE))
                posx = event.pos[0]
                if turn == PLAYER:
                    pygame.draw.circle(screen, red, (posx, int(SQUARESIZE / 2)), RADIUS)
            pygame.display.update()

            if event.type == pygame.MOUSEBUTTONDOWN:
                pygame.draw.rect(screen, black, (0, 0, WIDTH, SQUARESIZE))

                if turn == PLAYER:
                    posx = event.pos[0]
                    col = int(math.floor(posx / SQUARESIZE))

                    if is_valid_location(board, col):
                        row = get_next_open_row(board, col)
                        drop_piece(board, row, col, PLAYER_PIECE)

                        if winning_move(board, PLAYER_PIECE):
                            label = myfont.render("Player 1 wins!!", 1, red)
                            screen.blit(label, (40, 10))
                            GAME_OVER = True

                        turn = (turn + 1) % 2
                        draw_board(board, screen)

            # # Ask for Player 2 Input
        if turn == AI and not GAME_OVER:
            observation = []
            obs = np.zeros((6, 7))
            for row, sublist in enumerate(g.board):
                for col, i in enumerate(sublist):
                    observation.append(i)
                    obs[col, row] = i

            observation = np.asarray(observation)
            col = agent.choose_action(observation)

            if is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, AI_PIECE)

                if winning_move(board, AI_PIECE):
                    label = myfont.render("Player 2 wins!!", 1, yellow)
                    screen.blit(label, (40, 10))
                    GAME_OVER = True

                draw_board(board, screen)
                turn = (turn + 1) % 2

            else:
                print("AI random choice")
                col = np.random.randint(7)
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, AI_PIECE)

                if winning_move(board, AI_PIECE):
                    label = myfont.render("Player 2 wins!!", 1, yellow)
                    screen.blit(label, (40, 10))
                    GAME_OVER = True

                draw_board(board, screen)
                turn = (turn + 1) % 2


# Initialize hyperparameters and metrics
train_games = 1
win_cntr = 0
lose_cntr = 0
eps_dec = 1 / train_games
win_array = []

decay = 0.05
agent = Agent(0.99, 1, 0.01, 42, 128, 7, eps_dec)
agent.load_checkpoint()

# Start reinforcement learning process
for i in range(train_games):
    win_array.append((win_cntr / (i + 1)))
    if i % 100 == 0:
        print(f"Episode {i} trained successfully | Games Won: {win_cntr / (i + 1)} | Games Lost: {lose_cntr / (i + 1)}")
    play_minimax(1)

print(win_cntr)
print(win_cntr / train_games)
agent.save_checkpoint()

plt.plot(np.arange(1, train_games + 1), win_array)
plt.xticks(np.linspace(0, 20000, 11))
plt.xlabel("Episodes")
plt.ylabel("Win ratio")
plt.show()

play_gui()
