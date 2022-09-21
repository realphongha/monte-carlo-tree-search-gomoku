import pygame
import random
from typing import Tuple
from pygame import Rect
from pygame.font import Font
from utils import constants, exception
from .board import MnkBoard
from .mcts_mnkgame import mcts_solve


pygame.init()


class Game:
    NOT_STARTED = 0
    PLAYER_TURN = 1
    BOT_TURN = 2
    ENDED = 3

    def __init__(self, m: int, n: int, k: int, cell_size: int, name: str, 
            fps: int, menu_font_size: int, button_font_size: int, 
            symbol_font_size: int) -> None:
        # MAGIC NUMBERS: 1 and 2 are player symbols, 0 is empty cell
        self.cell_size = cell_size
        self.m, self.n, self.k = m, n, k
        self.w = m * cell_size
        self.h = n * cell_size * 8 // 7
        self.center_w = self.w / 2
        self.center_h = self.h / 2
        self.menu_font_size = menu_font_size
        self.button_font_size = button_font_size
        self.symbol_font_size = symbol_font_size
        # self.button_font_size = self.h // 15
        # self.button_font_size_small = self.h // 15 * 3 // 4
        # self.symbol_font_size = self.h // 15
        self.board =  MnkBoard(m, n, k)
        self.title = name.title()
        self.fps = fps
        self.turn_symbols = {0: None, 1: "X", 2: "O"}
        self.turn_colors = {0: None, 1: constants.PENCIL_COLOR, 
            2: constants.PENCIL_COLOR}
        self.window = pygame.display.set_mode((self.w, self.h))
        self.clock = pygame.time.Clock()
        self.state = Game.NOT_STARTED
        pygame.display.set_caption(self.title)
        self.cursor = pygame.SYSTEM_CURSOR_ARROW
        self.rect_cache = dict()
        self.bot = None
        self.bot_config = None

    def set_bot(self, name: str):
        self.bot = name

    def add_bot_config(self, cfg):
        self.bot_config = cfg

    def bot_play(self) -> None:
        if self.bot is None:
            pos = self.board.get_possible_pos()
            index = random.randrange(0, len(pos))
            i, j = pos[index]
        elif self.bot == "mcts":
            res = mcts_solve(**self.bot_config, board=self.board, turn=1)
            if res == (-1, -1):
                raise Exception("MCTS doesn't yield result!")
            else:
                i, j = res
        else:
            raise NotImplementedError("%s algorithm is not implemented!" % 
                self.bot)
        self.board.put(2, (i, j))

    def add_rect_to_cache(self, rect: Rect, left: int, top: int, 
            width: int, height: int):
        self.rect_cache[(left, top, width, height)] = rect

    def get_rect_from_cache(self, left: int, top: int, 
            width: int, height: int):
        coor = (left, top, width, height)
        return self.rect_cache.get(coor, None)

    def clear_rect_cache(self):
        self.rect_cache = dict()

    def render_rect(self, position, color, 
                    rect_width=1, border_radius=0, border_top_left_radius=-1, 
                    border_top_right_radius=-1, border_bottom_left_radius=-1, 
                    border_bottom_right_radius=-1,
                    text_str=None, text_size=25, text_color=constants.BLACK):
        left, top, width, height = position
        rect = self.get_rect_from_cache(left, top, width, height)
        if rect is None:
            rect = Rect(left, top, width, height)
            self.add_rect_to_cache(rect, left, top, width, height)
        pygame.draw.rect(self.window, color, rect, rect_width, border_radius,
            border_top_left_radius, border_top_right_radius,
            border_bottom_left_radius, border_bottom_right_radius)
        if text_str:
            font = Font(None, text_size)
            text = font.render(text_str, True, text_color)
            text_rect = text.get_rect(center=(left + width//2, top + height//2))
            self.window.blit(text, text_rect)
        return rect

    def render_board(self):
        rects = list()
        for i in range(self.n):
            rects.append(list())
            for j in range(self.m):
                symbol = self.board.board[i][j]
                rects[i].append(
                    self.render_rect((j*self.cell_size, i*self.cell_size, 
                    self.cell_size, self.cell_size), constants.PENCIL_COLOR, 
                    rect_width=1, text_str=self.turn_symbols[symbol], 
                    text_size=self.symbol_font_size, 
                    text_color=self.turn_colors[symbol])
                )
        return rects

    def render_start_screen(self) -> Tuple[Rect, Rect]:
        return \
        self.render_rect((self.center_w-self.w//4, self.center_h-self.h//16*5, 
            self.w//2, self.h//4), constants.ORANGE, rect_width=0,
            border_radius=self.cell_size,
            text_str="You go first", text_size=self.menu_font_size,
            text_color=constants.WHITE), \
        self.render_rect((self.center_w-self.w//4, self.center_h+self.h//16, 
            self.w//2, self.h//4), constants.ORANGE, rect_width=0,
            border_radius=self.cell_size, 
            text_str="Bot goes first", text_size=self.menu_font_size,
            text_color=constants.WHITE)

    def render_ingame_button(self) -> Tuple[Rect, Rect]:
        return \
        self.render_rect((0, self.n * self.cell_size+1, 
            self.w//4, self.n*self.cell_size//7 - 2), constants.ORANGE, 
            rect_width=0, border_radius=self.cell_size//4,
            text_str="Back to menu", text_size=self.button_font_size,
            text_color=constants.WHITE), \
        self.render_rect((self.w-self.w//4, self.n * self.cell_size+1, 
            self.w//4, self.n*self.cell_size//7 - 2), constants.ORANGE, 
            rect_width=0, border_radius=self.cell_size//4, 
            text_str="Reset board", text_size=self.button_font_size,
            text_color=constants.WHITE)

    def render_endgame_noti(self, res):
        who_won = "You" if res == 1 else \
            ("Computer" if res == 2 else "No one")
        text_color = constants.GREEN if res == 1 else \
            (constants.RED if res == 2 else constants.BLACK)
        self.render_rect((self.w//4, self.n * self.cell_size, 
            self.w//2, self.n*self.cell_size//7), constants.WHITE, rect_width=-1,
            text_str="%s won!" % who_won, 
            text_size=self.button_font_size,
            text_color=text_color)

    def main(self):
        res = 0
        while True:
            pygame.mouse.set_cursor(self.cursor)
            self.window.fill(constants.PAPER_WHITE_COLOR)
            if self.state == Game.NOT_STARTED:
                you_rect, bot_rect = self.render_start_screen()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()
                    if event.type == pygame.MOUSEMOTION:
                        if you_rect.collidepoint(event.pos) or \
                            bot_rect.collidepoint(event.pos):
                            self.cursor = pygame.SYSTEM_CURSOR_HAND
                        else:
                            self.cursor = pygame.SYSTEM_CURSOR_ARROW
                    if event.type == pygame.MOUSEBUTTONUP and \
                            event.button == 1:  # left mouse
                        if you_rect.collidepoint(event.pos):
                            self.state = Game.PLAYER_TURN
                            self.cursor = pygame.SYSTEM_CURSOR_HAND
                            self.clear_rect_cache()
                        elif bot_rect.collidepoint(event.pos):
                            self.state = Game.BOT_TURN
                            self.cursor = pygame.SYSTEM_CURSOR_HAND
                            self.clear_rect_cache()
            else:
                if self.state != Game.ENDED:
                    if len(self.board.get_possible_pos()) == 0:
                        self.state = Game.ENDED
                        res = 0
                rects = self.render_board()
                back_rect, reset_rect = self.render_ingame_button()
                try:
                    remaining_events = list()
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            quit()
                        if event.type == pygame.MOUSEBUTTONUP and \
                            event.button == 1:  # left mouse
                            if back_rect.collidepoint(event.pos):
                                self.state = Game.NOT_STARTED
                                self.clear_rect_cache()
                                raise exception.Break
                            elif reset_rect.collidepoint(event.pos):
                                self.state = random.choice((
                                    Game.PLAYER_TURN, Game.BOT_TURN
                                ))
                                self.board.reset_board()
                                raise exception.Break
                        remaining_events.append(event)
                            
                    if self.state == Game.PLAYER_TURN:
                        for event in remaining_events:
                            if event.type == pygame.MOUSEBUTTONUP and \
                                    event.button == 1:  # left mouse
                                for i, row in enumerate(rects):
                                    for j, rect in enumerate(row):
                                        if rect.collidepoint(event.pos):
                                            if self.board.board[i][j] != 0:
                                                raise exception.Break
                                            self.board.put(1, (j, i))
                                            res = self.board.check_endgame()
                                            if res:
                                                self.state = Game.ENDED
                                            else:
                                                self.state = Game.BOT_TURN
                                            raise exception.Break

                    elif self.state == Game.BOT_TURN:
                        self.bot_play()
                        res = self.board.check_endgame()
                        if res:
                            self.state = Game.ENDED
                        else:
                            self.state = Game.PLAYER_TURN
                        raise exception.Break
                    elif self.state == Game.ENDED:
                        self.render_endgame_noti(res)
                    else:
                        raise NotImplementedError("State %i is not implemented!" % 
                            self.state)
                except exception.Break:
                    pass
            pygame.display.update()
            self.clock.tick(self.fps)


if __name__ == "__main__":
    pass
