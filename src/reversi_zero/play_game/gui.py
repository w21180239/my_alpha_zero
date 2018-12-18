from logging import getLogger

from reversi_zero.config import Config, GuiConfig, PlayWithHumanConfig
import reversi_zero.env.reversi_env as reversiEnv
from reversi_zero.play_game.game_model import PlayWithHuman, GameEvent
from reversi_zero.lib.bitboard import board_to_string


logger = getLogger(__name__)


class AIvsHuman:
    def __init__(self, model: PlayWithHuman):
        self.model = model
        self.new_game(human_is_black=True)
        self.model.add_observer(self.handle_game_event)

    def handle_game_event(self, event):
        if event == GameEvent.update:
            self.update_status_bar()
        elif event == GameEvent.over:
            self.game_over()
        elif event == GameEvent.ai_move:
            self.ai_move()

    def new_game(self, human_is_black):
        self.model.start_game(human_is_black)
        self.model.play_next_turn()

    def ai_move(self):
        action=self.model.move_by_ai()
        if action:
            ai_x,ai_y = action_to_cor(action)
            print("ai落子x:"+str(ai_x))
            print("ai落子y:" + str(ai_y))
        self.model.play_next_turn()

    def try_move(self):
        if self.model.over:
            return
        while 1:
            x = eval(input("请输入x："))
            y = eval(input("请输入y："))
            print(x)
            print(y)
            if self.model.available(y, x):
                break

            print("该位置不能下，请重新输入。")

        self.model.move(y, x) #move的横纵坐标反了
        print(board_to_string(self.model.env.board.white, self.model.env.board.black))
        self.model.play_next_turn()

    def game_over(self):
        black, white = self.model.number_of_black_and_white
        mes = "black: %d\nwhite: %d\n" % (black, white)
        if black == white:
            mes += "** draw **"
        else:
            mes += "winner: %s" % ["black", "white"][black < white]
        print(mes)

    def update_status_bar(self):  # 考虑改个名
        print("current player is " + ["White", "Black"][self.model.next_player == reversiEnv.Player.black])
        if self.model.last_evaluation:
            print(self.model.last_evaluation)

    def refresh(self, event):
        self.update_status_bar()

def action_to_cor(action):
    for i in range(8):
        for j in range(8):
            if i*8+j == action:
                return i,j

def start(config: Config):
    config.play_with_human.update_play_config(config.play)
    reversi_model = PlayWithHuman(config)
    temp = AIvsHuman(reversi_model)
    MainLoop(temp)  # 自己定义


def MainLoop(temp: AIvsHuman):
    num = 0
    while not temp.model.over:
        temp.try_move()
        print("round:"+str(num))
        print(board_to_string(temp.model.env.board.white ,temp.model.env.board.black))
        num += 1
    print("finish")
