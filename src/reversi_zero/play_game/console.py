from ..config import Config
from ..env import reversi_env as reversiEnv
from ..lib.bitboard import board_to_string
from ..play_game.game_model import PlayWithHuman, GameEvent


# logger = getLogger(__name__)


class AIvsHuman:
    def __init__(self, model: PlayWithHuman):
        self.model = model
        choose = eval(input("选择先后手（1为先手，0为后手）:\n>>"))
        self.new_game(human_is_black=choose)
        self.role = choose  # 1为黑棋，0为白棋
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
        print("AI is thinking...")
        action = self.model.move_by_ai()

        if type(action) != bool:
            ai_x, ai_y = action_to_cor(action)
            print("ai落子x:" + str(ai_x))
            print("ai落子y:" + str(ai_y))
        else:
            print("AI被你打的跳步了，帅逼！！！！！")
        self.model.play_next_turn()

    def try_move(self):
        if self.model.over:
            return
        while True:
            print('请输入row col:\n>>', end="")
            tmp = input()
            tmp = tmp.split()
            x = int(tmp[0])
            y = int(tmp[1])
            if self.model.available(y, x):
                break

            print("该位置不能下，请重新输入。")

        self.model.move(y, x)
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

    def update_status_bar(self): 
        print("current player is " + ["White", "Black"][self.model.next_player == reversiEnv.Player.black])
        if self.model.last_evaluation:
            print(f"AI Confidence = {self.model.last_evaluation*100:.4f}%")
        # self.SetStatusText(msg)
    def refresh(self, event):
        self.update_status_bar()


def action_to_cor(action):
    for i in range(8):
        for j in range(8):
            if i * 8 + j == action:
                return i, j


def start(config: Config):
    config.play_with_human.update_play_config(config.play)
    reversi_model = PlayWithHuman(config)
    temp = AIvsHuman(reversi_model)
    MainLoop(temp)


def MainLoop(temp: AIvsHuman):
    num = 0
    print(board_to_string(temp.model.env.board.white, temp.model.env.board.black))
    if temp.role == 0:
        temp.handle_game_event(GameEvent.ai_move)
    while not temp.model.over:
        temp.try_move()
        print("round:" + str(num))
        print(board_to_string(temp.model.env.board.white, temp.model.env.board.black))
        temp.update_status_bar()
        num += 1
    print("finish")
