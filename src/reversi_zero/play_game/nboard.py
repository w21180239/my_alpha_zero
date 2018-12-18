import re
import sys
from collections import namedtuple
from logging import getLogger, StreamHandler, FileHandler
from time import time

from ..agent.player_serial import ReversiPlayer, CallbackInMCTS
from ..config import Config
from ..env.reversi_env import ReversiEnv, Player
from ..lib.ggf import parse_ggf, convert_to_bitboard_and_actions, convert_move_to_action, \
    convert_action_to_move
from ..lib.nonblocking_stream_reader import NonBlockingStreamReader
from ..play_game.common import load_model

logger = getLogger(__name__)

GameState = namedtuple("GameState", "black white actions player")
GoResponse = namedtuple("GoResponse", "action eval time")
HintResponse = namedtuple("HintResponse", "action value visit")


def start(config: Config):
    config.play_with_human.update_play_config(config.play)
    root_logger = getLogger()
    for h in root_logger.handlers:
        if isinstance(h, StreamHandler) and not isinstance(h, FileHandler):
            root_logger.removeHandler(h)
    logger.info(f"config type={config.type}")
    NBoardEngine(config).start()
    logger.info("finish nboard")


class NBoardEngine:
    def __init__(self, config: Config):
        self.config = config
        self.reader = NonBlockingStreamReader(sys.stdin)
        self.handler = NBoardProtocolVersion2(config, self)
        self.running = False
        self.nc = self.config.nboard

        self.env = ReversiEnv().reset()
        self.model = load_model(self.config)
        self.play_config = self.config.play
        self.player = self.create_player()
        self.turn_of_nboard = None

    def create_player(self):
        logger.debug("create new ReversiPlayer()")
        return ReversiPlayer(self.config, self.model, self.play_config, enable_resign=False)

    def start(self):
        self.running = True
        self.reader.start(push_callback=self.push_callback)
        while self.running and not self.reader.closed:
            message = self.reader.readline(self.nc.read_stdin_timeout)
            if message is None:
                continue
            message = message.strip()
            logger.debug(f"> {message}")
            self.handler.handle_message(message)

    def push_callback(self, message: str):

        if message.startswith("ping"):
            self.stop_thinkng()

    def stop(self):
        self.running = False

    def reply(self, message):
        logger.debug(f"< {message}")
        sys.stdout.write(message + "\n")
        sys.stdout.flush()

    def stop_thinkng(self):
        self.player.stop_thinking()

    def set_depth(self, n):
        try:
            n = int(n)
            self.play_config.required_visit_to_decide_action = n * self.nc.simulation_num_per_depth_about
            self.play_config.thinking_loop = min(
                30,
                int(self.play_config.required_visit_to_decide_action * 5 / self.play_config.simulation_num_per_move)
            )

            logger.info(f"set required_visit_to_decide_action to {self.play_config.required_visit_to_decide_action}")
        except ValueError:
            pass

    def reset_state(self):
        self.player = self.create_player()

    def set_game(self, game_state: GameState):
        self.env.reset()
        self.env.update(game_state.black, game_state.white, game_state.player)
        self.turn_of_nboard = game_state.player
        for action in game_state.actions:
            self._change_turn()
            if action is not None:
                self.env.step(action)

    def _change_turn(self):
        if self.turn_of_nboard:
            self.turn_of_nboard = Player.black if self.turn_of_nboard == Player.white else Player.white

    def move(self, action):
        self._change_turn()
        if action is not None:
            self.env.step(action)

    def go(self) -> GoResponse:
        if self.env.next_player != self.turn_of_nboard:
            return GoResponse(None, 0, 0)

        board = self.env.board
        if self.env.next_player == Player.black:
            states = (board.black, board.white)
        else:
            states = (board.white, board.black)
        start_time = time()
        action = self.player.action(*states)
        item = self.player.ask_thought_about(*states)
        evaluation = item.values[action]
        time_took = time() - start_time
        return GoResponse(action, evaluation, time_took)

    def hint(self, n_hint):

        board = self.env.board
        if self.env.next_player == Player.black:
            states = (board.black, board.white)
        else:
            states = (board.white, board.black)

        def hint_report_callback(values, visits):
            hint_list = []
            for action, visit in list(sorted(enumerate(visits), key=lambda x: -x[1]))[:n_hint]:
                if visit > 0:
                    hint_list.append(HintResponse(action, values[action], visit))
            self.handler.report_hint(hint_list)

        callback_info = CallbackInMCTS(self.config.nboard.hint_callback_per_sim, hint_report_callback)
        self.player.action(*states, callback_in_mtcs=callback_info)
        item = self.player.ask_thought_about(*states)
        hint_report_callback(item.values, item.visit)


class NBoardProtocolVersion2:
    def __init__(self, config: Config, engine: NBoardEngine):
        self.config = config
        self.engine = engine
        self.handlers = [
            (re.compile(r'nboard ([0-9]+)'), self.nboard),
            (re.compile(r'set depth ([0-9]+)'), self.set_depth),
            (re.compile(r'set game (.+)'), self.set_game),
            (re.compile(r'move ([^/]+)(/[^/]*)?(/[^/]*)?'), self.move),
            (re.compile(r'hint ([0-9]+)'), self.hint),
            (re.compile(r'go'), self.go),
            (re.compile(r'ping ([0-9]+)'), self.ping),
            (re.compile(r'learn'), self.learn),
            (re.compile(r'analyze'), self.analyze),
        ]

    def handle_message(self, message):
        for regexp, func in self.handlers:
            if self.scan(message, regexp, func):
                return
        logger.debug(f"ignore message: {message}")

    def scan(self, message, regexp, func):
        match = regexp.match(message)
        if match:
            func(*match.groups())
            return True
        return False

    def nboard(self, version):
        if version != "2":
            logger.warning(f"UNKNOWN NBoard Version {version}!!!")
        self.engine.reply(f"set myname {self.config.nboard.my_name}({self.config.type})")
        self.tell_status("waiting")

    def set_depth(self, depth):

        self.engine.set_depth(depth)

    def set_game(self, ggf_str):

        ggf = parse_ggf(ggf_str)
        black, white, actions = convert_to_bitboard_and_actions(ggf)
        player = Player.black if ggf.BO.color == "*" else Player.white
        self.engine.set_game(GameState(black, white, actions, player))


        if len(actions) <= 1:
            self.engine.reset_state()

    def move(self, move, evaluation, time_sec):



        action = convert_move_to_action(move)
        self.engine.move(action)

    def hint(self, n):

        self.tell_status("thinkng hint...")
        self.engine.hint(int(n))
        self.tell_status("waiting")

    def report_hint(self, hint_list):
        for hint in reversed(hint_list):
            move = convert_action_to_move(hint.action)
            self.engine.reply(f"search {move} {hint.value} 0 {int(hint.visit)}")

    def go(self):

        self.tell_status("thinking...")
        gr = self.engine.go()
        move = convert_action_to_move(gr.action)
        self.engine.reply(f"=== {move}/{gr.eval * 10}/{gr.time}")
        self.tell_status("waiting")

    def ping(self, n):


        self.engine.reply(f"pong {n}")

    def learn(self):

        self.engine.reply("learned")

    def analyze(self):

        pass

    def tell_status(self, status):
        self.engine.reply(f"status {status}")


