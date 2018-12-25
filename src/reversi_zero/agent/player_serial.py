import asyncio
import time
from collections import defaultdict, namedtuple
from logging import getLogger

import numpy as np
from numpy.random import random

from ..agent.api import ReversiModelAPI
from ..config import Config
from ..env.reversi_env import ReversiEnv, Player, Winner, another_player
from ..lib.bitboard import find_correct_moves, bit_to_array, flip_vertical, rotate90, dirichlet_noise_of_mask

CounterKey = namedtuple("CounterKey", "black white next_player")
HistoryItem = namedtuple("HistoryItem", "action policy values visit enemy_values enemy_visit")
MCTSInfo = namedtuple("MCTSInfo", "var_n var_w var_p")
ActionWithEvaluation = namedtuple("ActionWithEvaluation", "action n q")

logger = getLogger(__name__)


class ReversiPlayer:
    def __init__(self, config: Config, model, play_config=None, enable_resign=True, mtcs_info=None, api=None):
        self.config = config
        self.model = model
        self.play_config = play_config or self.config.play
        self.enable_resign = enable_resign
        self.api = api or ReversiModelAPI(self.config, self.model)

        mtcs_info = mtcs_info or self.create_mtcs_info()
        self.var_n, self.var_w, self.var_p = mtcs_info

        self.expanded = set(self.var_p.keys())
        self.now_expanding = set()

        self.moves = []
        self.loop = asyncio.get_event_loop()
        self.running_simulation_num = 0
        self.callback_in_mtcs = None

        self.thinking_history = {}
        self.resigned = False
        self.requested_stop_thinking = False

    def var_q(self, key):
        return self.var_w[key] / (self.var_n[key] + 1e-5)  # 防止除零

    def action(self, own, enemy, callback_in_mtcs=None):
        if self.config.play.enable_flash_mode:
            game = ReversiEnv().update(own, enemy, Player.black)
            black, white = game.board.black, game.board.white
            black_ary = bit_to_array(black, 64).reshape((8, 8))
            white_ary = bit_to_array(white, 64).reshape((8, 8))
            state = [black_ary, white_ary] if game.next_player == Player.black else [white_ary, black_ary]
            result = self.predict(np.array(state))
            self.update_thinking_history(black, white, int(np.argmax(result[0])), result[0])
            return int(np.argmax(result[0]))
        action_with_eval = self.action_with_evaluation(own, enemy, callback_in_mtcs=callback_in_mtcs)

        return action_with_eval.action

    def action_with_evaluation(self, me, enemy, callback_in_mtcs=None):
        game = ReversiEnv().update(me, enemy, Player.black)
        key = self.counter_key(game)
        self.callback_in_mtcs = callback_in_mtcs
        pc = self.play_config

        # 计时
        start = time.time()
        original_expanded_size = len(self.expanded)
        for tl in range(pc.thinking_loop):
            if pc.enable_max_dic_size:
                self.limit_size()
            if game.turn > 0:
                self.search_moves(me, enemy, start, original_expanded_size)
            else:
                self.bypass_first_move(key)
            logger.debug(f'Expanded node list size:{len(self.expanded)}')
            policy = self.calc_policy(me, enemy)
            action = int(np.random.choice(range(64), p=policy))  # 过去走的最多的路：蕴含了NN算出来的信息
            action_by_value = int(np.argmax(self.var_q(key) + (self.var_n[key] > 0) * 100))  # 胜率最高的路
            value_diff = self.var_q(key)[action] - self.var_q(key)[action_by_value]
            # 如果不一样且还有时间，再次思考
            if game.turn <= pc.start_rethinking_turn or self.requested_stop_thinking or \
                    (value_diff > -0.01 and self.var_n[key][action] >= pc.required_visit_to_decide_action) or \
                    time.time() - start > self.config.play.max_search_time:
                break
        print(f'Total time:{time.time()-start}')

        # 训练的时候可以不用
        self.update_thinking_history(me, enemy, action, policy)
        # 步数过多时提前结束
        if self.play_config.resign_threshold is not None and \
                np.max(self.var_q(key) - (self.var_n[key] == 0) * 10) <= self.play_config.resign_threshold:
            self.resigned = True
            if self.enable_resign:
                if game.turn >= self.config.play.allowed_resign_turn:
                    return ActionWithEvaluation(None, 0, 0)  # means resign
                else:
                    logger.debug(
                        f"Want to resign but disallowed turn {game.turn} < {self.config.play.allowed_resign_turn}")

        saved_policy = self.calc_policy_by_tau_1(key) if self.config.play_data.save_policy_of_tau_1 else policy
        # 将棋局拷贝成8个对称方向，增大数据量
        self.store_data_with_8_symmetries(me, enemy, saved_policy)
        return ActionWithEvaluation(action=action, n=self.var_n[key][action], q=self.var_q(key)[action])

    def update_thinking_history(self, black, white, action, policy):
        key = CounterKey(black, white, Player.black.value)
        next_key = self.get_next_key(black, white, action)
        self.thinking_history[(black, white)] = \
            HistoryItem(action, policy, list(self.var_q(key)), list(self.var_n[key]),
                        list(self.var_q(next_key)), list(self.var_n[next_key]))

    def bypass_first_move(self, key):  # 先手第一步随便下
        legal_array = bit_to_array(find_correct_moves(key.black, key.white), 64)
        action = np.argmax(legal_array)
        self.var_n[key][action] = 1
        self.var_w[key][action] = 0
        self.var_p[key] = legal_array / np.sum(legal_array)

    def stop_thinking(self):
        self.requested_stop_thinking = True

    def store_data_with_8_symmetries(self, own, enemy, policy):
        for flip in [False, True]:
            for rot_right in range(4):
                own_saved, enemy_saved, policy_saved = own, enemy, policy.reshape((8, 8))
                if flip:
                    own_saved = flip_vertical(own_saved)
                    enemy_saved = flip_vertical(enemy_saved)
                    policy_saved = np.flipud(policy_saved)
                if rot_right:
                    for _ in range(rot_right):
                        own_saved = rotate90(own_saved)
                        enemy_saved = rotate90(enemy_saved)
                    policy_saved = np.rot90(policy_saved, k=-rot_right)
                self.moves.append([(own_saved, enemy_saved), list(policy_saved.reshape((64,)))])

    def get_next_key(self, own, enemy, action):
        env = ReversiEnv().update(own, enemy, Player.black)
        env.step(action)
        return self.counter_key(env)

    def ask_thought_about(self, own, enemy):
        return self.thinking_history.get((own, enemy))

    def search_moves(self, own, enemy, start_time, original_expanded_size):
        '''最外层搜索函数'''
        self.requested_stop_thinking = False
        for i in range(self.config.play.simulation_num_per_move):
            self.start_search_my_move(own, enemy)
            if time.time() - start_time > self.config.play.max_search_time:
                logger.debug(f'Time out!\tTotal searched nodes:{len(self.expanded)-original_expanded_size}')
                break

    def start_search_my_move(self, own, enemy):
        '''一次MCTS搜索'''
        root_key = self.counter_key(ReversiEnv().update(own, enemy, Player.black))
        if self.requested_stop_thinking:
            return None
        env = ReversiEnv().update(own, enemy, Player.black)
        leaf_v = self.search_my_move(env, is_root_node=True)
        if self.callback_in_mtcs and self.callback_in_mtcs.per_sim > 0 and \
                self.running_simulation_num % self.callback_in_mtcs.per_sim == 0:
            self.callback_in_mtcs.callback(list(self.var_q(root_key)), list(self.var_n[root_key]))
        return leaf_v

    def search_my_move(self, env: ReversiEnv, is_root_node=False):
        """
        Q, V 是对黑方的
        P 是对要下棋的一方的
        """
        if env.done:  # 打完了
            if env.winner == Winner.black:
                return 1
            elif env.winner == Winner.white:
                return -1
            else:
                return 0

        key = self.counter_key(env)
        another_side_key = self.another_side_counter_key(env)

        if key not in self.expanded:  # 到达leaf
            leaf_v = self.expand_and_evaluate(env)
            if env.next_player == Player.black:
                return leaf_v  # Value for black
            else:
                return -leaf_v  # Value for white == -Value for black
        # 当前节点不是leaf
        action_t = self.select_action_q_and_u(env, is_root_node)
        _, _ = env.step(action_t)

        leaf_v = self.search_my_move(env)  # next move

        # 递归的更新N和W
        self.var_n[key][action_t] += 1
        self.var_w[key][action_t] += leaf_v
        # 对面的也要更新
        self.var_n[another_side_key][action_t] += 1
        self.var_w[another_side_key][action_t] -= leaf_v  # must flip the sign.
        return leaf_v

    def expand_and_evaluate(self, env):
        """扩展并评价leaf"""
        key = self.counter_key(env)
        another_side_key = self.another_side_counter_key(env)
        self.now_expanding.add(key)

        black, white = env.board.black, env.board.white

        is_flip_vertical = random() < 0.5
        rotate_right_num = int(random() * 4)
        if is_flip_vertical:
            black, white = flip_vertical(black), flip_vertical(white)
        for i in range(rotate_right_num):
            black, white = rotate90(black), rotate90(white)  # rotate90: rotate bitboard RIGHT 1 time

        black_ary = bit_to_array(black, 64).reshape((8, 8))
        white_ary = bit_to_array(white, 64).reshape((8, 8))
        state = [black_ary, white_ary] if env.next_player == Player.black else [white_ary, black_ary]
        result = self.predict(np.array(state))
        leaf_p, leaf_v = result[0], result[1]  # 都是NN算出来的

        if rotate_right_num > 0 or is_flip_vertical:  # reverse rotation and flip. rot -> flip.
            leaf_p = leaf_p.reshape((8, 8))
            if rotate_right_num > 0:
                leaf_p = np.rot90(leaf_p, k=rotate_right_num)  # rot90: rotate matrix LEFT k times
            if is_flip_vertical:
                leaf_p = np.flipud(leaf_p)
            leaf_p = leaf_p.reshape((64,))

        self.var_p[key] = leaf_p
        self.var_p[another_side_key] = leaf_p
        self.expanded.add(key)
        self.now_expanding.remove(key)
        return float(leaf_v)

    def predict(self, x):
        return self.api.predict(x)

    def finish_game(self, z):
        for move in self.moves:  # 对所有路上的节点更新
            move += [z]

    def calc_policy(self, own, enemy):
        """计算 π(a|s0)"""
        pc = self.play_config
        env = ReversiEnv().update(own, enemy, Player.black)
        key = self.counter_key(env)
        if env.turn < pc.change_tau_turn:
            return self.calc_policy_by_tau_1(key)
        else:
            action = np.argmax(self.var_n[key])  # tau = 0
            ret = np.zeros(64)
            ret[action] = 1
            return ret

    def calc_policy_by_tau_1(self, key):
        return self.var_n[key] / np.sum(self.var_n[key])  # tau = 1

    def select_action_q_and_u(self, env, is_root_node):
        key = self.counter_key(env)
        if env.next_player == Player.black:
            legal_moves = find_correct_moves(key.black, key.white)
        else:
            legal_moves = find_correct_moves(key.white, key.black)
        xx_ = np.sqrt(np.sum(self.var_n[key]))
        xx_ = max(xx_, 1)  # 避免为0
        p_ = self.var_p[key]

        # 归一化
        p_ = p_ * bit_to_array(legal_moves, 64)
        if np.sum(p_) > 0:
            _pc = self.config.play
            temperature = min(np.exp(1 - np.power(env.turn / _pc.policy_decay_turn, _pc.policy_decay_power)), 1)
            p_ = self.normalize(p_, temperature)

        if is_root_node and self.play_config.noise_eps > 0:
            noise = dirichlet_noise_of_mask(legal_moves, self.play_config.dirichlet_alpha)
            p_ = (1 - self.play_config.noise_eps) * p_ + self.play_config.noise_eps * noise

        u_ = self.play_config.c_puct * p_ * xx_ / (1 + self.var_n[key])  # 根据alpha zero的公式
        if env.next_player == Player.black:
            v_ = (self.var_q(key) + u_ + 1000) * bit_to_array(legal_moves, 64)
        else:
            v_ = (-self.var_q(key) + u_ + 1000) * bit_to_array(legal_moves, 64)

        action_t = int(np.argmax(v_))
        return action_t

    def limit_size(self):
        if len(self.var_n) > self.play_config.max_dic_size:
            self.var_n = self.var_n[-self.play_config.max_dic_size:]
        if len(self.var_p) > self.play_config.max_dic_size:
            self.var_p = self.var_p[-self.play_config.max_dic_size:]
        if len(self.var_w) > self.play_config.max_dic_size:
            self.var_w = self.var_w[-self.play_config.max_dic_size:]

    @staticmethod
    def normalize(p, t=1):
        pp = np.power(p, t)
        return pp / np.sum(pp)

    @staticmethod
    def counter_key(env: ReversiEnv):
        return CounterKey(env.board.black, env.board.white, env.next_player.value)

    @staticmethod
    def another_side_counter_key(env: ReversiEnv):
        return CounterKey(env.board.white, env.board.black, another_player(env.next_player).value)

    @staticmethod
    def create_mtcs_info():
        return MCTSInfo(defaultdict(lambda: np.zeros((64,))),
                        defaultdict(lambda: np.zeros((64,))),
                        defaultdict(lambda: np.zeros((64,))))
