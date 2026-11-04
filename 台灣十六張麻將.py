"""
台灣十六張麻將：蒙地卡羅模擬、聽牌巡數迴歸與期望值評估系統
Taiwan Mahjong Monte Carlo Simulation, Tenpai Turn Regression & Expected Value Framework

核心架構與方法論特徵：
1. 手牌與計數向量同步封裝 (PlayerHandState)：以 __slots__ 同步維護手牌清單與 34 維計數向量，內建榮和試探與終局一致性斷言。
2. 消融實驗參數化 (Ablation Parameterization)：支援 enable_chow 與 enable_tai 獨立開關，可量化吃牌對聽牌巡數之邊際影響。
3. 未加權底數與底台計分敏感度對照：並列輸出未加權底數基準 (Base_EV) 與 300/100、100/20、50/20 底台結構之損益兩平點。
4. 統計審計透明度：顯性輸出右設限截斷比例 (Censoring_Rate)、單調性檢驗與圖表樣本過濾門檻 (1D 曲線 N>=10，2D 熱力圖 N>=20)。
"""

import random
import time
import csv
from functools import lru_cache
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

TILE_NAMES = [
    "1萬", "2萬", "3萬", "4萬", "5萬", "6萬", "7萬", "8萬", "9萬",
    "1筒", "2筒", "3筒", "4筒", "5筒", "6筒", "7筒", "8筒", "9筒",
    "1條", "2條", "3條", "4條", "5條", "6條", "7條", "8條", "9條",
    "東", "南", "西", "北", "中", "發", "白"
]


class PlayerHandState:
    """
    手牌與計數向量同步容器：
    封裝玩家手牌清單與 34 維計數向量，確保摸牌、捨牌、補花、榮和試探與吃碰時同步更新。
    """
    __slots__ = ('hand', 'counts')

    def __init__(self, initial_hand=None):
        self.hand = []
        self.counts = [0] * 34
        if initial_hand:
            for t in initial_hand:
                self.add(t)

    def add(self, tile):
        self.hand.append(tile)
        if tile < 34:
            self.counts[tile] += 1

    def remove(self, tile):
        self.hand.remove(tile)
        if tile < 34:
            self.counts[tile] -= 1

    def sort(self):
        self.hand.sort()

    def can_win_with(self, tile, evaluator, melds_count):
        """封裝榮和試探邏輯：確保計數增減與狀態復原完全收斂於類別內部"""
        if tile >= 34:
            return False
        self.counts[tile] += 1
        try:
            return evaluator.is_win_from_counts(self.counts, melds_count)
        finally:
            self.counts[tile] -= 1

    def verify_consistency(self):
        """審計與測試不變量檢查：34 維計數總和必須恆等於非花牌手牌總數"""
        non_flower_len = sum(1 for t in self.hand if t < 34)
        assert sum(self.counts) == non_flower_len, (
            f"狀態不同步！Counts 總和: {sum(self.counts)} != 非花牌手牌數: {non_flower_len}"
        )


@lru_cache(maxsize=32768)
def _evaluate_single_suit(suit_counts):
    counts = list(suit_counts)
    results = set()

    def dfs(idx, m, t, p, pair_used_at_idx=False):
        if idx >= 9:
            results.add((m, t, p))
            return

        c = counts[idx]
        if c == 0:
            dfs(idx + 1, m, t, p, False)
            return

        if c >= 3:
            counts[idx] -= 3
            dfs(idx, m + 1, t, p, pair_used_at_idx)
            counts[idx] += 3

        if idx <= 6 and counts[idx + 1] > 0 and counts[idx + 2] > 0:
            counts[idx] -= 1
            counts[idx + 1] -= 1
            counts[idx + 2] -= 1
            dfs(idx, m + 1, t, p, pair_used_at_idx)
            counts[idx] += 1
            counts[idx + 1] += 1
            counts[idx + 2] += 1

        if c >= 2 and p == 0 and not pair_used_at_idx:
            counts[idx] -= 2
            dfs(idx, m, t, 1, True)
            counts[idx] += 2

        if idx <= 7 and counts[idx + 1] > 0:
            counts[idx] -= 1
            counts[idx + 1] -= 1
            dfs(idx, m, t + 1, p, pair_used_at_idx)
            counts[idx] += 1
            counts[idx + 1] += 1

        if idx <= 6 and counts[idx + 2] > 0:
            counts[idx] -= 1
            counts[idx + 2] -= 1
            dfs(idx, m, t + 1, p, pair_used_at_idx)
            counts[idx] += 1
            counts[idx + 2] += 1

        if c >= 2 and not pair_used_at_idx:
            counts[idx] -= 2
            dfs(idx, m, t + 1, p, True)
            counts[idx] += 2

        dfs(idx + 1, m, t, p, False)

    dfs(0, 0, 0, 0, False)

    best_t_for_mp = {}
    for m, t, p in results:
        m_cap = min(m, 5)
        t_cap = min(t, 5 - m_cap)
        key = (m_cap, p)
        if key not in best_t_for_mp or t_cap > best_t_for_mp[key]:
            best_t_for_mp[key] = t_cap

    return tuple((m, t, p) for (m, p), t in best_t_for_mp.items())


@lru_cache(maxsize=65536)
def _combine_suits(wan_res, tong_res, tiao_res, h_melds, h_pairs, target_melds):
    min_shanten = 10
    for m1, t1, p1 in wan_res:
        for m2, t2, p2 in tong_res:
            for m3, t3, p3 in tiao_res:
                total_m = min(target_melds, h_melds + m1 + m2 + m3)
                raw_p = h_pairs + p1 + p2 + p3
                has_pair = 1 if raw_p >= 1 else 0
                extra_pairs_as_taatsu = max(0, raw_p - 1)
                total_t = t1 + t2 + t3 + extra_pairs_as_taatsu
                valid_t = min(total_t, target_melds - total_m)
                s = 2 * (target_melds - total_m) - valid_t - has_pair
                if s < min_shanten:
                    min_shanten = s
    return min_shanten


class TaiwanMahjongEvaluator:
    @staticmethod
    def to_counts(hand):
        if isinstance(hand, PlayerHandState):
            return hand.counts
        counts = [0] * 34
        for tile in hand:
            if tile < 34:
                counts[tile] += 1
        return counts

    @classmethod
    def is_standard_win(cls, counts, melds_count=0):
        target_melds = 5 - melds_count
        if sum(counts) != target_melds * 3 + 2:
            return False

        for i in range(27, 34):
            if counts[i] == 1 or counts[i] == 4:
                return False

        for i in range(34):
            if counts[i] >= 2:
                counts[i] -= 2
                if cls._check_melds(counts, target_melds):
                    counts[i] += 2
                    return True
                counts[i] += 2
        return False

    @classmethod
    def _check_melds(cls, counts, needed_melds):
        if needed_melds == 0:
            return sum(counts) == 0

        first_idx = -1
        for i in range(34):
            if counts[i] > 0:
                first_idx = i
                break
        if first_idx == -1:
            return needed_melds == 0

        if counts[first_idx] >= 3:
            counts[first_idx] -= 3
            if cls._check_melds(counts, needed_melds - 1):
                counts[first_idx] += 3
                return True
            counts[first_idx] += 3

        if first_idx < 27 and (first_idx % 9 <= 6):
            if counts[first_idx + 1] > 0 and counts[first_idx + 2] > 0:
                counts[first_idx] -= 1
                counts[first_idx + 1] -= 1
                counts[first_idx + 2] -= 1
                if cls._check_melds(counts, needed_melds - 1):
                    counts[first_idx] += 1
                    counts[first_idx + 1] += 1
                    counts[first_idx + 2] += 1
                    return True
                counts[first_idx] += 1
                counts[first_idx + 1] += 1
                counts[first_idx + 2] += 1

        return False

    @classmethod
    def is_eight_pairs_win(cls, counts, melds_count=0):
        if melds_count > 0 or sum(counts) != 17:
            return False
        pairs = 0
        triplet = 0
        for c in counts:
            if c == 2:
                pairs += 1
            elif c == 3:
                triplet += 1
            elif c == 4:
                pairs += 2
        return pairs == 7 and triplet == 1

    @classmethod
    def is_win_from_counts(cls, counts, melds_count=0):
        return cls.is_standard_win(counts, melds_count) or cls.is_eight_pairs_win(counts, melds_count)

    @classmethod
    def is_win(cls, hand, melds=[]):
        counts = cls.to_counts(hand)
        return cls.is_win_from_counts(counts, len(melds))

    @classmethod
    def calculate_shanten_from_counts(cls, counts, melds_count=0):
        target_melds = 5 - melds_count
        if sum(counts) == target_melds * 3 + 2:
            if cls.is_standard_win(counts, melds_count) or cls.is_eight_pairs_win(counts, melds_count):
                return -1

        h_melds = 0
        h_pairs = 0
        for i in range(27, 34):
            c = counts[i]
            if c >= 3:
                h_melds += 1
            elif c == 2:
                h_pairs += 1

        wan_res = _evaluate_single_suit(tuple(counts[0:9]))
        tong_res = _evaluate_single_suit(tuple(counts[9:18]))
        tiao_res = _evaluate_single_suit(tuple(counts[18:27]))

        min_shanten = _combine_suits(wan_res, tong_res, tiao_res, h_melds, h_pairs, target_melds)

        if melds_count == 0:
            pair_units = min(8, sum(c // 2 for c in counts))
            has_triplet = 1 if any(c == 3 for c in counts) else 0
            eight_pairs_shanten = 8 - pair_units - has_triplet
            return min(min_shanten, eight_pairs_shanten)

        return min_shanten

    @classmethod
    def calculate_shanten(cls, hand, melds=[]):
        counts = cls.to_counts(hand)
        return cls.calculate_shanten_from_counts(counts, len(melds))


def run_expert_sanity_checks():
    """
    人類領域專家黑盒驗收與對抗性邊界測試（共 7 道自動化斷言檢驗）：
    包含 3 道正向胡聽牌型、3 道非法邊界陷阱牌型 與 1 道 PlayerHandState 同步測試。
    """
    ev = TaiwanMahjongEvaluator
    hand_tenpai = PlayerHandState([0, 1, 2, 12, 13, 14, 24, 25, 26, 27, 27, 27, 26, 26, 4, 5])
    hand_tenpai.verify_consistency()
    assert ev.calculate_shanten(hand_tenpai) == 0, "驗收失敗：標準聽牌型未判定為 0 向聽"

    hand_win = PlayerHandState(hand_tenpai.hand + [3])
    hand_win.verify_consistency()
    assert ev.is_win(hand_win) is True and ev.calculate_shanten(hand_win) == -1, "驗收失敗：標準胡牌型未正確判定"

    hand_ligu = PlayerHandState([0, 0, 2, 2, 9, 9, 13, 13, 18, 18, 27, 27, 31, 31, 32, 32, 32])
    hand_ligu.verify_consistency()
    assert ev.is_win(hand_ligu) is True and ev.calculate_shanten(hand_ligu) == -1, "驗收失敗：嚦咕嚦咕未正確判定"

    hand_cross_suit = PlayerHandState([7, 8, 9, 0, 1, 2, 12, 13, 14, 18, 19, 20, 27, 27, 27, 31, 31])
    assert ev.is_win(hand_cross_suit) is False, "對抗測試失敗：跨花色 89萬1筒 被誤判為順子"

    hand_honor_seq = PlayerHandState([27, 28, 29, 0, 1, 2, 12, 13, 14, 18, 19, 20, 30, 30, 30, 31, 31])
    assert ev.is_win(hand_honor_seq) is False, "對抗測試失敗：字牌 東南西 被誤判為順子"

    hand_fake_ligu = PlayerHandState([0, 0, 2, 2, 4, 4, 9, 9, 13, 13, 18, 18, 27, 27, 31, 31, 32])
    assert ev.is_win(hand_fake_ligu) is False, "對抗測試失敗：8對1單 無刻子牌型被誤判為嚦咕嚦咕"

    print("[專家與對抗測試通過] 3項正向胡聽牌型(True)、3項非法邊界陷阱(False) 與 PlayerHandState 同步斷言全數通過。")


class CMLMahjongTable:
    def __init__(self, dealer_seat=0, enable_chow=True, enable_tai=True):
        self.dealer_seat = dealer_seat
        self.enable_chow = enable_chow
        self.enable_tai = enable_tai
        self.wall = []
        self.players_hands = [PlayerHandState() for _ in range(4)]
        self.players_flowers = [[], [], [], []]
        self.discards = [[], [], [], []]
        self.melds = [[], [], [], []]
        self.init_shantens = [0, 0, 0, 0]
        self.current_shantens = [0, 0, 0, 0]
        self.init_conn_scores = [0, 0, 0, 0]
        self.init_flowers = [0, 0, 0, 0]
        self.first_tenpai_ply = [None, None, None, None]
        self.turn_shanten_history = [{}, {}, {}, {}]
        self.evaluator = TaiwanMahjongEvaluator()

    @staticmethod
    def _tile_connectivity_score(tile, counts):
        if tile >= 27:
            return (counts[tile] - 1) * 50
        pos = tile % 9
        base = 30
        if pos == 0 or pos == 8:
            base = 10
        elif pos == 1 or pos == 7:
            base = 20

        neighbors = (counts[tile] - 1) * 40
        if pos > 0 and counts[tile - 1] > 0:
            neighbors += 25
        if pos < 8 and counts[tile + 1] > 0:
            neighbors += 25
        if pos > 1 and counts[tile - 2] > 0:
            neighbors += 10
        if pos < 7 and counts[tile + 2] > 0:
            neighbors += 10
        return base + neighbors

    def build_wall_and_deal(self):
        full_wall = [tile_id for tile_id in range(34) for _ in range(4)] + list(range(34, 42))
        random.shuffle(full_wall)
        self.wall = full_wall

        for i in range(4):
            self.players_hands[i] = PlayerHandState([self.wall.pop(0) for _ in range(16)])

        self.players_hands[self.dealer_seat].add(self.wall.pop(0))
        self.execute_initial_flower_replacement()

        for i in range(4):
            p_state = self.players_hands[i]
            p_state.sort()
            p_state.verify_consistency()
            counts = p_state.counts

            if i == self.dealer_seat:
                best_s = 999
                for cand in range(34):
                    if counts[cand] > 0:
                        counts[cand] -= 1
                        s = self.evaluator.calculate_shanten_from_counts(counts, 0)
                        counts[cand] += 1
                        if s < best_s:
                            best_s = s
                self.init_shantens[i] = max(0, best_s)
            else:
                self.init_shantens[i] = self.evaluator.calculate_shanten_from_counts(counts, 0)

            self.current_shantens[i] = self.init_shantens[i]
            if self.init_shantens[i] <= 0:
                self.first_tenpai_ply[i] = 1
            self.turn_shanten_history[i][1] = self.init_shantens[i]

            self.init_conn_scores[i] = sum(self._tile_connectivity_score(t, counts) for t in p_state.hand)
            self.init_flowers[i] = len(self.players_flowers[i])

    def execute_initial_flower_replacement(self):
        for i in range(4):
            p_state = self.players_hands[i]
            while True:
                flowers = [t for t in p_state.hand if t >= 34]
                if not flowers:
                    break
                f = flowers[0]
                p_state.remove(f)
                self.players_flowers[i].append(f)
                p_state.add(self.wall.pop(-1))

    def decide_discard(self, player_id):
        p_state = self.players_hands[player_id]
        melds_count = len(self.melds[player_id])
        if len(p_state.hand) == 1:
            tile = p_state.hand[0]
            p_state.remove(tile)
            self.current_shantens[player_id] = 0
            return tile

        counts = p_state.counts
        best_tile = -1
        min_shanten = 999
        min_connectivity = 99999

        for candidate in range(34):
            if counts[candidate] > 0:
                conn_score = self._tile_connectivity_score(candidate, counts)
                counts[candidate] -= 1
                s = self.evaluator.calculate_shanten_from_counts(counts, melds_count)
                counts[candidate] += 1

                if (s < min_shanten) or (s == min_shanten and conn_score < min_connectivity):
                    min_shanten = s
                    min_connectivity = conn_score
                    best_tile = candidate

        p_state.remove(best_tile)
        self.current_shantens[player_id] = min_shanten
        return best_tile

    def check_rational_pong(self, discarded_tile, discarder_id):
        for offset in range(1, 4):
            target = (discarder_id + offset) % 4
            p_state = self.players_hands[target]
            counts = p_state.counts
            if counts[discarded_tile] >= 2 and len(p_state.hand) >= 4:
                current_shanten = self.current_shantens[target]
                counts[discarded_tile] -= 2
                melds_count = len(self.melds[target]) + 1

                best_after_shanten = 999
                for cand in range(34):
                    if counts[cand] > 0:
                        counts[cand] -= 1
                        s = self.evaluator.calculate_shanten_from_counts(counts, melds_count)
                        counts[cand] += 1
                        if s < best_after_shanten:
                            best_after_shanten = s
                counts[discarded_tile] += 2

                if best_after_shanten < current_shanten:
                    return target
        return None

    def check_rational_chow(self, discarded_tile, discarder_id):
        """
        上家理性吃牌判定（受 enable_chow 參數控制）：
        僅限下家 (discarder_id + 1) % 4 吃上家打出的數牌，且吃牌後必須使向聽數嚴格下降。
        """
        if not self.enable_chow or discarded_tile >= 27:
            return None, None
        target = (discarder_id + 1) % 4
        p_state = self.players_hands[target]
        counts = p_state.counts
        if len(p_state.hand) < 4:
            return None, None

        pos = discarded_tile % 9
        candidates = []
        if pos >= 2 and counts[discarded_tile - 2] > 0 and counts[discarded_tile - 1] > 0:
            candidates.append((discarded_tile - 2, discarded_tile - 1))
        if 1 <= pos <= 7 and counts[discarded_tile - 1] > 0 and counts[discarded_tile + 1] > 0:
            candidates.append((discarded_tile - 1, discarded_tile + 1))
        if pos <= 6 and counts[discarded_tile + 1] > 0 and counts[discarded_tile + 2] > 0:
            candidates.append((discarded_tile + 1, discarded_tile + 2))

        if not candidates:
            return None, None

        current_shanten = self.current_shantens[target]
        melds_count = len(self.melds[target]) + 1
        best_combo = None
        best_after_shanten = current_shanten

        for c1, c2 in candidates:
            counts[c1] -= 1
            counts[c2] -= 1
            after_s = 999
            for cand in range(34):
                if counts[cand] > 0:
                    counts[cand] -= 1
                    s = self.evaluator.calculate_shanten_from_counts(counts, melds_count)
                    counts[cand] += 1
                    if s < after_s:
                        after_s = s
            counts[c1] += 1
            counts[c2] += 1
            if after_s < best_after_shanten:
                best_after_shanten = after_s
                best_combo = (c1, c2)

        if best_combo is not None:
            return target, best_combo
        return None, None

    def run_simulation(self):
        current_turn = self.dealer_seat
        ply_count = 1
        is_dealer_first_turn = True
        skip_draw = False

        outcome_type = "DRAW"
        winner_id = None
        loser_id = None

        while len(self.wall) > 16:
            turn_num = min(16, (ply_count - 1) // 4 + 1)
            for p in range(4):
                if turn_num not in self.turn_shanten_history[p]:
                    self.turn_shanten_history[p][turn_num] = self.current_shantens[p]

            p_state = self.players_hands[current_turn]
            melds_len = len(self.melds[current_turn])

            if is_dealer_first_turn:
                is_dealer_first_turn = False
                if self.evaluator.is_win_from_counts(p_state.counts, melds_len):
                    self.first_tenpai_ply[current_turn] = 1
                    outcome_type = "TSUMO"
                    winner_id = current_turn
                    break
            elif skip_draw:
                skip_draw = False
            else:
                drawn_tile = self.wall.pop(0)
                while drawn_tile >= 34:
                    self.players_flowers[current_turn].append(drawn_tile)
                    if len(self.wall) <= 16:
                        break
                    drawn_tile = self.wall.pop(-1)

                if len(self.wall) <= 16:
                    break

                p_state.add(drawn_tile)
                if self.evaluator.is_win_from_counts(p_state.counts, melds_len):
                    if self.first_tenpai_ply[current_turn] is None:
                        self.first_tenpai_ply[current_turn] = ply_count
                    outcome_type = "TSUMO"
                    winner_id = current_turn
                    break

            discarded_tile = self.decide_discard(current_turn)
            self.discards[current_turn].append(discarded_tile)
            p_state.sort()

            if self.current_shantens[current_turn] == 0 and self.first_tenpai_ply[current_turn] is None:
                self.first_tenpai_ply[current_turn] = ply_count

            # 攔截優先序 1：榮和胡牌（逆時針截胡優先，透過 can_win_with 封裝方法驗證）
            ron_winner = None
            for offset in range(1, 4):
                verifier = (current_turn + offset) % 4
                if self.current_shantens[verifier] == 0:
                    v_state = self.players_hands[verifier]
                    if v_state.can_win_with(discarded_tile, self.evaluator, len(self.melds[verifier])):
                        ron_winner = verifier
                        if self.first_tenpai_ply[verifier] is None:
                            self.first_tenpai_ply[verifier] = ply_count
                        break

            if ron_winner is not None:
                outcome_type = "RON"
                winner_id = ron_winner
                loser_id = current_turn
                break

            # 攔截優先序 2：理性碰牌（可碰任一家）
            pong_player = self.check_rational_pong(discarded_tile, current_turn)
            if pong_player is not None:
                self.discards[current_turn].pop()
                p_pong = self.players_hands[pong_player]
                p_pong.remove(discarded_tile)
                p_pong.remove(discarded_tile)
                self.melds[pong_player].append(("碰", discarded_tile))
                current_turn = pong_player
                skip_draw = True
                ply_count += 1
                continue

            # 攔截優先序 3：上家理性吃牌（僅限下家吃上家，受 enable_chow 控制）
            if self.enable_chow:
                chow_player, chow_combo = self.check_rational_chow(discarded_tile, current_turn)
                if chow_player is not None:
                    self.discards[current_turn].pop()
                    p_chow = self.players_hands[chow_player]
                    p_chow.remove(chow_combo[0])
                    p_chow.remove(chow_combo[1])
                    self.melds[chow_player].append(("吃", discarded_tile))
                    current_turn = chow_player
                    skip_draw = True
                    ply_count += 1
                    continue

            current_turn = (current_turn + 1) % 4
            ply_count += 1

        # 終局狀態同步審計：確保 4 位玩家之 PlayerHandState 計數無任何偏移
        for p in range(4):
            self.players_hands[p].verify_consistency()

        # 結算全場純零和底數單位 (base_units) 與內生基本台數 (tai_units)
        base_units = [0.0, 0.0, 0.0, 0.0]
        tai_units = [0.0, 0.0, 0.0, 0.0]

        if outcome_type == "TSUMO" and winner_id is not None:
            is_menqing = (len(self.melds[winner_id]) == 0)
            hand_tai = 3 if is_menqing else 1
            for p in range(4):
                if p != winner_id:
                    dealer_tai = 1 if (winner_id == self.dealer_seat or p == self.dealer_seat) else 0
                    total_tai = (hand_tai + dealer_tai) if self.enable_tai else 0
                    base_units[p] -= 1.0
                    base_units[winner_id] += 1.0
                    tai_units[p] -= float(total_tai)
                    tai_units[winner_id] += float(total_tai)

        elif outcome_type == "RON" and winner_id is not None and loser_id is not None:
            is_menqing = (len(self.melds[winner_id]) == 0)
            hand_tai = 1 if is_menqing else 0
            dealer_tai = 1 if (winner_id == self.dealer_seat or loser_id == self.dealer_seat) else 0
            total_tai = (hand_tai + dealer_tai) if self.enable_tai else 0
            base_units[loser_id] -= 1.0
            base_units[winner_id] += 1.0
            tai_units[loser_id] -= float(total_tai)
            tai_units[winner_id] += float(total_tai)

        records = []
        turn_matrix_rows = []
        for p in range(4):
            tenpai_ply = self.first_tenpai_ply[p]
            is_tenpai = 1 if tenpai_ply is not None else 0
            tenpai_turn = ((tenpai_ply - 1) // 4 + 1) if is_tenpai else np.nan

            is_win = 1 if p == winner_id else 0
            is_deal_in = 1 if (outcome_type == "RON" and p == loser_id) else 0
            net_ev = base_units[p]
            net_tai = tai_units[p]

            records.append({
                "player_seat": p,
                "is_dealer": 1 if p == self.dealer_seat else 0,
                "init_shanten": self.init_shantens[p],
                "init_conn": self.init_conn_scores[p],
                "init_flowers": self.init_flowers[p],
                "is_tenpai": is_tenpai,
                "tenpai_turn": tenpai_turn,
                "is_win": is_win,
                "is_deal_in": is_deal_in,
                "net_ev": net_ev,
                "net_tai": net_tai,
                "ev_300_100": net_ev * 300.0 + net_tai * 100.0,
                "ev_100_20": net_ev * 100.0 + net_tai * 20.0,
                "ev_50_20": net_ev * 50.0 + net_tai * 20.0
            })

            for t_idx, sh_val in self.turn_shanten_history[p].items():
                if 0 <= sh_val <= 6 and 1 <= t_idx <= 16:
                    turn_matrix_rows.append((t_idx, sh_val, net_ev))

        return records, turn_matrix_rows


def run_batch_simulation(num_games=5000, csv_filename="mahjong_all_players.csv", seed=42, enable_chow=True, enable_tai=True):
    random.seed(seed)
    np.random.seed(seed)
    start_time = time.perf_counter()
    all_data = []
    all_turn_rows = []

    for game_id in range(1, num_games + 1):
        dealer = (game_id - 1) % 4
        table = CMLMahjongTable(dealer_seat=dealer, enable_chow=enable_chow, enable_tai=enable_tai)
        table.build_wall_and_deal()
        game_records, turn_rows = table.run_simulation()
        for r in game_records:
            r["game_id"] = game_id
            all_data.append(r)
        all_turn_rows.extend(turn_rows)

    elapsed = time.perf_counter() - start_time
    fieldnames = [
        "game_id", "player_seat", "is_dealer", "init_shanten",
        "init_conn", "init_flowers", "is_tenpai", "tenpai_turn",
        "is_win", "is_deal_in", "net_ev", "net_tai",
        "ev_300_100", "ev_100_20", "ev_50_20"
    ]
    with open(csv_filename, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)

    mode_str = f"enable_chow={enable_chow}, enable_tai={enable_tai}, Seed={seed}"
    print(f"[模擬完成] {num_games} 局 ({len(all_data)} 筆樣本, {mode_str}) 耗時：{elapsed:.2f} 秒 | 輸出：{csv_filename}")
    return all_turn_rows


def compute_ols_cluster_robust(df_sub, feature_cols, target_col, cluster_col="game_id"):
    X_raw = df_sub[feature_cols].values
    y = df_sub[target_col].values
    clusters = df_sub[cluster_col].values

    n, k = X_raw.shape
    X = np.column_stack([np.ones(n), X_raw])
    k_total = k + 1

    inv_XX = np.linalg.inv(X.T @ X)
    beta = inv_XX @ (X.T @ y)
    pred = X @ beta
    resid = y - pred

    sigma2 = np.sum(resid ** 2) / (n - k_total)
    se_naive = np.sqrt(np.diag(sigma2 * inv_XX))[1:]

    df_u = pd.DataFrame(X * resid[:, None])
    df_u["cluster"] = clusters
    cluster_sums = df_u.groupby("cluster").sum().values
    G = len(cluster_sums)
    meat = cluster_sums.T @ cluster_sums
    scale = (G / (G - 1)) * ((n - 1) / (n - k_total))
    cov_cluster = scale * (inv_XX @ meat @ inv_XX)
    se_cluster = np.sqrt(np.diag(cov_cluster))[1:]

    r2 = 1.0 - (np.sum(resid ** 2) / np.sum((y - np.mean(y)) ** 2))
    rmse_val = np.sqrt(np.mean(resid ** 2))

    res_df = pd.DataFrame({
        "Feature": feature_cols,
        "Coefficient": beta[1:],
        "Naive_SE": se_naive,
        "Cluster_SE": se_cluster,
        "CI_95_Lower": beta[1:] - 1.96 * se_cluster,
        "CI_95_Upper": beta[1:] + 1.96 * se_cluster
    })
    return beta[0], res_df, r2, rmse_val, pred, resid


def interpolate_breakeven(ev_df, col_name, min_samples=10):
    """
    線性內插計算損益兩平向聽數，並檢驗有效樣本區間之單調遞減特性與唯一穿零交點。
    """
    valid = ev_df[ev_df["Samples"] >= min_samples].sort_values("init_shanten")
    shantens = valid["init_shanten"].values
    vals = valid[col_name].values

    diffs = np.diff(vals)
    is_monotonic = bool(np.all(diffs <= 0))
    crossings = []

    for i in range(len(vals) - 1):
        if vals[i] >= 0 and vals[i + 1] < 0:
            be_point = shantens[i] + (0.0 - vals[i]) / (vals[i + 1] - vals[i])
            crossings.append(be_point)

    if not is_monotonic or len(crossings) != 1:
        print(f"[數值邊界提示] 欄位 {col_name} 單調遞減={is_monotonic}, 穿零交點數={len(crossings)}")

    return crossings[0] if crossings else np.nan


def run_regression_and_ev_pipeline(csv_filename="mahjong_all_players.csv", turn_rows=None):
    df = pd.read_csv(csv_filename)
    feature_cols = ["init_shanten", "init_conn", "init_flowers", "is_dealer"]

    # 1. 第一階段：邏輯迴歸預測聽牌成功率
    X_all = df[feature_cols]
    y_cls = df["is_tenpai"]
    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X_all, y_cls, test_size=0.25, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tr_c, y_tr_c)
    cls_probs = clf.predict_proba(X_te_c)[:, 1]
    cls_acc = accuracy_score(y_te_c, clf.predict(X_te_c))
    cls_auc = roc_auc_score(y_te_c, cls_probs)

    # 2. 第二階段：聽牌巡數條件 OLS 迴歸（僅針對未受右設限截斷之 is_tenpai==1 樣本）
    tenpai_df = df[df["is_tenpai"] == 1].dropna(subset=["tenpai_turn"]).copy()
    train_tp, test_tp = train_test_split(tenpai_df, test_size=0.25, random_state=42)

    intercept_t, coef_t_df, r2_train, rmse_train, _, _ = compute_ols_cluster_robust(train_tp, feature_cols, "tenpai_turn")
    X_test_design = np.column_stack([np.ones(len(test_tp)), test_tp[feature_cols].values])
    beta_vec = np.concatenate([[intercept_t], coef_t_df["Coefficient"].values])
    pred_test = X_test_design @ beta_vec
    y_test = test_tp["tenpai_turn"].values
    r2_test = 1.0 - (np.sum((y_test - pred_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
    rmse_test = np.sqrt(mse(y_test, pred_test))

    # 3. 第三階段：全樣本單局未加權底數期望值 (Base_EV) 群聚穩健 OLS 迴歸
    intercept_ev, coef_ev_df, r2_ev, _, _, _ = compute_ols_cluster_robust(df, feature_cols, "net_ev")

    # 4. 各起手向聽數之右設限截斷率、未加權底數 EV 與三種底台加權 EV 對照總表
    ev_summary = df.groupby("init_shanten").agg(
        Samples=("game_id", "count"),
        Tenpai_Rate=("is_tenpai", lambda x: x.mean() * 100),
        Censoring_Rate=("is_tenpai", lambda x: (1.0 - x.mean()) * 100),
        Avg_Tenpai_Turn=("tenpai_turn", "mean"),
        Win_Rate=("is_win", lambda x: x.mean() * 100),
        Deal_In_Rate=("is_deal_in", lambda x: x.mean() * 100),
        Base_EV=("net_ev", "mean"),
        EV_300_100=("ev_300_100", "mean"),
        EV_100_20=("ev_100_20", "mean"),
        EV_50_20=("ev_50_20", "mean")
    ).reset_index()
    ev_summary["Marginal_Turn_Diff"] = ev_summary["Avg_Tenpai_Turn"].diff()

    # 並列計算未加權底數基準與三種底台結構之損益兩平向聽數
    be_base = interpolate_breakeven(ev_summary, "Base_EV", min_samples=10)
    be_300_100 = interpolate_breakeven(ev_summary, "EV_300_100", min_samples=10)
    be_100_20 = interpolate_breakeven(ev_summary, "EV_100_20", min_samples=10)
    be_50_20 = interpolate_breakeven(ev_summary, "EV_50_20", min_samples=10)

    overall_censor_rate = (1.0 - len(tenpai_df) / len(df)) * 100

    print("\n================== 系統狀態、右設限比例與全員分類統計 ==================")
    print(f"總生成樣本數：{len(df)} 筆 | 成功聽牌：{len(tenpai_df)} 筆 (聽牌率：{100 - overall_censor_rate:.2f}% | 右設限截斷率：{overall_censor_rate:.2f}%)")
    print(f"全場零和檢驗 (Base_EV總和 / 300_100總和)：{df['net_ev'].sum():.1f} / {df['ev_300_100'].sum():.1f} | 邏輯迴歸 ROC-AUC: {cls_auc:.4f} (Accuracy: {cls_acc:.4f})")

    print("\n======== 模型一：聽牌巡數條件迴歸 (Target: tenpai_turn, 僅限未設限樣本 N={}, Cluster-Robust SE) ========".format(len(tenpai_df)))
    print(f"截距 (Baseline Tenpai Turn): {intercept_t:.4f}")
    print(coef_t_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"訓練集 R^2 : {r2_train:.4f} (RMSE: {rmse_train:.4f}) | 測試集 R^2 : {r2_test:.4f} (RMSE: {rmse_test:.4f})")

    print("\n======== 模型二：未加權底數期望值全員迴歸 (Target: net_ev, 全樣本 N={}, Cluster-Robust SE) ========".format(len(df)))
    print(f"截距 (Baseline Base EV): {intercept_ev:.4f} | 全樣本 R^2 : {r2_ev:.4f}")
    print(coef_ev_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print("\n======== 起手向聽數 vs. 未加權底數基準 (Base_EV) 與三種底台加權期望值對照總表 ========")
    print(ev_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("--------------------------------------------------------------------------------------")
    print("[損益兩平邊界與底台計分權重敏感度對照 (Break-Even Shanten Comparison, 樣本過濾門檻 N >= 10)]")
    print(f"  1. 未加權底數基準 (Base_EV)          : {be_base:.3f} 向聽")
    print(f"  2. 300/100 結構 (底台比 3.0 : 1, 標準型) : {be_300_100:.3f} 向聽 (差異: {be_300_100 - be_base:+.3f})")
    print(f"  3. 100/20  結構 (底台比 5.0 : 1, 快攻型) : {be_100_20:.3f} 向聽 (差異: {be_100_20 - be_base:+.3f})")
    print(f"  4. 50/20   結構 (底台比 2.5 : 1, 門清型) : {be_50_20:.3f} 向聽 (差異: {be_50_20 - be_base:+.3f})")

    # 5. 建構 16 巡 × 7 向聽 二維狀態期望值矩陣（樣本過濾邊界 N >= 20）
    tdf = pd.DataFrame(turn_rows, columns=["Turn", "Shanten", "Net_EV"])
    pivot_mean = tdf.pivot_table(index="Shanten", columns="Turn", values="Net_EV", aggfunc="mean")
    pivot_count = tdf.pivot_table(index="Shanten", columns="Turn", values="Net_EV", aggfunc="count")
    pivot_masked = pivot_mean.where(pivot_count >= 20, np.nan)

    print("\n======== 16 巡 × 7 向聽 二維狀態期望值矩陣 (Base_EV, 樣本過濾門檻：單格 N >= 20，不足標為 NaN) ========")
    print(pivot_masked.to_string(float_format=lambda x: f"{x:+.2f}"))
    print("=======================================================================================================\n")

    # 繪製 2×2 四合一綜合診斷與二維狀態期望值熱力圖
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    axes[0, 0].scatter(pred_test, y_test - pred_test, alpha=0.25, color="crimson", s=18)
    axes[0, 0].axhline(y=0, color="black", linestyle="--")
    axes[0, 0].set_title(f"1. Residual Plot (Uncensored Testing Set, Censoring={overall_censor_rate:.1f}%)")
    axes[0, 0].set_xlabel("Predicted Tenpai Turn")
    axes[0, 0].set_ylabel("Residual (Observed - Predicted)")

    tenpai_df.boxplot(column="tenpai_turn", by="init_shanten", ax=axes[0, 1], grid=False)
    axes[0, 1].set_title("2. Tenpai Turn Distribution by Initial Shanten (Right-Censored at Turn 16)")
    axes[0, 1].set_xlabel("Initial Shanten (t=0)")
    axes[0, 1].set_ylabel("Observed Tenpai Turn")

    valid_ev = ev_summary[ev_summary["Samples"] >= 10]
    axes[1, 0].plot(valid_ev["init_shanten"], valid_ev["Base_EV"], marker="o", linewidth=2.5, color="navy", label="Base Net EV (Unweighted)")
    axes[1, 0].axhline(y=0, color="darkgreen", linestyle="--", linewidth=1.5, label="Break-Even Line (EV = 0)")
    axes[1, 0].axvline(x=be_base, color="darkorange", linestyle=":", linewidth=1.8, label=f"Base Threshold ({be_base:.2f} Shanten)")
    axes[1, 0].plot(be_base, 0, marker="*", markersize=13, color="red")
    axes[1, 0].fill_between(valid_ev["init_shanten"], valid_ev["Base_EV"], 0,
                            where=(valid_ev["Base_EV"] >= 0), color="royalblue", alpha=0.2, label="Positive EV (Attack)")
    axes[1, 0].fill_between(valid_ev["init_shanten"], valid_ev["Base_EV"], 0,
                            where=(valid_ev["Base_EV"] < 0), color="salmon", alpha=0.2, label="Negative EV (Defend/Fold)")
    axes[1, 0].set_title("3. Unweighted Base EV Curve & Break-Even Boundary (Filter: N >= 10)")
    axes[1, 0].set_xlabel("Initial Shanten (t=0)")
    axes[1, 0].set_ylabel("Expected Payoff per Game (Base Units)")
    axes[1, 0].legend(loc="upper right", fontsize=8.5)

    mat_vals = pivot_masked.reindex(index=range(0, 7), columns=range(1, 17)).values
    cmap = plt.cm.RdBu.copy()
    cmap.set_bad(color="whitesmoke")
    im = axes[1, 1].imshow(mat_vals, cmap=cmap, vmin=-0.5, vmax=0.8, aspect="auto")
    axes[1, 1].set_title("4. 2D Dynamic Base EV Heatmap (Turns 1-16, Cell Filter: N >= 20)")
    axes[1, 1].set_xlabel("Game Progression Turn (1 to 16)")
    axes[1, 1].set_ylabel("Current Hand Shanten (0 = Tenpai to 6)")
    axes[1, 1].set_xticks(range(16))
    axes[1, 1].set_xticklabels(range(1, 17))
    axes[1, 1].set_yticks(range(7))
    axes[1, 1].set_yticklabels([f"{s} Shanten" if s > 0 else "0 (Tenpai)" for s in range(7)])

    for i in range(7):
        for j in range(16):
            val = mat_vals[i, j]
            if not np.isnan(val):
                txt_color = "white" if abs(val) > 0.35 else "black"
                axes[1, 1].text(j, i, f"{val:+.2f}", ha="center", va="center", color=txt_color, fontsize=7.5)

    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04, label="Dynamic Base EV (Units)")

    plt.suptitle("")
    plt.tight_layout()
    plt.savefig("mahjong_regression_analysis.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    run_expert_sanity_checks()
    turn_matrix_data = run_batch_simulation(
        num_games=5000,
        csv_filename="mahjong_all_players.csv",
        seed=42,
        enable_chow=True,
        enable_tai=True
    )
    run_regression_and_ev_pipeline(csv_filename="mahjong_all_players.csv", turn_rows=turn_matrix_data)