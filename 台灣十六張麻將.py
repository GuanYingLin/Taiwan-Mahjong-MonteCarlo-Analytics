"""
台灣十六張麻將：蒙地卡羅高併發模擬、進聽動力學與單局期望值 (EV) 決策分析系統
Taiwan Mahjong Monte Carlo Simulation, Tenpai Dynamics & Expected Value (EV) Framework

研究假設與模型邊界（Model Axioms & Scope Conditions）：
1. 同質性理性代理人假設：排除人類牌齡、記憶失誤與情緒等個體差異，四家皆以最小化向聽數與最大化牌效為決策準則。
2. 純化進攻基準線：將吃牌與棄胡防守設為控制變數（僅開放向聽推進之理性碰牌），測定物理極限下之進聽速度與基礎期望值。
3. 零和博弈損益結算：自摸 +3（三家各 -1）、榮和 +1（放銃者 -1，旁觀者 0）、流局 0，全場期望值總和嚴格為 0。
4. 統計偏誤校正：採用牌局群聚穩健標準誤（Cluster-Robust SE）校正同桌零和相依性，並揭露右設限（Right-Censoring）邊際遞減效應。
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


class TaiwanMahjongEvaluator:
    @staticmethod
    def to_counts(hand):
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
    def is_win(cls, hand, melds=[]):
        counts = cls.to_counts(hand)
        melds_count = len(melds)
        return cls.is_standard_win(counts, melds_count) or cls.is_eight_pairs_win(counts, melds_count)

    @classmethod
    def calculate_shanten(cls, hand, melds=[]):
        counts = cls.to_counts(hand)
        melds_count = len(melds)
        if cls.is_standard_win(counts, melds_count) or cls.is_eight_pairs_win(counts, melds_count):
            return -1

        target_melds = 5 - melds_count
        h_melds = 0
        h_pairs = 0
        for i in range(27, 34):
            if counts[i] >= 3:
                h_melds += 1
            elif counts[i] == 2:
                h_pairs += 1

        wan_res = _evaluate_single_suit(tuple(counts[0:9]))
        tong_res = _evaluate_single_suit(tuple(counts[9:18]))
        tiao_res = _evaluate_single_suit(tuple(counts[18:27]))

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

        if melds_count == 0:
            pair_units = min(8, sum(c // 2 for c in counts))
            has_triplet = 1 if any(c == 3 for c in counts) else 0
            eight_pairs_shanten = 8 - pair_units - has_triplet
            return min(min_shanten, eight_pairs_shanten)

        return min_shanten


def run_expert_sanity_checks():
    ev = TaiwanMahjongEvaluator
    hand_tenpai = [0, 1, 2, 12, 13, 14, 24, 25, 26, 27, 27, 27, 26, 26, 4, 5]
    assert ev.calculate_shanten(hand_tenpai) == 0, "驗收失敗：標準聽牌型未判定為 0 向聽"

    hand_win = hand_tenpai + [3]
    assert ev.is_win(hand_win) is True and ev.calculate_shanten(hand_win) == -1, "驗收失敗：標準胡牌型未正確判定"

    hand_ligu = [0, 0, 2, 2, 9, 9, 13, 13, 18, 18, 27, 27, 31, 31, 32, 32, 32]
    assert ev.is_win(hand_ligu) is True and ev.calculate_shanten(hand_ligu) == -1, "驗收失敗：嚦咕嚦咕未正確判定"

    print("[專家牌型驗收通過] 標準聽牌型(0向聽)、5面1眼胡牌型(-1)、嚦咕嚦咕特殊胡牌型(-1) 全數吻合人類驗算標準。")


class CMLMahjongTable:
    def __init__(self, dealer_seat=0):
        self.dealer_seat = dealer_seat
        self.wall = []
        self.players_hands = [[], [], [], []]
        self.players_flowers = [[], [], [], []]
        self.discards = [[], [], [], []]
        self.melds = [[], [], [], []]
        self.init_shantens = [0, 0, 0, 0]
        self.init_conn_scores = [0, 0, 0, 0]
        self.init_flowers = [0, 0, 0, 0]
        self.first_tenpai_ply = [None, None, None, None]
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
            self.players_hands[i] = [self.wall.pop(0) for _ in range(16)]

        self.players_hands[self.dealer_seat].append(self.wall.pop(0))
        self.execute_initial_flower_replacement()

        for i in range(4):
            self.players_hands[i].sort()
            if i == self.dealer_seat:
                best_s = min(
                    self.evaluator.calculate_shanten(
                        [t for idx, t in enumerate(self.players_hands[i]) if idx != self.players_hands[i].index(cand)],
                        self.melds[i]
                    )
                    for cand in set(self.players_hands[i])
                )
                self.init_shantens[i] = max(0, best_s)
            else:
                self.init_shantens[i] = self.evaluator.calculate_shanten(self.players_hands[i], self.melds[i])

            if self.init_shantens[i] <= 0:
                self.first_tenpai_ply[i] = 1

            counts = self.evaluator.to_counts(self.players_hands[i])
            self.init_conn_scores[i] = sum(self._tile_connectivity_score(t, counts) for t in self.players_hands[i])
            self.init_flowers[i] = len(self.players_flowers[i])

    def execute_initial_flower_replacement(self):
        for i in range(4):
            while True:
                flowers = [t for t in self.players_hands[i] if t >= 34]
                if not flowers:
                    break
                f = flowers[0]
                self.players_hands[i].remove(f)
                self.players_flowers[i].append(f)
                self.players_hands[i].append(self.wall.pop(-1))

    def decide_discard(self, player_id):
        hand = self.players_hands[player_id]
        melds = self.melds[player_id]
        if len(hand) == 1:
            return hand.pop(0)

        counts = self.evaluator.to_counts(hand)
        unique_tiles = sorted(list(set(hand)))

        best_tile = unique_tiles[0]
        min_shanten = 999
        min_connectivity = 99999

        for candidate in unique_tiles:
            temp_hand = list(hand)
            temp_hand.remove(candidate)
            s = self.evaluator.calculate_shanten(temp_hand, melds)
            conn_score = self._tile_connectivity_score(candidate, counts)

            if (s < min_shanten) or (s == min_shanten and conn_score < min_connectivity):
                min_shanten = s
                min_connectivity = conn_score
                best_tile = candidate

        hand.remove(best_tile)
        return best_tile

    def check_rational_pong(self, discarded_tile, discarder_id):
        for offset in range(1, 4):
            target = (discarder_id + offset) % 4
            hand = self.players_hands[target]
            if hand.count(discarded_tile) >= 2 and len(hand) >= 4:
                current_shanten = self.evaluator.calculate_shanten(hand, self.melds[target])
                sim_hand = list(hand)
                sim_hand.remove(discarded_tile)
                sim_hand.remove(discarded_tile)
                sim_melds = self.melds[target] + [("碰", discarded_tile)]

                best_after_shanten = min(
                    self.evaluator.calculate_shanten(
                        [t for idx, t in enumerate(sim_hand) if idx != sim_hand.index(cand)],
                        sim_melds
                    )
                    for cand in set(sim_hand)
                )
                if best_after_shanten < current_shanten:
                    return target
        return None

    def run_simulation(self):
        current_turn = self.dealer_seat
        ply_count = 1
        is_dealer_first_turn = True
        skip_draw = False

        outcome_type = "DRAW"
        winner_id = None
        loser_id = None

        while len(self.wall) > 16:
            if is_dealer_first_turn:
                is_dealer_first_turn = False
                if self.evaluator.is_win(self.players_hands[current_turn], self.melds[current_turn]):
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

                self.players_hands[current_turn].append(drawn_tile)
                if self.evaluator.is_win(self.players_hands[current_turn], self.melds[current_turn]):
                    if self.first_tenpai_ply[current_turn] is None:
                        self.first_tenpai_ply[current_turn] = ply_count
                    outcome_type = "TSUMO"
                    winner_id = current_turn
                    break

            discarded_tile = self.decide_discard(current_turn)
            self.discards[current_turn].append(discarded_tile)
            self.players_hands[current_turn].sort()

            shanten = self.evaluator.calculate_shanten(self.players_hands[current_turn], self.melds[current_turn])
            if shanten == 0 and self.first_tenpai_ply[current_turn] is None:
                self.first_tenpai_ply[current_turn] = ply_count

            ron_winner = None
            for offset in range(1, 4):
                verifier = (current_turn + offset) % 4
                test_hand = self.players_hands[verifier] + [discarded_tile]
                if self.evaluator.is_win(test_hand, self.melds[verifier]):
                    ron_winner = verifier
                    if self.first_tenpai_ply[verifier] is None:
                        self.first_tenpai_ply[verifier] = ply_count
                    break

            if ron_winner is not None:
                outcome_type = "RON"
                winner_id = ron_winner
                loser_id = current_turn
                break

            pong_player = self.check_rational_pong(discarded_tile, current_turn)
            if pong_player is not None:
                self.discards[current_turn].pop()
                self.players_hands[pong_player].remove(discarded_tile)
                self.players_hands[pong_player].remove(discarded_tile)
                self.melds[pong_player].append(("碰", discarded_tile))
                current_turn = pong_player
                skip_draw = True
            else:
                current_turn = (current_turn + 1) % 4

            ply_count += 1

        records = []
        for p in range(4):
            tenpai_ply = self.first_tenpai_ply[p]
            is_tenpai = 1 if tenpai_ply is not None else 0
            tenpai_turn = ((tenpai_ply - 1) // 4 + 1) if is_tenpai else np.nan

            is_win = 1 if p == winner_id else 0
            is_deal_in = 1 if (outcome_type == "RON" and p == loser_id) else 0

            if outcome_type == "TSUMO":
                net_ev = 3.0 if p == winner_id else -1.0
            elif outcome_type == "RON":
                if p == winner_id:
                    net_ev = 1.0
                elif p == loser_id:
                    net_ev = -1.0
                else:
                    net_ev = 0.0
            else:
                net_ev = 0.0

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
                "net_ev": net_ev
            })
        return records


def run_batch_simulation(num_games=5000, csv_filename="mahjong_all_players.csv", seed=42):
    random.seed(seed)
    np.random.seed(seed)
    start_time = time.perf_counter()
    all_data = []

    for game_id in range(1, num_games + 1):
        dealer = (game_id - 1) % 4
        table = CMLMahjongTable(dealer_seat=dealer)
        table.build_wall_and_deal()
        game_records = table.run_simulation()
        for r in game_records:
            r["game_id"] = game_id
            all_data.append(r)

    elapsed = time.perf_counter() - start_time
    fieldnames = [
        "game_id", "player_seat", "is_dealer", "init_shanten",
        "init_conn", "init_flowers", "is_tenpai", "tenpai_turn",
        "is_win", "is_deal_in", "net_ev"
    ]
    with open(csv_filename, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)

    print(f"[模擬完成] {num_games} 局 ({len(all_data)} 筆玩家樣本, Seed={seed}) 耗時：{elapsed:.2f} 秒 | 輸出：{csv_filename}")


def compute_ols_cluster_robust(df_sub, feature_cols, target_col, cluster_col="game_id"):
    """
    計算普通最小平方法 (OLS) 係數、傳統獨立標準誤 (Naive SE) 與牌局群聚穩健標準誤 (Cluster-Robust SE)。
    """
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

    # 傳統 i.i.d. 標準誤
    sigma2 = np.sum(resid ** 2) / (n - k_total)
    se_naive = np.sqrt(np.diag(sigma2 * inv_XX))[1:]

    # 依 game_id 分群之 Liang-Zeger 群聚穩健標準誤 (Cluster-Robust SE)
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


def run_regression_and_ev_pipeline(csv_filename="mahjong_all_players.csv"):
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

    # 2. 第二階段：聽牌巡數條件 OLS 迴歸（含群聚穩健標準誤與訓練/測試集驗證）
    tenpai_df = df[df["is_tenpai"] == 1].dropna(subset=["tenpai_turn"]).copy()
    train_tp, test_tp = train_test_split(tenpai_df, test_size=0.25, random_state=42)

    intercept_t, coef_t_df, r2_train, rmse_train, _, _ = compute_ols_cluster_robust(train_tp, feature_cols, "tenpai_turn")
    X_test_design = np.column_stack([np.ones(len(test_tp)), test_tp[feature_cols].values])
    beta_vec = np.concatenate([[intercept_t], coef_t_df["Coefficient"].values])
    pred_test = X_test_design @ beta_vec
    y_test = test_tp["tenpai_turn"].values
    r2_test = 1.0 - (np.sum((y_test - pred_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
    rmse_test = np.sqrt(mse(y_test, pred_test))

    # 3. 第三階段：全樣本單局期望值 (Net EV) 群聚穩健 OLS 迴歸
    intercept_ev, coef_ev_df, r2_ev, _, _, _ = compute_ols_cluster_robust(df, feature_cols, "net_ev")

    # 4. 各起手向聽數之期望值 (EV)、胡銃率與右設限邊際巡數差彙整表
    ev_summary = df.groupby("init_shanten").agg(
        Samples=("game_id", "count"),
        Tenpai_Rate=("is_tenpai", lambda x: x.mean() * 100),
        Avg_Tenpai_Turn=("tenpai_turn", "mean"),
        Win_Rate=("is_win", lambda x: x.mean() * 100),
        Deal_In_Rate=("is_deal_in", lambda x: x.mean() * 100),
        Expected_Value_EV=("net_ev", "mean")
    ).reset_index()
    ev_summary["Marginal_Turn_Diff"] = ev_summary["Avg_Tenpai_Turn"].diff()

    # 線性內插計算精確之損益兩平向聽數 (Break-Even Shanten, EV = 0)
    ev3 = ev_summary.loc[ev_summary["init_shanten"] == 3, "Expected_Value_EV"].values[0]
    ev4 = ev_summary.loc[ev_summary["init_shanten"] == 4, "Expected_Value_EV"].values[0]
    breakeven_shanten = 3.0 + (0.0 - ev3) / (ev4 - ev3)

    print("\n================== 系統狀態與全員分類統計 ==================")
    print(f"總生成樣本數：{len(df)} 筆 | 成功聽牌樣本數：{len(tenpai_df)} 筆 (整體聽牌率：{len(tenpai_df)/len(df)*100:.2f}%)")
    print(f"全場零和檢驗 (總損益和)：{df['net_ev'].sum():.1f} | 邏輯迴歸 ROC-AUC: {cls_auc:.4f} (Accuracy: {cls_acc:.4f})")

    print("\n======== 模型一：聽牌巡數條件迴歸 (Target: tenpai_turn, Cluster-Robust SE) ========")
    print(f"截距 (Baseline Tenpai Turn): {intercept_t:.4f}")
    print(coef_t_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"訓練集 R^2 : {r2_train:.4f} (RMSE: {rmse_train:.4f}) | 測試集 R^2 : {r2_test:.4f} (RMSE: {rmse_test:.4f})")

    print("\n======== 模型二：單局淨期望值全員迴歸 (Target: net_ev, Cluster-Robust SE) ========")
    print(f"截距 (Baseline Net EV): {intercept_ev:.4f} | 全樣本 R^2 : {r2_ev:.4f}")
    print(coef_ev_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print("\n================ 起手向聽數 vs. 牌局期望值 (EV) 與邊際遞減總表 ================")
    print(ev_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"---> 線性內插之全速進攻損益兩平臨界點 (Break-Even Shanten, EV = 0): {breakeven_shanten:.3f} 向聽")
    print("===============================================================================\n")

    # 繪製三合一診斷與決策圖表
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(pred_test, y_test - pred_test, alpha=0.25, color="crimson", s=18)
    axes[0].axhline(y=0, color="black", linestyle="--")
    axes[0].set_title("1. Residual Plot (Testing Set)")
    axes[0].set_xlabel("Predicted Tenpai Turn")
    axes[0].set_ylabel("Residual (Observed - Predicted)")

    tenpai_df.boxplot(column="tenpai_turn", by="init_shanten", ax=axes[1], grid=False)
    axes[1].set_title("2. Tenpai Turn Distribution by Initial Shanten")
    axes[1].set_xlabel("Initial Shanten (t=0)")
    axes[1].set_ylabel("Observed Tenpai Turn")

    valid_ev = ev_summary[ev_summary["Samples"] >= 10]
    axes[2].plot(valid_ev["init_shanten"], valid_ev["Expected_Value_EV"], marker="o", linewidth=2.5, color="navy", label="Net EV (Zero-Sum Units)")
    axes[2].axhline(y=0, color="darkgreen", linestyle="--", linewidth=1.5, label="Break-Even Line (EV = 0)")
    axes[2].axvline(x=breakeven_shanten, color="darkorange", linestyle=":", linewidth=1.8, label=f"Threshold ({breakeven_shanten:.2f} Shanten)")
    axes[2].plot(breakeven_shanten, 0, marker="*", markersize=12, color="red")
    axes[2].fill_between(valid_ev["init_shanten"], valid_ev["Expected_Value_EV"], 0,
                         where=(valid_ev["Expected_Value_EV"] >= 0), color="royalblue", alpha=0.2, label="Positive EV (Attack)")
    axes[2].fill_between(valid_ev["init_shanten"], valid_ev["Expected_Value_EV"], 0,
                         where=(valid_ev["Expected_Value_EV"] < 0), color="salmon", alpha=0.2, label="Negative EV (Defend/Fold)")
    axes[2].set_title("3. Expected Value (EV) Curve & Break-Even Threshold")
    axes[2].set_xlabel("Initial Shanten (t=0)")
    axes[2].set_ylabel("Expected Payoff per Game (Units)")
    axes[2].legend(loc="upper right", fontsize=8.5)

    plt.suptitle("")
    plt.tight_layout()
    plt.savefig("mahjong_regression_analysis.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    run_expert_sanity_checks()
    run_batch_simulation(num_games=5000, csv_filename="mahjong_all_players.csv", seed=42)
    run_regression_and_ev_pipeline(csv_filename="mahjong_all_players.csv")