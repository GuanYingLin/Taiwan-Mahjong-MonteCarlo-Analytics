"""
台灣十六張麻將：基於蒙地卡羅模擬之起手牌效與聽牌巡數分析系統
Taiwan Mahjong Monte Carlo Simulation & Tenpai Turn Evaluation Framework

升級與修正項目：
1. 消除選擇偏誤（Selection Bias）：追蹤每局四位玩家全員數據，結合邏輯迴歸（全員聽牌率分類）與條件線性迴歸（聽牌巡數預測）。
2. 消除時間序列資料洩漏（Data Leakage）：自變數嚴格限縮為開局補花後（t=0）之純外生靜態特徵。
3. 狀態機邊界修正：修復起手天聽（init_shanten <= 0）與天胡時未記錄聽牌步數之漏洞。
4. 嚴謹推論統計：透過共變異數矩陣解析計算 OLS 迴歸係數之標準誤（SE）與 95% 信賴區間（95% CI）。
"""

import random
import time
import csv
from functools import lru_cache
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, accuracy_score, roc_auc_score
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

        # 1. 刻子
        if c >= 3:
            counts[idx] -= 3
            dfs(idx, m + 1, t, p, pair_used_at_idx)
            counts[idx] += 3

        # 2. 順子
        if idx <= 6 and counts[idx + 1] > 0 and counts[idx + 2] > 0:
            counts[idx] -= 1
            counts[idx + 1] -= 1
            counts[idx + 2] -= 1
            dfs(idx, m + 1, t, p, pair_used_at_idx)
            counts[idx] += 1
            counts[idx + 1] += 1
            counts[idx + 2] += 1

        # 3. 雀頭
        if c >= 2 and p == 0 and not pair_used_at_idx:
            counts[idx] -= 2
            dfs(idx, m, t, 1, True)
            counts[idx] += 2

        # 4. 兩面或邊張搭子
        if idx <= 7 and counts[idx + 1] > 0:
            counts[idx] -= 1
            counts[idx + 1] -= 1
            dfs(idx, m, t + 1, p, pair_used_at_idx)
            counts[idx] += 1
            counts[idx + 1] += 1

        # 5. 嵌張搭子
        if idx <= 6 and counts[idx + 2] > 0:
            counts[idx] -= 1
            counts[idx + 2] -= 1
            dfs(idx, m, t + 1, p, pair_used_at_idx)
            counts[idx] += 1
            counts[idx + 2] += 1

        # 6. 雙碰搭子
        if c >= 2 and not pair_used_at_idx:
            counts[idx] -= 2
            dfs(idx, m, t + 1, p, True)
            counts[idx] += 2

        # 7. 孤張跳過
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
                # 莊家起手 17 張，評估打出一張後之最佳起手向聽數
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

            # 若起手即天聽或天胡，預先登記第 1 步聽牌
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

        while len(self.wall) > 16:
            if is_dealer_first_turn:
                is_dealer_first_turn = False
                if self.evaluator.is_win(self.players_hands[current_turn], self.melds[current_turn]):
                    self.first_tenpai_ply[current_turn] = 1
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

            records.append({
                "player_seat": p,
                "is_dealer": 1 if p == self.dealer_seat else 0,
                "init_shanten": self.init_shantens[p],
                "init_conn": self.init_conn_scores[p],
                "init_flowers": self.init_flowers[p],
                "is_tenpai": is_tenpai,
                "tenpai_turn": tenpai_turn
            })
        return records


def run_batch_simulation(num_games=5000, csv_filename="mahjong_all_players.csv"):
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
        "init_conn", "init_flowers", "is_tenpai", "tenpai_turn"
    ]
    with open(csv_filename, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)

    print(f"[模擬完成] {num_games} 局 ({len(all_data)} 筆玩家樣本) 耗時：{elapsed:.2f} 秒 | 輸出檔案：{csv_filename}")


def run_regression_pipeline(csv_filename="mahjong_all_players.csv"):
    df = pd.read_csv(csv_filename)
    feature_cols = ["init_shanten", "init_conn", "init_flowers", "is_dealer"]

    # 1. 第一階段（全樣本無偏分類模型）：使用邏輯迴歸預測達成聽牌之機率
    X_all = df[feature_cols]
    y_cls = df["is_tenpai"]
    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X_all, y_cls, test_size=0.25, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tr_c, y_tr_c)
    cls_probs = clf.predict_proba(X_te_c)[:, 1]
    cls_acc = accuracy_score(y_te_c, clf.predict(X_te_c))
    cls_auc = roc_auc_score(y_te_c, cls_probs)

    # 2. 第二階段（條件線性迴歸模型）：針對成功聽牌樣本預測聽牌巡數與 95% 信賴區間
    tenpai_df = df[df["is_tenpai"] == 1].dropna(subset=["tenpai_turn"]).copy()
    X = tenpai_df[feature_cols]
    y = tenpai_df["tenpai_turn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    # 解析計算 OLS 迴歸係數之標準誤 (Standard Error) 與 95% 信賴區間
    n, k = X_train.shape
    X_design = np.column_stack([np.ones(n), X_train.values])
    residuals = y_train.values - pred_train
    sigma2 = np.sum(residuals ** 2) / (n - k - 1)
    cov_matrix = sigma2 * np.linalg.inv(X_design.T @ X_design)
    se = np.sqrt(np.diag(cov_matrix))[1:]  # 排除截距項之標準誤

    coef_df = pd.DataFrame({
        "Feature": feature_cols,
        "Coefficient": model.coef_,
        "Std_Error": se,
        "CI_95_Lower": model.coef_ - 1.96 * se,
        "CI_95_Upper": model.coef_ + 1.96 * se
    })

    print("\n================== 系統狀態與全員分類統計 ==================")
    print(f"總生成樣本數：{len(df)} 筆 | 成功聽牌樣本數：{len(tenpai_df)} 筆 (整體聽牌率：{len(tenpai_df)/len(df)*100:.2f}%)")
    print(f"邏輯迴歸 (聽牌率預測) 測試集準確率 (Accuracy): {cls_acc:.4f} | ROC-AUC: {cls_auc:.4f}")

    print("\n================ 多元線性迴歸係數 (OLS & 95% CI) ================")
    print(f"截距 (Baseline Tenpai Turn): {model.intercept_:.4f}")
    print(coef_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print("\n---------------- 迴歸模型評估指標 ----------------")
    print(f"訓練集 MAE : {mae(y_train, pred_train):.4f} | 測試集 MAE : {mae(y_test, pred_test):.4f}")
    print(f"訓練集 RMSE: {mse(y_train, pred_train)**0.5:.4f} | 測試集 RMSE: {mse(y_test, pred_test)**0.5:.4f}")
    print(f"訓練集 R^2 : {model.score(X_train, y_train):.4f} | 測試集 R^2 : {model.score(X_test, y_test):.4f}")
    print("=================================================================\n")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(pred_test, y_test - pred_test, alpha=0.3, color="crimson", s=20)
    axes[0].axhline(y=0, color="black", linestyle="--")
    axes[0].set_title("Residual Plot (Testing Set)")
    axes[0].set_xlabel("Predicted Turn")
    axes[0].set_ylabel("Residual (Observed - Predicted)")

    tenpai_df.boxplot(column="tenpai_turn", by="init_shanten", ax=axes[1], grid=False)
    axes[1].set_title("Tenpai Turn Distribution by Initial Shanten")
    axes[1].set_xlabel("Initial Shanten (t=0)")
    axes[1].set_ylabel("Observed Tenpai Turn")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig("mahjong_regression_analysis.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    run_batch_simulation(num_games=5000, csv_filename="mahjong_all_players.csv")
    run_regression_pipeline(csv_filename="mahjong_all_players.csv")