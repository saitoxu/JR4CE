import torch
import torch.nn.functional as F


class DPP:
    def __init__(self, model, epsilon=1e-10, temperature=1.0):
        self.model = model
        self.epsilon = epsilon
        self.temperature = temperature

    def rerank(self, pred: torch.Tensor, candidates, k):
        # torch.no_grad()でメモリ使用量を削減
        with torch.no_grad():
            item_features = self.model()[1]

            # 候補アイテムの特徴ベクトルとスコアを抽出
            candidate_features = item_features[candidates]
            relevance_scores = pred[candidates]

            # L行列の構築
            L = self.construct_L_matrix(relevance_scores, candidate_features)

            # DPPによるアイテム選択
            selected_indices = self.dpp(L, k)

            # 選択されたインデックスを元のcandidatesのインデックスに変換
            final_recommendations = [candidates[idx] for idx in selected_indices]

            # 残りのアイテムを効率的に追加
            if len(final_recommendations) < len(pred):
                final_recommendation_set = set(final_recommendations)
                # 一度のソートで済ませる
                pred_sorted = pred.argsort(descending=True).tolist()
                remaining = [item for item in pred_sorted if item not in final_recommendation_set]
                final_recommendations.extend(remaining)

        return final_recommendations

    def construct_L_matrix(self, relevance_scores, item_features):
        # 特徴ベクトルの正規化（インプレース演算でメモリ効率化）
        item_features_norm = F.normalize(item_features, p=2, dim=1)

        # アイテム間の類似度行列 C の構築 (C_ij = x_i^T x_j)
        C = torch.mm(item_features_norm, item_features_norm.t())

        # relevance scoresをsoftmaxで正規化（温度パラメータ適用）
        _relevance_scores = F.softmax(relevance_scores / self.temperature, dim=0)

        # L = Diag{r}⋅C⋅Diag{r} の計算を効率化
        # 対角行列の掛け算は要素ごとの掛け算で代替可能
        # L[i,j] = r[i] * C[i,j] * r[j]
        L = C * _relevance_scores.unsqueeze(0) * _relevance_scores.unsqueeze(1)

        return L

    def dpp(self, L, k):
        device = L.device
        item_size = L.shape[0]

        # 早期終了条件
        if k >= item_size:
            return list(range(item_size))

        cis = torch.zeros((k, item_size), device=device)
        di2s = torch.diagonal(L).clone()
        selected_items = []
        selected_mask = torch.zeros(item_size, dtype=torch.bool, device=device)

        # 最初のアイテムを選択
        selected_item = torch.argmax(di2s).item()
        selected_items.append(selected_item)
        selected_mask[selected_item] = True

        if k == 1:
            return selected_items

        # 2番目以降のアイテムの選択（ベクトル化された演算で高速化）
        for i in range(1, k):
            s = len(selected_items)
            ci_optimal = cis[: s - 1, selected_item]
            di_optimal = di2s[selected_item].sqrt()
            elements = L[selected_item, :]

            if di_optimal < self.epsilon:
                # 数値的に不安定な場合は早期終了
                break

            if s == 1:
                eis = elements / di_optimal
            else:
                # ベクトル化された演算で高速化
                eis = (elements - torch.mv(cis[: s - 1, :].t(), ci_optimal)) / di_optimal

            cis[s - 1, :] = eis
            # インプレース演算でメモリ効率化
            di2s.sub_(eis.square())

            # 選択済みアイテムをマスク
            di2s[selected_mask] = float("-inf")

            # 次のアイテムを選択
            max_val = di2s.max()
            if max_val < self.epsilon:
                break

            selected_item = torch.argmax(di2s).item()
            selected_items.append(selected_item)
            selected_mask[selected_item] = True

        return selected_items
