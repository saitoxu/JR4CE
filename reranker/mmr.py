import numpy as np
import torch


class MMR:
    def __init__(self, lambda_factor, model):
        self.lambda_factor = lambda_factor
        self.model = model

    def rerank(self, pred, candidates, k):
        # 類似度行列の計算
        _, items = self.model()
        sim_matrix = torch.mm(items, items.t())

        # 関連性スコアの取得
        relevance = pred[candidates].cpu().numpy()
        similarity = sim_matrix[candidates][:, candidates].cpu().numpy()

        final_recommendations = []
        selected_indices = []

        for i in range(len(candidates)):
            if i == 0 or i >= k:
                next_item_idx = np.argmax(relevance)
                final_recommendations.append(candidates[next_item_idx])
                selected_indices.append(next_item_idx)
                relevance[next_item_idx] = np.NINF
                continue

            # 既に選択したアイテムとの最大類似度を計算
            max_similarities = np.max(similarity[selected_indices][:, :], axis=0)

            # MMRスコアの計算
            mmr_score = (
                self.lambda_factor * relevance
                - (1 - self.lambda_factor) * max_similarities
            )
            next_item_idx = np.argmax(mmr_score)

            final_recommendations.append(candidates[next_item_idx])
            selected_indices.append(next_item_idx)
            relevance[next_item_idx] = np.NINF

        return final_recommendations
