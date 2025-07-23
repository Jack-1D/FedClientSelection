# utils/pncs_selector.py

import numpy as np
from itertools import combinations

class PNCSClientSelector:
    def __init__(self, total_clients, select_num, queue_length=4):
        self.K = total_clients
        self.J = select_num
        self.queue_len = queue_length
        self.queue = {}  # client_id -> cooldown

    def update_queue(self):
        for cid in list(self.queue.keys()):
            self.queue[cid] -= 1
            if self.queue[cid] <= 0:
                del self.queue[cid]

    def in_queue(self, cid):
        return cid in self.queue

    def cos_p_vectorized(self, gradients_matrix, p=4):
        """
        Vectorized computation of cos_p for all pairs
        gradients_matrix: (n_clients, grad_dim)
        Returns: (n_clients, n_clients) symmetric matrix
        """
        n = gradients_matrix.shape[0]
        
        # 預計算所有梯度的 p-norm
        norms = np.linalg.norm(gradients_matrix, ord=p, axis=1)  # (n,)
        
        # 避免除零
        valid_mask = norms > 1e-12
        if not valid_mask.any():
            return np.zeros((n, n))
        
        # 計算所有配對的 inner_lp
        cos_matrix = np.zeros((n, n))
        
        for i in range(n):
            if not valid_mask[i]:
                continue
            for j in range(i+1, n):
                if not valid_mask[j]:
                    continue
                
                g1, g2 = gradients_matrix[i], gradients_matrix[j]
                
                # 向量化的 inner_lp 計算
                u_plus_v = g1 + g2
                u_minus_v = g1 - g2
                
                try:
                    numerator = (np.linalg.norm(u_plus_v, ord=p) - np.linalg.norm(u_minus_v, ord=p)) / 4
                    denominator = norms[i] * norms[j] + 1e-12
                    cos_val = numerator / denominator
                    
                    if not (np.isnan(cos_val) or np.isinf(cos_val)):
                        cos_matrix[i, j] = cos_val
                        cos_matrix[j, i] = cos_val  # 對稱矩陣
                except:
                    continue
        
        return cos_matrix

    def average_cos4_vectorized(self, subset, cos_matrix):
        """
        Fast computation using precomputed cos_p matrix
        """
        if len(subset) < 2:
            return 0.0
        
        # 提取子集對應的餘弦值
        pairwise_cos = []
        for i, client_i in enumerate(subset):
            for j, client_j in enumerate(subset):
                if i < j:  # 避免重複和自己與自己比較
                    cos_val = cos_matrix[client_i, client_j]
                    if not (np.isnan(cos_val) or np.isinf(cos_val)):
                        pairwise_cos.append(cos_val)
        
        if not pairwise_cos:
            return float('inf')
        
        return np.mean(pairwise_cos)

    def cos_p(self, g1, g2, p=4):
        # 保留原始函數以備單獨使用
        if len(g1) == 0 or len(g2) == 0:
            return 0.0
        
        if len(g1) != len(g2):
            return 0.0
            
        if np.isnan(g1).any() or np.isnan(g2).any():
            return 0.0
        if np.isinf(g1).any() or np.isinf(g2).any():
            return 0.0
            
        def inner_lp(u, v):
            try:
                result = (np.linalg.norm(u + v, ord=p) - np.linalg.norm(u - v, ord=p)) / 4
                return result
            except Exception as e:
                return 0.0
        
        numerator = inner_lp(g1, g2)
        norm_g1 = np.linalg.norm(g1, ord=p)
        norm_g2 = np.linalg.norm(g2, ord=p)
        
        if norm_g1 == 0 or norm_g2 == 0:
            return 0.0
            
        denominator = norm_g1 * norm_g2 + 1e-12
        result = numerator / denominator
        
        if np.isnan(result) or np.isinf(result):
            return 0.0
            
        return result

    def average_cos4(self, subset, gradients):
        # 保留原始函數以備使用
        if len(subset) < 2:
            return 0.0
            
        pairwise_cos = []
        for i, j in combinations(subset, 2):
            cos_val = self.cos_p(gradients[i], gradients[j], p=4)
            pairwise_cos.append(cos_val)
        
        if not pairwise_cos:
            return 0.0
            
        valid_cos = [x for x in pairwise_cos if not (np.isnan(x) or np.isinf(x))]
        if not valid_cos:
            return float('inf')
            
        return np.mean(valid_cos)

    def select_clients(self, gradients, max_combinations=10000):
        import time
        from math import comb
        start_time = time.time()
        
        self.update_queue()
        
        # 檢查 gradients 的有效性和長度一致性
        print(f"Total gradients received: {len(gradients)}")
        valid_gradients = {}
        gradient_lengths = []
        
        for cid, grad in gradients.items():
            if grad is not None and len(grad) > 0 and not np.isnan(grad).all():
                gradient_lengths.append(len(grad))
                valid_gradients[cid] = grad
            else:
                print(f"Warning: Invalid gradient for client {cid}")
        
        # 檢查所有有效梯度的長度是否一致
        if gradient_lengths:
            unique_lengths = set(gradient_lengths)
            if len(unique_lengths) > 1:
                print(f"Error: Inconsistent gradient lengths detected: {unique_lengths}")
                from collections import Counter
                most_common_length = Counter(gradient_lengths).most_common(1)[0][0]
                print(f"Using most common length: {most_common_length}")
                
                filtered_gradients = {}
                for cid, grad in valid_gradients.items():
                    if len(grad) == most_common_length:
                        filtered_gradients[cid] = grad
                    else:
                        print(f"Excluding client {cid} due to length mismatch: {len(grad)} vs {most_common_length}")
                valid_gradients = filtered_gradients
        
        print(f"Valid gradients: {len(valid_gradients)}")
        
        candidates = [cid for cid in valid_gradients if not self.in_queue(cid)]
        print(f"Candidates (not in queue): {candidates}")
        
        if len(candidates) < self.J:
            print(f"Error: Too few candidates ({len(candidates)}) to select {self.J} clients")
            if len(candidates) > 0:
                for cid in candidates:
                    self.queue[cid] = self.queue_len
                return candidates
            else:
                raise ValueError("No valid candidates to select from")

        # 檢查組合數量，決定使用精確算法還是近似算法
        num_combinations = comb(len(candidates), self.J)
        print(f"Number of combinations to evaluate: {num_combinations}")
        
        if num_combinations > max_combinations:
            print(f"Too many combinations ({num_combinations}), switching to fast approximation")
            return self.select_clients_fast_approximation(gradients, max_combinations)

        # === 使用向量化版本加速 ===
        preparation_time = time.time()
        
        # 將梯度轉換為矩陣形式
        client_ids = list(valid_gradients.keys())
        id_to_idx = {cid: idx for idx, cid in enumerate(client_ids)}
        gradients_matrix = np.stack([valid_gradients[cid] for cid in client_ids])
        
        print(f"Computing cos_p matrix for {len(client_ids)} clients...")
        cos_matrix = self.cos_p_vectorized(gradients_matrix)
        
        vectorization_time = time.time()
        print(f"Cos_p matrix computed in {vectorization_time - preparation_time:.3f}s")
        
        best_subset = None
        min_score = float('inf')
        
        subset_count = 0
        candidate_indices = [id_to_idx[cid] for cid in candidates]
        
        for subset_indices in combinations(candidate_indices, self.J):
            subset_count += 1
            avg_cos = self.average_cos4_vectorized(subset_indices, cos_matrix)
            
            if subset_count <= 10:  # 只印前10個子集的詳細資訊
                subset_cids = [client_ids[idx] for idx in subset_indices]
                print(f"Subset {subset_cids}: average_cos4 = {avg_cos}")
            
            if not (np.isnan(avg_cos) or np.isinf(avg_cos)) and avg_cos < min_score:
                min_score = avg_cos
                best_subset = subset_indices
        
        selection_time = time.time()
        print(f"Evaluated {subset_count} subsets in {selection_time - vectorization_time:.3f}s")
        print(f"Total selection time: {selection_time - start_time:.3f}s")
        print(f"Best score: {min_score}")
        
        if best_subset is None:
            print("Warning: No valid subset found, selecting first J candidates")
            best_subset = candidate_indices[:self.J]

        # 轉換回客戶端 ID
        selected_clients = [client_ids[idx] for idx in best_subset]
        
        for cid in selected_clients:
            self.queue[cid] = self.queue_len

        print(f"Selected clients: {selected_clients}")
        return selected_clients

    def select_clients_fast_approximation(self, gradients, max_combinations=10000):
        """
        Fast approximation when the number of combinations is too large
        Uses greedy selection based on gradient diversity
        """
        import time
        start_time = time.time()
        
        self.update_queue()
        
        # 數據預處理（與原函數相同）
        valid_gradients = {}
        gradient_lengths = []
        
        for cid, grad in gradients.items():
            if grad is not None and len(grad) > 0 and not np.isnan(grad).all():
                gradient_lengths.append(len(grad))
                valid_gradients[cid] = grad
        
        if gradient_lengths:
            unique_lengths = set(gradient_lengths)
            if len(unique_lengths) > 1:
                from collections import Counter
                most_common_length = Counter(gradient_lengths).most_common(1)[0][0]
                filtered_gradients = {}
                for cid, grad in valid_gradients.items():
                    if len(grad) == most_common_length:
                        filtered_gradients[cid] = grad
                valid_gradients = filtered_gradients
        
        candidates = [cid for cid in valid_gradients if not self.in_queue(cid)]
        
        if len(candidates) < self.J:
            if len(candidates) > 0:
                for cid in candidates:
                    self.queue[cid] = self.queue_len
                return candidates
            else:
                raise ValueError("No valid candidates to select from")
        
        # Greedy selection: 選擇最分散的梯度
        client_ids = list(valid_gradients.keys())
        id_to_idx = {cid: idx for idx, cid in enumerate(client_ids)}
        gradients_matrix = np.stack([valid_gradients[cid] for cid in client_ids])
        
        # 計算所有候選者的餘弦相似度矩陣
        candidate_indices = [id_to_idx[cid] for cid in candidates]
        candidate_gradients = gradients_matrix[candidate_indices]
        
        # 計算候選者之間的成對相似度
        n_candidates = len(candidate_indices)
        similarity_matrix = np.zeros((n_candidates, n_candidates))
        
        for i in range(n_candidates):
            for j in range(i+1, n_candidates):
                # 使用簡化的餘弦相似度
                cos_sim = np.dot(candidate_gradients[i], candidate_gradients[j]) / (
                    np.linalg.norm(candidate_gradients[i]) * np.linalg.norm(candidate_gradients[j]) + 1e-12
                )
                similarity_matrix[i, j] = cos_sim
                similarity_matrix[j, i] = cos_sim
        
        # Greedy selection: 選擇相似度最低的組合
        selected_indices = []
        
        # 選擇第一個客戶端（隨機或選擇梯度範數最大的)
        norms = np.linalg.norm(candidate_gradients, axis=1)
        selected_indices.append(np.argmax(norms))
        
        # 貪婪地選擇剩餘的客戶端
        for _ in range(self.J - 1):
            best_candidate = -1
            min_max_similarity = float('inf')
            
            for i in range(n_candidates):
                if i in selected_indices:
                    continue
                
                # 計算與已選擇客戶端的最大相似度
                max_similarity = max(similarity_matrix[i, j] for j in selected_indices)
                
                if max_similarity < min_max_similarity:
                    min_max_similarity = max_similarity
                    best_candidate = i
            
            if best_candidate != -1:
                selected_indices.append(best_candidate)
        
        # 轉換回客戶端 ID
        selected_clients = [candidates[i] for i in selected_indices]
        
        for cid in selected_clients:
            self.queue[cid] = self.queue_len
        
        end_time = time.time()
        print(f"Fast approximation selection completed in {end_time - start_time:.3f}s")
        print(f"Selected clients: {selected_clients}")
        
        return selected_clients
