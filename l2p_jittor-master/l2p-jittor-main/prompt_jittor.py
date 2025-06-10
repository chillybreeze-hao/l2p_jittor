import jittor as jt
import jittor.nn as nn

def compatible_unique(tensor, return_counts=False, sorted_result=True):
    """
    兼容PyTorch和Jittor的unique函数
    
    Args:
        tensor: 输入张量
        return_counts: 是否返回计数
        sorted_result: 是否需要排序结果
    
    Returns:
        unique_values: 唯一值
        counts (可选): 对应的计数
    """
    
    if return_counts:
        unique_vals, counts = jt.unique(tensor, return_counts=True)
        
        if sorted_result:
            # 手动排序
            sorted_indices = jt.argsort(unique_vals)
            unique_vals = unique_vals[sorted_indices]
            counts = counts[sorted_indices]
        
        return unique_vals, counts
    else:
        unique_vals = jt.unique(tensor)
        
        if sorted_result:
            unique_vals = jt.sort(unique_vals)[0]
        
        return unique_vals
    
    
class Prompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', 
                 prompt_pool=False, prompt_key=False, pool_size=None, top_k=None, 
                 batchwise_prompt=False, prompt_key_init='uniform',):
        super().__init__()
        self.length = length
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        if self.prompt_pool:
            prompt_pool_shape = (pool_size, length, embed_dim)
            if prompt_init == 'zero':
                self.prompt = jt.zeros(prompt_pool_shape)
            elif prompt_init == 'uniform':
                self.prompt = jt.randn(prompt_pool_shape)
                jt.init.uniform_(self.prompt, -1, 1)
        
        # if using learnable prompt keys
        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = jt.zeros(key_shape)
            elif prompt_key_init == 'uniform':
                self.prompt_key = jt.randn(key_shape)
                jt.init.uniform_(self.prompt_key, -1, 1)
        else:
            prompt_mean = jt.mean(self.prompt, dim=1)
            self.prompt_key = prompt_mean
    
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = jt.sum(x ** 2, dim=dim, keepdims=True)
        x_inv_norm = jt.rsqrt(jt.maximum(square_sum, jt.array(epsilon)))
        return x * x_inv_norm
    
    def execute(self, x_embed, prompt_mask=None, cls_features=None):
        out = dict()
        if self.prompt_pool:
            # Calculate embedding keys
            if self.embedding_key == 'mean':
                x_embed_mean = jt.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = jt.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = jt.max(x_embed, dim=1)[0] + 2 * jt.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = jt.max(x_embed, dim=1)[0]  # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")
            # Normalize prompt keys and embeddings
            prompt_norm = self.l2_normalize(self.prompt_key, dim=1)  # Pool_size, C
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1)  # B, C
            # Calculate similarity
            similarity = jt.matmul(x_embed_norm, prompt_norm.transpose(-1, -2))  # B, Pool_size
            
            if prompt_mask is None:
                # Get top-k similar prompts
                _, idx = jt.topk(similarity, k=self.top_k, dim=1)  # B, top_k
                
                if self.batchwise_prompt:
                    # Get unique prompt indices and their counts
                    prompt_id, _, id_counts = jt.unique(idx, return_inverse=True, return_counts=True)
                    
                    # Pad if necessary
                    if prompt_id.shape[0] < self.pool_size:
                        prompt_id = jt.concat([prompt_id, jt.full((self.pool_size - prompt_id.shape[0],), jt.min(idx.flatten()))])
                        id_counts = jt.concat([id_counts, jt.full((self.pool_size - id_counts.shape[0],), 0)])
                    
                    # Get top-k most frequent prompts
                    _, major_idx = jt.topk(id_counts, k=self.top_k)  # top_k
                    major_prompt_id = prompt_id[major_idx]  # top_k
                    
                    # Expand to batch
                    idx = major_prompt_id.expand(x_embed.shape[0], -1) # B, top_k
            else:
                idx = prompt_mask  # B, top_k
            # Get batched prompts
            batched_prompt_raw = self.prompt[idx]  # B, top_k, length, C
            batch_size, top_k, length, c = batched_prompt_raw.shape
            batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c)
            out['prompt_idx'] = idx
            # Debugging outputs
            out['prompt_norm'] = prompt_norm
            out['x_embed_norm'] = x_embed_norm
            out['similarity'] = similarity
            # Pull constraint loss calculation
            batched_key_norm = prompt_norm[idx]  # B, top_k, C
            out['selected_key'] = batched_key_norm
            x_embed_norm = x_embed_norm.unsqueeze(1)  # B, 1, C
            sim = batched_key_norm * x_embed_norm # B, top_k, C
            reduce_sim = jt.sum(sim) / x_embed.shape[0]  # Scalar
            
            out['reduce_sim'] = reduce_sim
        else:
            # Non-pooled prompts
            if self.prompt_init == 'zero':
                self.prompt = jt.zeros(self.length, self.embed_dim)
            elif self.prompt_init == 'uniform':
                self.prompt = jt.randn(self.length, self.embed_dim)
                jt.init.uniform_(self.prompt)
            
            batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)
        
        # Concatenate prompt with input embeddings
        out['total_prompt_len'] = batched_prompt.shape[1]
        out['prompted_embedding'] = jt.cat([batched_prompt, x_embed], dim=1)
        return out
    
