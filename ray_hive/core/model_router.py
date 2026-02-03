"""Model Router - dynamic capacity-aware load balancing."""
from ray import serve
from typing import Dict, List, Union
import asyncio
import time


@serve.deployment(
    ray_actor_options={"num_cpus": 0.1},
    autoscaling_config=None,
    num_replicas=1
)
class ModelRouter:
    """Router that dynamically load balances based on real-time capacity and queue depth."""
    
    def __init__(self, model_id: str, gpu_deployment_names: List[str]):
        self.model_id = model_id
        self.gpu_deployment_names = gpu_deployment_names
        self._handles = None
        self._capacity_cache = {}
        self._queue_depth = {}
        self._performance_factors = {}
        self._max_vram_gb = None
        self._cache_ttl = 0.1
        self._last_update = {}
        self._lock = asyncio.Lock()
    
    def _get_handles(self):
        if self._handles is None:
            self._handles = {}
            for deployment_name in self.gpu_deployment_names:
                handle = serve.get_deployment_handle(deployment_name, app_name=deployment_name)
                self._handles[deployment_name] = handle
        return self._handles
    
    async def _get_capacity_info(self, deployment_name: str) -> Dict:
        """Get capacity info for a deployment (cached)."""
        current_time = time.time()
        
        if (deployment_name not in self._capacity_cache or 
            deployment_name not in self._last_update or
            current_time - self._last_update[deployment_name] > self._cache_ttl):
            handles = self._get_handles()
            capacity_info = await handles[deployment_name].get_capacity_info.remote()
            self._capacity_cache[deployment_name] = capacity_info
            self._last_update[deployment_name] = current_time
            
            # Update max VRAM if needed
            if capacity_info.get("total_vram_gb") is not None:
                if self._max_vram_gb is None or capacity_info["total_vram_gb"] > self._max_vram_gb:
                    self._max_vram_gb = capacity_info["total_vram_gb"]
                    # Recalculate performance factors when max VRAM changes
                    self._performance_factors = {}
        
        return self._capacity_cache[deployment_name]
    
    def _estimate_request_size(self, request: Union[str, Dict]) -> int:
        """Estimate request size in tokens."""
        if isinstance(request, dict):
            prompt = request.get("prompt") or (request.get("prompts", [""])[0] if request.get("prompts") else "")
            max_tokens = request.get("max_tokens", 100)
        else:
            prompt = str(request)
            max_tokens = 100
        
        # Rough token estimate: word count * 1.3 (approximate tokens per word)
        if isinstance(prompt, list):
            prompt = " ".join(str(p) for p in prompt)
        prompt_tokens = int(len(str(prompt).split()) * 1.3)
        
        return int(prompt_tokens + max_tokens)
    
    async def _calculate_performance_factor(self, deployment_name: str) -> float:
        """Calculate performance factor for a deployment."""
        if deployment_name in self._performance_factors:
            return self._performance_factors[deployment_name]
        
        capacity_info = await self._get_capacity_info(deployment_name)
        sm_count = capacity_info.get("sm_count")
        total_vram_gb = capacity_info.get("total_vram_gb")
        
        if sm_count is None or total_vram_gb is None or self._max_vram_gb is None:
            # Fallback: use max_num_seqs as proxy
            max_num_seqs = capacity_info.get("max_num_seqs", 1)
            factor = float(max_num_seqs)
        else:
            # Normalize by max VRAM across all deployments
            vram_factor = total_vram_gb / self._max_vram_gb if self._max_vram_gb > 0 else 1.0
            factor = sm_count * vram_factor
        
        self._performance_factors[deployment_name] = factor
        return factor
    
    async def _calculate_score(self, deployment_name: str, request_size: int) -> float:
        """Calculate routing score for a deployment."""
        capacity_info = await self._get_capacity_info(deployment_name)
        performance_factor = await self._calculate_performance_factor(deployment_name)
        
        max_num_seqs = capacity_info.get("max_num_seqs", 1)
        queue_depth = self._queue_depth.get(deployment_name, 0)
        available_capacity = max(0, max_num_seqs - queue_depth)
        
        score = available_capacity * performance_factor
        
        # Penalize if request is too large for this deployment
        if request_size > max_num_seqs * 0.8:
            score *= 0.5
        
        return score
    
    async def _select_best_deployment(self, request: Union[str, Dict]) -> str:
        """Select best deployment based on capacity, queue depth, and request characteristics."""
        handles = self._get_handles()
        request_size = self._estimate_request_size(request)
        
        best_deployment = None
        best_score = float('-inf')
        
        for deployment_name in self.gpu_deployment_names:
            score = await self._calculate_score(deployment_name, request_size)
            if score > best_score:
                best_score = score
                best_deployment = deployment_name
        
        # Fallback: round-robin if all at capacity
        if best_deployment is None or best_score <= 0:
            import random
            best_deployment = random.choice(self.gpu_deployment_names)
        
        return best_deployment
    
    async def __call__(self, request: Union[str, Dict]) -> Union[str, List]:
        handles = self._get_handles()
        
        deployment_name = await self._select_best_deployment(request)
        handle = handles[deployment_name]
        
        # Track queue depth
        async with self._lock:
            self._queue_depth[deployment_name] = self._queue_depth.get(deployment_name, 0) + 1
        
        try:
            if isinstance(request, dict):
                result = await handle.remote(request)
            else:
                result = await handle.remote({"prompt": request})
            return result
        finally:
            # Decrement queue depth
            async with self._lock:
                self._queue_depth[deployment_name] = max(0, self._queue_depth.get(deployment_name, 0) - 1)
    
    async def get_max_num_seqs(self) -> int:
        """Get total max_num_seqs across all deployments."""
        handles = self._get_handles()
        total = 0
        for deployment_name in self.gpu_deployment_names:
            capacity_info = await self._get_capacity_info(deployment_name)
            total += capacity_info.get("max_num_seqs", 0)
        return total

