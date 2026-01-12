"""Model Router - load balances across multiple GPU deployments."""
import ray
from ray import serve
from typing import Dict, List, Optional, Union


@serve.deployment(
    ray_actor_options={"num_cpus": 0.1},
    autoscaling_config=None,
    num_replicas=1
)
class ModelRouter:
    """Router that load balances requests across multiple GPU deployments using weighted selection."""
    
    def __init__(self, model_id: str, gpu_deployment_names: List[str]):
        """Initialize router with list of GPU deployment names to route to.
        
        Args:
            model_id: Model identifier
            gpu_deployment_names: List of deployment names (one per GPU)
        """
        self.model_id = model_id
        self.gpu_deployment_names = gpu_deployment_names
        self._handles = None
        self._weights = None
        self._current_weights = None  # Track current_weight for each deployment
    
    def _get_handles(self):
        """Lazy initialization of deployment handles."""
        if self._handles is None:
            self._handles = []
            for deployment_name in self.gpu_deployment_names:
                # Each GPU deployment is in its own application: {model_id}-{gpu_name}
                app_name = f"{self.model_id}-{deployment_name}"
                try:
                    handle = serve.get_deployment_handle(deployment_name, app_name=app_name)
                    self._handles.append(handle)
                except Exception as e:
                    print(f"Warning: Could not get handle for {deployment_name}: {e}")
        
        # Filter out None handles
        self._handles = [h for h in self._handles if h is not None]
        return self._handles
    
    async def _initialize_weights(self):
        """Initialize weights based on max_num_seqs from each deployment."""
        if self._weights is not None:
            return
        
        handles = self._get_handles()
        if not handles:
            raise RuntimeError(f"No available GPU deployments for model {self.model_id}")
        
        # Query max_num_seqs from each deployment
        weights = []
        for handle in handles:
            try:
                max_num_seqs = await handle.get_max_num_seqs.remote()
                # Use max_num_seqs as weight (higher max_num_seqs = more requests)
                weights.append(max(max_num_seqs, 1))  # Ensure at least 1
            except Exception as e:
                print(f"Warning: Could not get max_num_seqs, using default weight: {e}")
                weights.append(32)  # Default weight
        
        self._weights = weights
        
        # Initialize current_weights to 0 for each deployment
        self._current_weights = [0] * len(weights)
        
        total_weight = sum(weights)
        print(f"[Router] Initialized weighted load balancing: {dict(zip(self.gpu_deployment_names, weights))}")
        print(f"[Router] Total weight: {total_weight}")
    
    def _get_next_handle(self):
        """Get next handle using weighted round-robin (WRR algorithm)."""
        if self._weights is None or self._current_weights is None:
            raise RuntimeError("Weights not initialized. Call _initialize_weights() first.")
        
        handles = self._get_handles()
        if not handles:
            raise RuntimeError(f"No available GPU deployments for model {self.model_id}")
        
        if len(handles) != len(self._weights):
            raise RuntimeError(f"Mismatch: {len(handles)} handles but {len(self._weights)} weights")
        
        # Weighted Round-Robin algorithm
        # 1. Add each deployment's weight to its current_weight
        # 2. Select the deployment with the highest current_weight
        # 3. Subtract total weight from selected deployment's current_weight
        
        total_weight = sum(self._weights)
        max_current_weight = -1
        selected_index = 0
        
        for i, weight in enumerate(self._weights):
            # Add weight to current_weight
            self._current_weights[i] += weight
            
            # Track the highest
            if self._current_weights[i] > max_current_weight:
                max_current_weight = self._current_weights[i]
                selected_index = i
        
        # Subtract total weight from selected deployment
        self._current_weights[selected_index] -= total_weight
        
        return handles[selected_index]
    
    async def __call__(self, request: Union[str, Dict]) -> Union[str, List]:
        """Route request to a GPU deployment using weighted load balancing based on max_num_seqs."""
        # Initialize weights on first call
        await self._initialize_weights()
        
        handle = self._get_next_handle()
        
        # Forward request to the selected GPU deployment
        # Supports both single prompts and batch prompts
        if isinstance(request, dict):
            # Already a dict - forward as-is (handles both "prompt" and "prompts" keys)
            return await handle.remote(request)
        else:
            # Single string prompt - wrap in dict
            return await handle.remote({"prompt": request})
    
    async def get_max_num_seqs(self) -> int:
        """Get max_num_seqs from first available GPU deployment."""
        handles = self._get_handles()
        if not handles:
            return 32  # Default fallback
        
        try:
            return await handles[0].get_max_num_seqs.remote()
        except Exception:
            return 32  # Default fallback

