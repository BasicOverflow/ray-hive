"""Test script to deploy one replica on the 12GB GPU on ergos02nv."""
import os
import sys
import ray
from ray import serve

vram_scheduler_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, vram_scheduler_dir)

import ray_utils
import vram_allocator
import vllm_model_actor

VLLMModel = vllm_model_actor.VLLMModel
get_vram_allocator = vram_allocator.get_vram_allocator

SUPPRESS_LOGGING = False

MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
ACTUAL_VRAM_GB = 3.89  # Total VRAM per replica (model + KV cache + overhead)
VRAM_BUFFER_GB = 0.1  # Hard buffer to leave free on every GPU (GB)
TARGET_NODE = "ergos-02-nv"  # Node to target (exact name from GPU keys)
TARGET_GPU_ID = "2"  # Specific GPU ID to target
TARGET_GPU_KEY = f"{TARGET_NODE}:gpu{TARGET_GPU_ID}"  # Hardcoded: ergos-02-nv:gpu2

def main():
    ray_utils.init_ray(suppress_logging=SUPPRESS_LOGGING)
    
    allocator = get_vram_allocator()
    
    import time
    time.sleep(1)
    
    state = ray.get(allocator.get_all_gpus.remote())
    
    print(f"\nTargeting SPECIFIC GPU: {TARGET_GPU_KEY}")
    print(f"  Node: {TARGET_NODE}")
    print(f"  GPU ID: {TARGET_GPU_ID}")
    
    # Directly look for the hardcoded GPU key
    target_gpu_info = state.get(TARGET_GPU_KEY)
    
    if not target_gpu_info:
        print(f"\n❌ Could not find GPU key '{TARGET_GPU_KEY}' in VRAM monitoring")
        print("\nAvailable GPUs:")
        for gpu_key, info in state.items():
            if len(gpu_key) > 50 or gpu_key.startswith('c'):
                continue
            node_name = gpu_key.split(":")[0]
            total_gb = info.get("total", 0)
            free_gb = info.get("free", 0)
            available_gb = info.get("available", 0)
            marker = " ✅ TARGET" if gpu_key == TARGET_GPU_KEY else ""
            print(f"  {gpu_key}: {total_gb:.2f}GB total, {free_gb:.2f}GB free, {available_gb:.2f}GB available{marker}")
        return
    
    target_gpu_key = TARGET_GPU_KEY
    target_gpu_id = TARGET_GPU_ID
    
    print(f"\n✅ Found target GPU from VRAM monitoring:")
    print(f"  GPU ID: {target_gpu_id} (HARDCODED)")
    print(f"  GPU Key: {target_gpu_key} (HARDCODED)")
    print(f"  Node: {TARGET_NODE}")
    print(f"  Total VRAM: {target_gpu_info.get('total', 0):.2f}GB")
    print(f"  Free VRAM: {target_gpu_info.get('free', 0):.2f}GB")
    print(f"  Available VRAM: {target_gpu_info.get('available', 0):.2f}GB")
    print(f"  Pending: {target_gpu_info.get('pending', 0):.2f}GB")
    print(f"  Active: {target_gpu_info.get('active', 0):.2f}GB")
    
    available_gb = target_gpu_info.get("available", 0)
    available_with_buffer = max(0, available_gb - VRAM_BUFFER_GB)
    
    if available_with_buffer < ACTUAL_VRAM_GB:
        print(f"\n❌ Not enough VRAM: need {ACTUAL_VRAM_GB:.2f}GB, have {available_with_buffer:.2f}GB available (after {VRAM_BUFFER_GB:.2f}GB buffer)")
        return
    
    # Create resource name (matching model_orchestrator format)
    # Use the EXACT node name and GPU ID (hardcoded)
    node_name_from_key = TARGET_NODE
    gpu_id_from_key = TARGET_GPU_ID
    resource_name = f"{node_name_from_key}_gpu{gpu_id_from_key}"
    
    print(f"\nResource name construction (HARDCODED GPU ID):")
    print(f"  GPU key: {target_gpu_key}")
    print(f"  Node name: {node_name_from_key}")
    print(f"  GPU ID: {target_gpu_id} (HARDCODED)")
    print(f"  Resource name: {resource_name}")
    print(f"  → This will deploy to GPU {target_gpu_id} on {node_name_from_key}")
    
    # Verify resource exists in Ray cluster
    print(f"\nVerifying resource availability...")
    cluster_resources = ray.cluster_resources()
    matching_resources = {k: v for k, v in cluster_resources.items() if resource_name == k or k.startswith(node_name_from_key + "_")}
    if matching_resources:
        print(f"  Found matching resources in cluster:")
        for res_name, res_value in matching_resources.items():
            print(f"    {res_name}: {res_value}")
        if resource_name not in matching_resources:
            print(f"  ⚠️  WARNING: Exact resource '{resource_name}' not found, but found similar resources")
    else:
        print(f"  ⚠️  Resource '{resource_name}' not found in cluster resources")
        print(f"  Available resources containing '{node_name_from_key}':")
        for res_name, res_value in cluster_resources.items():
            if node_name_from_key in res_name:
                print(f"    {res_name}: {res_value}")
    
    # Check node-level resources to see where the resource actually exists
    print(f"\nChecking node-level resources for '{resource_name}':")
    nodes = ray.nodes()
    found_on_nodes = []
    for node in nodes:
        node_resources = node.get("Resources", {})
        if resource_name in node_resources:
            node_id = node.get("NodeID", "unknown")
            node_name = node.get("NodeName", "unknown")
            resource_value = node_resources[resource_name]
            found_on_nodes.append((node_id, node_name, resource_value))
            print(f"  ✅ Found on node: {node_name} (ID: {node_id[:8]}...), value: {resource_value}")
    
    if not found_on_nodes:
        print(f"  ❌ Resource '{resource_name}' not found on any node!")
        print(f"  Checking all GPU resources on all nodes:")
        for node in nodes:
            node_resources = node.get("Resources", {})
            node_name = node.get("NodeName", "unknown")
            gpu_resources = {k: v for k, v in node_resources.items() if "_gpu" in k}
            if gpu_resources:
                print(f"    Node {node_name}: {gpu_resources}")
        print(f"\n  ⚠️  WARNING: Resource not found! Deployment may fail or schedule incorrectly.")
        return
    else:
        print(f"  Found resource on {len(found_on_nodes)} node(s)")
        # Check if resource has available capacity
        for node_id, node_name, resource_value in found_on_nodes:
            if resource_value < 1:
                print(f"  ⚠️  WARNING: Resource '{resource_name}' on node {node_name} has value {resource_value} < 1")
            else:
                print(f"  ✅ Resource '{resource_name}' on node {node_name} has value {resource_value} (sufficient)")
    
    # Calculate GPU fraction
    total_gb = target_gpu_info.get("total", 12.0)
    gpu_fraction = ACTUAL_VRAM_GB / total_gb
    gpu_fraction = max(gpu_fraction, 0.01)
    gpu_fraction = round(gpu_fraction, 2)
    
    print(f"\nDeployment details:")
    print(f"  Target node: {node_name_from_key}")
    print(f"  Target GPU ID: {target_gpu_id} (HARDCODED)")
    print(f"  Target GPU key: {target_gpu_key} (HARDCODED)")
    print(f"  Resource: {resource_name} = 1")
    print(f"  GPU fraction per replica: {gpu_fraction:.2f}")
    print(f"  VRAM per replica: {ACTUAL_VRAM_GB:.2f}GB")
    print(f"  VRAM buffer: {VRAM_BUFFER_GB:.2f}GB")
    
    # Create deployment targeting this specific GPU
    model_id = "test-ergos02nv-12gb"
    deployment_name = f"{model_id}-{target_gpu_key.replace(':', '-').replace('_', '-')}"
    app_name = f"{model_id}-gpu-{target_gpu_key.replace(':', '-')}"
    
    print(f"\nCreating deployment:")
    print(f"  Deployment name: {deployment_name}")
    print(f"  Application name: {app_name}")
    print(f"  Route prefix: /{deployment_name}")
    
    # Set CUDA_VISIBLE_DEVICES to force the actor to use only GPU 2
    # This ensures torch.cuda.current_device() returns device 0, which maps to physical GPU 2
    import os
    cuda_visible_devices = TARGET_GPU_ID
    
    print(f"\n⚠️  Setting CUDA_VISIBLE_DEVICES={cuda_visible_devices} to force GPU {TARGET_GPU_ID}")
    print(f"  This ensures the actor only sees GPU {TARGET_GPU_ID} on the node")
    
    serve.run(
        VLLMModel.options(
            name=deployment_name,
            ray_actor_options={
                "num_gpus": gpu_fraction,
                "memory": 2 * 1024 * 1024 * 1024,
                "resources": {
                    resource_name: 1
                }
            },
            autoscaling_config={
                "min_replicas": 1,
                "max_replicas": 1
            }
        ).bind(
            model_id=model_id,
            model_name=MODEL_NAME,
            required_vram_gb=ACTUAL_VRAM_GB,
            target_gpu_id=TARGET_GPU_ID
        ),
        name=app_name,
        route_prefix=f"/{deployment_name}"
    )
    
    print(f"\n✅ Deployment created! Waiting for replica to initialize...")
    
    # Wait a bit and check status
    time.sleep(5)
    try:
        status = serve.status()
        for app_name_check, app_info in status.applications.items():
            if app_name_check == app_name and hasattr(app_info, 'deployments') and deployment_name in app_info.deployments:
                dep = app_info.deployments[deployment_name]
                if hasattr(dep, 'replicas') and dep.replicas:
                    for replica in dep.replicas:
                        if hasattr(replica, 'state'):
                            print(f"  Replica state: {replica.state}")
                        # Check where the replica is actually running
                        if hasattr(replica, 'actor_id'):
                            try:
                                actor_handle = ray.get_actor(replica.actor_id)
                                actor_resources = ray.get_runtime_context().get_node_id()
                                print(f"  Replica actor ID: {replica.actor_id}")
                                # Try to get node info
                                nodes = ray.nodes()
                                for node in nodes:
                                    if node.get("NodeID") == actor_resources:
                                        node_name = node.get("NodeName", "unknown")
                                        print(f"  ⚠️  Replica is running on node: {node_name}")
                                        # Check what resources this node has
                                        node_resources = node.get("Resources", {})
                                        gpu_resources = {k: v for k, v in node_resources.items() if "_gpu" in k}
                                        if gpu_resources:
                                            print(f"  Node GPU resources: {gpu_resources}")
                                        break
                            except Exception as e:
                                print(f"  Could not get replica location: {e}")
    except Exception as e:
        print(f"  Could not check status: {e}")
    
    print(f"\n✅ Test deployment complete!")
    print(f"  Access at: /{deployment_name}")

if __name__ == "__main__":
    main()

