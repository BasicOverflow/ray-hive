"""
Test Ray REST API endpoints.

Verifies:
- Ray dashboard is accessible
- Cluster status endpoint works
- Nodes endpoint (if available)
"""
import requests
import sys

RAY_DASHBOARD_URL = "http://10.0.1.53:8265"

def main():
    try:
        # Test cluster status
        print(f"Testing Ray dashboard at {RAY_DASHBOARD_URL}...")
        status_resp = requests.get(f"{RAY_DASHBOARD_URL}/api/cluster_status", timeout=5)
        status_resp.raise_for_status()
        status = status_resp.json()
        print(f"✅ Cluster status: {status.get('result', False)}")
        
        # Test nodes endpoint (may not be available in all Ray versions)
        try:
            nodes_resp = requests.get(f"{RAY_DASHBOARD_URL}/api/nodes", timeout=5)
            if nodes_resp.status_code == 200:
                try:
                    nodes_data = nodes_resp.json()
                    if isinstance(nodes_data, dict):
                        node_count = len(nodes_data.get("data", {}).get("nodes", []))
                    else:
                        node_count = len(nodes_data) if isinstance(nodes_data, list) else 0
                    print(f"✅ Nodes endpoint accessible (found {node_count} nodes)")
                except ValueError:
                    print(f"⚠️  Nodes endpoint returned non-JSON response")
            else:
                print(f"ℹ️  Nodes endpoint not available (404) - this is normal for some Ray versions")
        except requests.exceptions.RequestException:
            print(f"ℹ️  Nodes endpoint not available - this is normal for some Ray versions")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error connecting to Ray dashboard: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
