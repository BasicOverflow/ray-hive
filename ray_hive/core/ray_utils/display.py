"""Rich console helpers for Ray Hive banners, messages, and VRAM panels."""
import sys
from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Windows consoles often default to cp1252; banner uses box-drawing chars.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

console = Console(legacy_windows=False)

_BANNER = r"""
 ╱$$$$$$$                            ╱$$   ╱$$ ╱$$                    
│ $$__  $$                          │ $$  │ $$│__╱                    
│ $$  ╲ $$  ╱$$$$$$  ╱$$   ╱$$      │ $$  │ $$ ╱$$ ╱$$    ╱$$ ╱$$$$$$ 
│ $$$$$$$╱ │____  $$│ $$  │ $$      │ $$$$$$$$│ $$│  $$  ╱$$╱╱$$__  $$
│ $$__  $$  ╱$$$$$$$│ $$  │ $$      │ $$__  $$│ $$ ╲  $$╱$$╱│ $$$$$$$$
│ $$  ╲ $$ ╱$$__  $$│ $$  │ $$      │ $$  │ $$│ $$  ╲  $$$╱ │ $$_____╱
│ $$  │ $$│  $$$$$$$│  $$$$$$$      │ $$  │ $$│ $$   ╲  $╱  │  $$$$$$$
│__╱  │__╱ ╲_______╱ ╲____  $$      │__╱  │__╱│__╱    ╲_╱    ╲_______╱
                     ╱$$  │ $$                                        
                    │  $$$$$$╱                                        
                     ╲______╱                                         
"""

# Component rows for deploy plan printer (only render keys present).
_BREAKDOWN_ROWS = (
    ("weights_gb", "Weights"),
    ("weight_need_gb", "Weight need (per GPU)"),
    ("kv_cache_gb", "KV cache"),
    ("activation_gb", "Activations"),
    ("overhead_gb", "Overhead"),
    ("misc_gb", "Misc"),
    ("total_vram_gb", "Total (per GPU)"),
)


def print_banner():
    """Print Ray Hive ASCII banner and package version."""
    from ray_hive import __version__

    console.print(Text(_BANNER.lstrip("\n"), style="bold cyan"))
    console.print(Text(f"  ray-hive {__version__}", style="dim"))


def info(msg: str):
    console.print(f"[cyan]{msg}[/cyan]")


def warn(msg: str):
    console.print(f"[yellow]{msg}[/yellow]")


def error(msg: str):
    console.print(f"[red]{msg}[/red]")


def success(msg: str):
    console.print(f"[green]{msg}[/green]")


def print_panel(title: str, renderable: Any, style: str = "cyan"):
    """Print a bordered Rich Panel."""
    console.print(Panel(renderable, title=title, border_style=style, expand=False))


def _vram_breakdown_table(plan: dict) -> Table:
    """Shared GiB component rows for deploy (skips missing keys)."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(justify="right")
    for key, label in _BREAKDOWN_ROWS:
        if key not in plan:
            continue
        val = plan[key]
        style = "bold" if key == "total_vram_gb" else None
        table.add_row(label, f"{val:.2f} GiB", style=style)
    return table


def print_deployment_plan(model_id: str, results: dict):
    """Print packed plan summary for all replicas of one model."""
    panels = []
    for replica_id, summary in results.items():
        plan = summary["plan"]
        gpu_keys = summary["gpu_keys"]
        tp = plan["tensor_parallel_size"]

        meta = Table(show_header=False, box=None, padding=(0, 2))
        meta.add_column(style="dim")
        meta.add_column()
        meta.add_row("Replica", replica_id)
        meta.add_row("GPU(s)", ", ".join(gpu_keys))
        meta.add_row("tensor_parallel_size", str(tp))
        meta.add_row("max_num_seqs", str(plan["max_num_seqs"]))
        meta.add_row("max_num_batched_tokens", str(plan["max_num_batched_tokens"]))
        meta.add_row("gpu_memory_utilization", f"{plan['gpu_memory_utilization']:.3f}")
        if plan.get("pooling"):
            meta.add_row("mode", "pooling/embed")
        if plan.get("mm_tokens_per_prompt"):
            meta.add_row("mm_tokens_per_prompt", str(plan["mm_tokens_per_prompt"]))

        body = Group(meta, Text(""), _vram_breakdown_table(plan))
        panels.append(
            Panel(body, title=f"Deployment Plan: {model_id}", border_style="green", expand=False)
        )
    console.print(Group(*panels))
