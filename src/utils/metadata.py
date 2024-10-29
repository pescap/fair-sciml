from datetime import datetime
import platform
import psutil
import cpuinfo
from typing import Dict, Any


class MetadataCollector:
    """Handles collection and storage of simulation metadata."""

    @staticmethod
    def collect_session_metadata(num_simulations: int) -> Dict[str, Any]:
        """Collect session-level metadata about hardware and execution environment."""
        cpu_info = cpuinfo.get_cpu_info()
        return {
            "timestamp": datetime.now().isoformat(),
            "hardware": {
                "machine": platform.machine(),
                "processor": cpu_info.get("brand_raw", "Unknown"),
                "architecture": cpu_info.get("arch", "Unknown"),
                "system": platform.system(),
                "version": platform.version(),
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "total_memory": f"{psutil.virtual_memory().total / (1024 ** 3):.2f} GB",
            },
            "num_simulations": num_simulations,
        }

    @staticmethod
    def collect_simulation_metadata(
        parameters: Dict[str, Any], execution_time: float
    ) -> Dict[str, Any]:
        """Collect simulation-specific metadata."""
        parameters = {
            f"parameter_{k}": (v.item() if hasattr(v, "item") else v)
            for k, v in parameters.items()
        }
        return {
            "parameters": parameters,
            "execution_time": f"{execution_time:.4f} seconds",
        }
