"""
Dynamic Performance Optimizer for DaxNodes

Intelligently detects system capabilities and optimizes processing parameters
to maximize performance without overloading user devices.
"""

import torch
import psutil
import platform
import time
from typing import Dict, Tuple, Optional

class PerformanceOptimizer:
    """Dynamic performance optimization based on system capabilities"""
    
    def __init__(self):
        self.system_info = self._detect_system_capabilities()
        self.performance_profile = self._generate_performance_profile()
        self._benchmark_cache = {}
        
    def _detect_system_capabilities(self) -> Dict:
        """Detect comprehensive system capabilities"""
        info = {}
        
        # GPU Detection
        if torch.cuda.is_available():
            info['gpu_count'] = torch.cuda.device_count()
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info['gpu_compute_capability'] = torch.cuda.get_device_properties(0).major
        else:
            info['gpu_count'] = 0
            info['gpu_name'] = None
            info['gpu_memory_gb'] = 0
            info['gpu_compute_capability'] = 0
        
        # CPU Detection
        info['cpu_count'] = psutil.cpu_count(logical=True)
        info['cpu_physical_cores'] = psutil.cpu_count(logical=False)
        info['cpu_freq_max'] = psutil.cpu_freq().max if psutil.cpu_freq() else 3000
        
        # Memory Detection
        memory = psutil.virtual_memory()
        info['ram_total_gb'] = memory.total / (1024**3)
        info['ram_available_gb'] = memory.available / (1024**3)
        
        # Storage Detection (for temp files)
        disk = psutil.disk_usage('/')
        info['disk_free_gb'] = disk.free / (1024**3)
        
        # Platform
        info['platform'] = platform.system()
        info['is_wsl'] = 'microsoft' in platform.uname().release.lower()
        
        return info
    
    def _generate_performance_profile(self) -> str:
        """Generate performance profile based on detected capabilities"""
        gpu_mem = self.system_info['gpu_memory_gb']
        ram_gb = self.system_info['ram_total_gb']
        gpu_count = self.system_info['gpu_count']
        
        if gpu_count > 1 and gpu_mem >= 16:
            return "high_end"
        elif gpu_count == 1 and gpu_mem >= 12:
            return "high_performance"  
        elif gpu_count == 1 and gpu_mem >= 8:
            return "mid_range"
        elif gpu_count == 1 and gpu_mem >= 4:
            return "entry_level"
        else:
            return "cpu_only"
    
    def get_optimal_upscaler_settings(self, model_scale: float = 2.0) -> Dict:
        """Get optimal upscaler settings based on system capabilities"""
        profile = self.performance_profile
        gpu_mem = self.system_info['gpu_memory_gb']
        
        settings = {
            "batch_size": 1,
            "tile_size": 512,
            "overlap": 32,
            "memory_threshold_mb": 1000,
            "use_cpu_offload": False,
            "prefetch_frames": 2,
            "use_memory_mapping": True
        }
        
        if profile == "high_end":
            settings.update({
                "batch_size": 16,
                "tile_size": 1024,
                "overlap": 64,
                "memory_threshold_mb": int(gpu_mem * 800),  # Use 80% of GPU memory
                "prefetch_frames": 8
            })
        elif profile == "high_performance":
            settings.update({
                "batch_size": 8,
                "tile_size": 768,
                "overlap": 48,
                "memory_threshold_mb": int(gpu_mem * 700),
                "prefetch_frames": 6
            })
        elif profile == "mid_range":
            settings.update({
                "batch_size": 4,
                "tile_size": 512,
                "overlap": 32,
                "memory_threshold_mb": int(gpu_mem * 600),
                "prefetch_frames": 4
            })
        elif profile == "entry_level":
            settings.update({
                "batch_size": 2,
                "tile_size": 256,
                "overlap": 16,
                "memory_threshold_mb": int(gpu_mem * 500),
                "prefetch_frames": 2
            })
        else:  # CPU only
            settings.update({
                "batch_size": 1,
                "tile_size": 256,
                "overlap": 16,
                "memory_threshold_mb": 500,
                "use_cpu_offload": True,
                "prefetch_frames": 1,
                "use_memory_mapping": False  # CPU processing prefers direct loading
            })
        
        # Adjust for model scale (larger models need more memory)
        memory_factor = max(1.0, model_scale / 2.0)
        settings["memory_threshold_mb"] = int(settings["memory_threshold_mb"] / memory_factor)
        
        return settings
    
    def get_optimal_rife_settings(self, multiplier: int = 2) -> Dict:
        """Get optimal RIFE interpolation settings"""
        profile = self.performance_profile  
        gpu_mem = self.system_info['gpu_memory_gb']
        cpu_cores = self.system_info['cpu_physical_cores']
        
        settings = {
            "buffer_size": 30,
            "cache_size": 10,
            "clear_cache_interval": 10,
            "use_threading": True,
            "thread_workers": 2,
            "memory_threshold_mb": 1000,
            "use_cuda_streams": False,
            "prefetch_frames": 5
        }
        
        if profile == "high_end":
            settings.update({
                "buffer_size": 120,
                "cache_size": 50,
                "clear_cache_interval": 25,
                "thread_workers": min(8, cpu_cores),
                "memory_threshold_mb": int(gpu_mem * 600),
                "use_cuda_streams": True,
                "prefetch_frames": 20
            })
        elif profile == "high_performance":
            settings.update({
                "buffer_size": 80,
                "cache_size": 30,
                "clear_cache_interval": 20,
                "thread_workers": min(6, cpu_cores),
                "memory_threshold_mb": int(gpu_mem * 500),
                "use_cuda_streams": True,
                "prefetch_frames": 15
            })
        elif profile == "mid_range":
            settings.update({
                "buffer_size": 50,
                "cache_size": 20,
                "clear_cache_interval": 15,
                "thread_workers": min(4, cpu_cores),
                "memory_threshold_mb": int(gpu_mem * 400),
                "prefetch_frames": 10
            })
        elif profile == "entry_level":
            settings.update({
                "buffer_size": 30,
                "cache_size": 10,
                "clear_cache_interval": 10,
                "thread_workers": 2,
                "memory_threshold_mb": int(gpu_mem * 300),
                "prefetch_frames": 5
            })
        else:  # CPU only
            settings.update({
                "buffer_size": 10,
                "cache_size": 5,
                "clear_cache_interval": 5,
                "thread_workers": min(2, cpu_cores//2),
                "memory_threshold_mb": int(self.system_info['ram_available_gb'] * 300),
                "use_threading": False,  # Simpler processing for CPU
                "prefetch_frames": 2
            })
        
        # Adjust for interpolation multiplier
        multiplier_factor = max(1.0, multiplier / 2.0)
        settings["buffer_size"] = int(settings["buffer_size"] / multiplier_factor)
        settings["cache_size"] = int(settings["cache_size"] / multiplier_factor)
        
        return settings
    
    def benchmark_operation(self, operation_name: str, operation_func, *args, **kwargs) -> float:
        """Benchmark an operation and cache results"""
        cache_key = f"{operation_name}_{hash(str(args) + str(kwargs))}"
        
        if cache_key in self._benchmark_cache:
            return self._benchmark_cache[cache_key]
        
        # Run benchmark
        start_time = time.time()
        try:
            operation_func(*args, **kwargs)
            elapsed = time.time() - start_time
            self._benchmark_cache[cache_key] = elapsed
            return elapsed
        except Exception as e:
            print(f"Benchmark failed for {operation_name}: {e}")
            return float('inf')
    
    def get_memory_usage(self) -> Dict:
        """Get current memory usage stats"""
        stats = {}
        
        if torch.cuda.is_available():
            stats['gpu_used_mb'] = torch.cuda.memory_allocated() / (1024**2)
            stats['gpu_reserved_mb'] = torch.cuda.memory_reserved() / (1024**2)
            stats['gpu_free_mb'] = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / (1024**2)
        
        memory = psutil.virtual_memory()
        stats['ram_used_mb'] = (memory.total - memory.available) / (1024**2)
        stats['ram_available_mb'] = memory.available / (1024**2)
        stats['ram_percent'] = memory.percent
        
        return stats
    
    def should_reduce_batch_size(self, current_batch_size: int) -> bool:
        """Determine if batch size should be reduced based on memory pressure"""
        memory_stats = self.get_memory_usage()
        
        # Check GPU memory pressure
        if torch.cuda.is_available():
            gpu_usage_percent = (memory_stats['gpu_used_mb'] / (self.system_info['gpu_memory_gb'] * 1024)) * 100
            if gpu_usage_percent > 85:  # Over 85% GPU memory usage
                return True
        
        # Check RAM pressure
        if memory_stats['ram_percent'] > 90:  # Over 90% RAM usage
            return True
        
        return False
    
    def adaptive_tile_size(self, image_resolution: Tuple[int, int], current_tile_size: int = 512) -> int:
        """Dynamically adjust tile size based on image resolution and available memory"""
        width, height = image_resolution
        total_pixels = width * height
        memory_stats = self.get_memory_usage()
        
        # Base tile size on available GPU memory
        if torch.cuda.is_available():
            available_memory_mb = memory_stats['gpu_free_mb']
            
            # Larger images can use larger tiles if memory allows
            if total_pixels > 4096*4096 and available_memory_mb > 4000:
                return min(1024, current_tile_size * 2)
            elif total_pixels > 2048*2048 and available_memory_mb > 2000:
                return min(768, int(current_tile_size * 1.5))
            elif available_memory_mb < 500:
                return max(256, current_tile_size // 2)
        
        return current_tile_size
    
    def get_system_info_summary(self) -> str:
        """Get human-readable system info summary"""
        info = self.system_info
        summary = f"Performance Profile: {self.performance_profile.upper()}\n"
        
        if info['gpu_count'] > 0:
            summary += f"GPU: {info['gpu_name']} ({info['gpu_memory_gb']:.1f}GB)\n"
        else:
            summary += "GPU: None (CPU-only mode)\n"
            
        summary += f"CPU: {info['cpu_physical_cores']} cores ({info['cpu_count']} threads)\n"
        summary += f"RAM: {info['ram_total_gb']:.1f}GB total, {info['ram_available_gb']:.1f}GB available\n"
        summary += f"Platform: {info['platform']}"
        
        if info['is_wsl']:
            summary += " (WSL)"
            
        return summary

# Global instance for easy access
PERF_OPTIMIZER = PerformanceOptimizer()