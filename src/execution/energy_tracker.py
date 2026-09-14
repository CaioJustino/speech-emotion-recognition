"""
    Module energy_tracker: Estimates energy consumption and CO2 emissions during training. Based on the methodology from Strubell et al. (2019): "Energy and Policy Considerations for Deep Learning in NLP".
"""

import time
import threading
import subprocess
from typing import Tuple


class EnergyTracker:
    """
    Class EnergyTracker: Tracks energy consumption using nvidia-smi for GPUs and TDP estimates for CPU/DRAM.
    """
    
    def __init__(self, cpu_tdp_w: float = 100.0, dram_tdp_w: float = 50.0):
        self.cpu_tdp_w = cpu_tdp_w
        self.dram_tdp_w = dram_tdp_w
        self.running = False
        self.thread = None
        self.gpu_power_samples = []
        self.start_time = None
        self.end_time = None

    def _sample_power(self) -> None:
        """
    Function _sample_power: Samples power consumption from GPUs.
        """
        while self.running:
            try:
                # Query GPU power draw in Watts
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=True
                )
                # Parse output and sum over all available GPUs
                gpu_powers = [float(x) for x in result.stdout.strip().split('\n') if x]
                if gpu_powers:
                    self.gpu_power_samples.append(sum(gpu_powers))
            except Exception:
                pass
            time.sleep(1.0)

    def start(self) -> None:
        """
    Function start: Starts the background power sampling thread.
        """
        self.running = True
        self.start_time = time.time()
        self.gpu_power_samples = []
        self.thread = threading.Thread(target=self._sample_power, daemon=True)
        self.thread.start()

    def stop(self) -> Tuple[float, float]:
        """
    Function stop: Stops sampling and calculates energy (kWh) and CO2 emissions (lbs).
    
    Returns:
        - Tuple[float, float] (Energy in kWh, CO2e in lbs).
        """
        self.running = False
        if self.thread:
            self.thread.join()
        self.end_time = time.time()
        
        t_hours = (self.end_time - self.start_time) / 3600.0
        
        avg_gpu_power = 0.0
        if self.gpu_power_samples:
            avg_gpu_power = sum(self.gpu_power_samples) / len(self.gpu_power_samples)
            
        # Calculation from Strubell et al. 2019:
        # pt = 1.58 * t * (pc + pr + g*pg) / 1000
        # Note: avg_gpu_power already represents g*pg (sum of all GPUs)
        pt_kwh = 1.58 * t_hours * (self.cpu_tdp_w + self.dram_tdp_w + avg_gpu_power) / 1000.0
        
        # CO2e = 0.954 * pt
        co2e_lbs = 0.954 * pt_kwh
        
        return pt_kwh, co2e_lbs
