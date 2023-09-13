import os
import tracemalloc
import psutil

from pytorch_lightning import Callback


class MemoryProfilingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        tracemalloc.start()

    def on_train_end(self, trainer, pl_module):
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        print("[ Top 10 ]")
        for index, stat in enumerate(top_stats[:10], 1):
            print(f"#{index}: {stat}")

        tracemalloc.stop()
        

class ProcessMemoryMonitorCallback(Callback):
    
    def on_train_end(self, trainer, pl_module):
        main_process_id = os.getpid()
        print(f"Memory consumption by child processes of main process (PID={main_process_id}) after training:")
        
        for proc in psutil.process_iter():
            try:
                # Fetch process details
                p_info = proc.as_dict(attrs=['pid', 'ppid', 'name', 'memory_percent'])
                
                # Check if this process is a child of the main process
                if p_info['ppid'] == main_process_id:
                    # Extract the memory percentage information
                    mem_percent = p_info['memory_percent']
                    # Print
                    print(f"PID={p_info['pid']}, Name={p_info['name']}, Memory Percent={mem_percent}%")
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass