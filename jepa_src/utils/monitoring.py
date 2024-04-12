# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import dataclasses
import threading
from typing import Dict, Tuple

import psutil


@dataclasses.dataclass
class ResourceStatsSample:
    timestamp: float
    cpu_percent: float
    read_count: int
    write_count: int
    read_bytes: int
    write_bytes: int
    read_chars: int
    write_chars: int
    cpu_times_user: float
    cpu_times_system: float
    cpu_times_children_user: float
    cpu_times_children_system: float
    cpu_times_iowait: float
    cpu_affinity: str
    cpu_num: int
    num_threads: int
    num_voluntary_ctx_switches: int
    num_involuntary_ctx_switches: int

    def as_tuple(self) -> Dict:
        """Return values mirroring fields."""
        return dataclasses.astuple(self)

    def fields(self) -> Tuple[dataclasses.Field, ...]:
        """Return fields in this dataclass."""
        return dataclasses.fields(self.__class__)


class ResourceMonitoringThread(threading.Thread):
    def __init__(self, pid=None, refresh_interval=None, stats_callback_fn=None):
        """Starts a thread to monitor pid every refresh_interval seconds.

        Passes a ResourceStatsSample object to the callback."""
        super(ResourceMonitoringThread, self).__init__()
        if refresh_interval is None:
            refresh_interval = 5
        self.is_running_event = threading.Event()
        self.p = psutil.Process(pid)
        self.refresh_interval = refresh_interval
        if stats_callback_fn is None:
            # Default callback
            def stats_callback_fn(resource_sample: ResourceStatsSample):
                print(
                    f"PID {self.p.pid} Stats: {resource_sample.resource_stats}")
        elif not callable(stats_callback_fn):
            raise ValueError("Callback needs to be callable, got {}".format(
                type(stats_callback_fn)))
        self.stats_callback_fn = stats_callback_fn

    def stop(self) -> None:
        self.is_running_event.set()

    def run(self) -> None:
        while not self.is_running_event.is_set():
            self.sample_counters()
            self.is_running_event.wait(self.refresh_interval)

    def log_sample(self, resource_sample: ResourceStatsSample) -> None:
        self.stats_callback_fn(resource_sample)

    def sample_counters(self) -> None:
        if not self.p.is_running():
            self.stop()
            return

        with self.p.oneshot():
            cpu_percent = self.p.cpu_percent()
            cpu_times = self.p.cpu_times()
            io_counters = self.p.io_counters()
            cpu_affinity = self.p.cpu_affinity()
            cpu_num = self.p.cpu_num()
            num_threads = self.p.num_threads()
            num_ctx_switches = self.p.num_ctx_switches()
        timestamp = time.time()

        read_count = io_counters.read_count
        write_count = io_counters.write_count
        read_bytes = io_counters.read_bytes
        write_bytes = io_counters.write_bytes
        read_chars = io_counters.read_chars
        write_chars = io_counters.write_chars

        def compress_cpu_affinity(cpu_affinity):
            """Change list representation to interval/range representation."""
            if not cpu_affinity:
                return ""
            cpu_affinity_compressed = []
            min_x = None
            max_x = None
            last_x = None

            # Find contiguous ranges
            for x in cpu_affinity:
                if last_x is None:
                    # Start interval
                    min_x = x
                    max_x = x
                    last_x = x
                    continue
                elif x == (last_x + 1):
                    # Move interval up
                    max_x = x
                elif max_x is not None:
                    # Interval ended, start again
                    if min_x == max_x:
                        cpu_affinity_compressed.append("{}".format(min_x))
                    else:
                        cpu_affinity_compressed.append(
                            "{}-{}".format(min_x, max_x))
                    min_x = x
                    max_x = x
                last_x = x
            # Terminate last range
            if max_x is not None:
                if min_x == max_x:
                    cpu_affinity_compressed.append("{}".format(min_x))
                else:
                    cpu_affinity_compressed.append(
                        "{}-{}".format(min_x, max_x))

            # Concat
            cpu_affinity_compressed = ",".join(cpu_affinity_compressed)

            return cpu_affinity_compressed

        cpu_affinity = compress_cpu_affinity(cpu_affinity)

        resource_sample = ResourceStatsSample(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            read_count=read_count,
            write_count=write_count,
            read_bytes=read_bytes,
            write_bytes=write_bytes,
            read_chars=read_chars,
            write_chars=write_chars,
            cpu_times_user=cpu_times.user,
            cpu_times_system=cpu_times.system,
            cpu_times_children_user=cpu_times.children_user,
            cpu_times_children_system=cpu_times.children_system,
            cpu_times_iowait=cpu_times.iowait,
            cpu_affinity=cpu_affinity,
            cpu_num=cpu_num,
            num_threads=num_threads,
            num_voluntary_ctx_switches=num_ctx_switches.voluntary,
            num_involuntary_ctx_switches=num_ctx_switches.involuntary,
        )
        self.log_sample(resource_sample)


if __name__ == "__main__":
    import multiprocessing
    import time
    pid = multiprocessing.current_process().pid
    monitor_thread = ResourceMonitoringThread(pid, 1)
    monitor_thread.start()
    time.sleep(5)
    print("Shutdown")
    monitor_thread.stop()
