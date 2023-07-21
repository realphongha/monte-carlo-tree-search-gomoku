import numpy as np


class PerfMonitorMixin:
    def update_perf(self, name, latency):
        if not hasattr(self, 'perf_monitor'):
            self.perf_monitor = {}
        if name not in self.perf_monitor:
            self.perf_monitor[name] = []
        self.perf_monitor[name].append(latency)

    def get_perf(self, name, display=True):
        if not hasattr(self, 'perf_monitor'):
            if display:
                print("Performance monitor is not initialized!")
            return -1, -1
        if name not in self.perf_monitor:
            if display:
                print("%s is not in performance monitor!" % name)
            return -1, -1
        latency = np.mean(self.perf_monitor[name])
        fps = 1/latency
        if display:
            print("PERFORMANCE '%s': FPS - %.8f, Latency: %.8f" % \
                (name, fps, latency))
        return fps, latency
