import numpy as np


class PerfMonitorMixin:
    PERF_MONITOR = {}

    def update_perf(self, name, latency, ignore_classwise=False):
        if not ignore_classwise:
            name = self.__class__.__name__ + "_" + name
        if name not in PerfMonitorMixin.PERF_MONITOR:
            PerfMonitorMixin.PERF_MONITOR[name] = []
        PerfMonitorMixin.PERF_MONITOR[name].append(latency)

    @staticmethod
    def update_perf_static(name, latency):
        if name not in PerfMonitorMixin.PERF_MONITOR:
            PerfMonitorMixin.PERF_MONITOR[name] = []
        PerfMonitorMixin.PERF_MONITOR[name].append(latency)

    def get_perf_for_this_class(self, name, display=True):
        name = self.__class__.__name__ + "_" + name
        return PerfMonitorMixin.get_perf(name, display)

    def get_all_perfs_for_this_class(self, display=True):
        class_name = self.__class__.__name__ + "_"
        all_perfs = {}
        perf_names = list(PerfMonitorMixin.PERF_MONITOR.keys())
        # Higest latency goes first
        perf_names.sort(key=lambda k:
                        -np.mean(PerfMonitorMixin.PERF_MONITOR[k]))
        for name in perf_names:
            if name.startswith(class_name):
                all_perfs[name] = PerfMonitorMixin.get_perf(name, display)
        return all_perfs

    @staticmethod
    def get_perf(name, display=True):
        if name not in PerfMonitorMixin.PERF_MONITOR:
            if display:
                print(f"{name} is not in performance monitor!")
            return -1, -1
        latency = np.mean(PerfMonitorMixin.PERF_MONITOR[name])
        total_time = sum(PerfMonitorMixin.PERF_MONITOR[name])
        fps = 1 / latency
        if display:
            print("SPEED PERFORMANCE for '%s':" % name)
            print("FPS - %10.2f, Latency - %15.8f, Total time - %15.8f" %
                (fps, latency, total_time))
        return fps, latency

    @staticmethod
    def get_all_perfs(display=True):
        all_perfs = {}
        perf_names = list(PerfMonitorMixin.PERF_MONITOR.keys())
        # Higest latency goes first
        perf_names.sort(key=lambda k:
                        -np.mean(PerfMonitorMixin.PERF_MONITOR[k]))
        for name in perf_names:
            all_perfs[name] = PerfMonitorMixin.get_perf(name, display)
        return all_perfs

