
import time
from collections import deque
import heapq
import copy


class Process:
    def __init__(self, pid, burst_time, arrival_time=0, priority=0):
        self.pid = pid
        self.burst_time = burst_time
        self.arrival_time = arrival_time
        self.priority = priority
        self.remaining_time = burst_time
        self.waiting_time = 0
        self.turnaround_time = 0

    def __repr__(self):
        return f"P{self.pid}(BT={self.burst_time}, AT={self.arrival_time}, PR={self.priority})"


def calculate_metrics(processes):
    n = len(processes)
    avg_wt = sum(p.waiting_time for p in processes) / n
    avg_tat = sum(p.turnaround_time for p in processes) / n
    return avg_wt, avg_tat


def print_results(name, processes):
    print(f"\n{name} Scheduling Results:")
    print("PID\tBT\tWT\tTAT")
    for p in processes:
        print(f"{p.pid}\t{p.burst_time}\t{p.waiting_time}\t{p.turnaround_time}")
    avg_wt, avg_tat = calculate_metrics(processes)
    print(f"Average Waiting Time: {avg_wt:.2f}")
    print(f"Average Turnaround Time: {avg_tat:.2f}")


# -------------------- SRTF (Preemptive) with time delay --------------------
def srtf(processes, delay=0.5):
    time_unit = 0
    ready = []
    completed = []
    processes = sorted(processes, key=lambda x: x.arrival_time)

    print("\nSRTF Execution Timeline:")
    while processes or ready:
        while processes and processes[0].arrival_time <= time_unit:
            heapq.heappush(ready, (processes[0].remaining_time, processes[0].arrival_time, processes.pop(0)))

        if ready:
            rt, at, p = heapq.heappop(ready)
            print(f"t={time_unit}: Running P{p.pid} (Remaining={p.remaining_time})")
            time.sleep(delay)  # simulate CPU execution
            p.remaining_time -= 1
            time_unit += 1
            if p.remaining_time == 0:
                p.turnaround_time = time_unit - p.arrival_time
                p.waiting_time = p.turnaround_time - p.burst_time
                completed.append(p)
                print(f"t={time_unit}: P{p.pid} finished")
            else:
                heapq.heappush(ready, (p.remaining_time, p.arrival_time, p))
        else:
            print(f"t={time_unit}: CPU idle")
            time.sleep(delay)
            time_unit += 1

    print_results("SRTF", completed)


# -------------------- SJF (Non-preemptive) with timeline --------------------
def sjf(processes, delay=0.5):
    time_unit = 0
    completed = []
    processes = sorted(processes, key=lambda x: (x.arrival_time, x.burst_time))
    ready = []

    print("\nSJF Execution Timeline:")
    while processes or ready:
        while processes and processes[0].arrival_time <= time_unit:
            heapq.heappush(ready, (processes[0].burst_time, processes.pop(0)))
        if ready:
            bt, p = heapq.heappop(ready)
            print(f"t={time_unit}: Running P{p.pid} for {p.burst_time} units")
            time.sleep(delay)
            p.waiting_time = time_unit - p.arrival_time
            time_unit += p.burst_time
            p.turnaround_time = p.waiting_time + p.burst_time
            completed.append(p)
            print(f"t={time_unit}: P{p.pid} finished")
        else:
            print(f"t={time_unit}: CPU idle")
            time.sleep(delay)
            time_unit += 1

    print_results("SJF", completed)


if __name__ == "__main__":
    # Example dataset to SHOW difference
    demo_processes = [
        Process(1, 8, 0),  # Long process arrives first
        Process(2, 4, 1),  # Shorter process arrives later
        Process(3, 2, 2)  # Even shorter, arrives later
    ]

    # Run both
    sjf(copy.deepcopy(demo_processes), delay=0.3)
    srtf(copy.deepcopy(demo_processes), delay=0.3)
