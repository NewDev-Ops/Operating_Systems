from collections import deque
import heapq

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


def fcfs(processes):
    time = 0
    for p in processes:
        if time < p.arrival_time:
            time = p.arrival_time
        p.waiting_time = time - p.arrival_time
        time += p.burst_time
        p.turnaround_time = p.waiting_time + p.burst_time
    print_results("FCFS", processes)


def sjf(processes):
    time = 0
    completed = []
    processes = sorted(processes, key=lambda x: (x.arrival_time, x.burst_time))
    ready = []

    while processes or ready:
        while processes and processes[0].arrival_time <= time:
            heapq.heappush(ready, (processes[0].burst_time, processes.pop(0)))
        if ready:
            bt, p = heapq.heappop(ready)
            p.waiting_time = time - p.arrival_time
            time += p.burst_time
            p.turnaround_time = p.waiting_time + p.burst_time
            completed.append(p)
        else:
            time += 1

    print_results("SJF", completed)


def srtf(processes):
    time = 0
    ready = []
    completed = []
    processes = sorted(processes, key=lambda x: x.arrival_time)

    while processes or ready:
        while processes and processes[0].arrival_time <= time:
            heapq.heappush(ready, (processes[0].remaining_time, processes[0].arrival_time, processes.pop(0)))
        if ready:
            rt, at, p = heapq.heappop(ready)
            p.remaining_time -= 1
            time += 1
            if p.remaining_time == 0:
                p.turnaround_time = time - p.arrival_time
                p.waiting_time = p.turnaround_time - p.burst_time
                completed.append(p)
            else:
                heapq.heappush(ready, (p.remaining_time, p.arrival_time, p))
        else:
            time += 1

    print_results("SRTF", completed)


def round_robin(processes, quantum=2):
    time = 0
    queue = deque(sorted(processes, key=lambda x: x.arrival_time))
    completed = []

    while queue:
        p = queue.popleft()
        if p.remaining_time > quantum:
            if time < p.arrival_time:
                time = p.arrival_time
            p.remaining_time -= quantum
            time += quantum
            while queue and queue[0].arrival_time <= time:
                queue.append(queue.popleft())
            queue.append(p)
        else:
            if time < p.arrival_time:
                time = p.arrival_time
            time += p.remaining_time
            p.turnaround_time = time - p.arrival_time
            p.waiting_time = p.turnaround_time - p.burst_time
            p.remaining_time = 0
            completed.append(p)

    print_results("Round Robin", completed)


def priority_scheduling(processes):
    time = 0
    ready = []
    completed = []
    processes = sorted(processes, key=lambda x: x.arrival_time)

    while processes or ready:
        while processes and processes[0].arrival_time <= time:
            heapq.heappush(ready, (processes[0].priority, processes[0].arrival_time, processes.pop(0)))
        if ready:
            pr, at, p = heapq.heappop(ready)
            if time < p.arrival_time:
                time = p.arrival_time
            p.waiting_time = time - p.arrival_time
            time += p.burst_time
            p.turnaround_time = p.waiting_time + p.burst_time
            completed.append(p)
        else:
            time += 1

    print_results("Priority", completed)


if __name__ == "__main__":
    sample_processes = [
        Process(1, 6, 0, 2),
        Process(2, 8, 1, 1),
        Process(3, 7, 2, 3),
        Process(4, 3, 3, 2)
    ]

    # Make deep copies for each algorithm
    import copy
    fcfs(copy.deepcopy(sample_processes))
    sjf(copy.deepcopy(sample_processes))
    srtf(copy.deepcopy(sample_processes))
    round_robin(copy.deepcopy(sample_processes), quantum=3)
    priority_scheduling(copy.deepcopy(sample_processes))
