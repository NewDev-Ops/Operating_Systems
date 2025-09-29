# Enhanced CPU Scheduling Simulator: FCFS, SJF, SRTF, Priority, and Round Robin
# Features: Individual Gantt charts + Comprehensive comparison visualizations
# All modules combined into one file for easy execution.

# --- 1. CORE IMPORTS ---
import random
import os
import sys
import math
from typing import List, Dict, Any, Union, Tuple, Callable

# Note: These external libraries are required for data manipulation and visualization.
# Please ensure they are installed: pip install pandas numpy matplotlib seaborn
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import Normalize
    from matplotlib.cm import get_cmap

    plt.style.use('seaborn-v0_8')  # Better default styling
except ImportError:
    print("Error: Required libraries (pandas, numpy, matplotlib, seaborn) not found.")
    print("Please install them using: pip install pandas numpy matplotlib seaborn")
    sys.exit(1)


# -----------------------------------------------------------------------------
# --- 2. UTILITY FUNCTIONS ---
# -----------------------------------------------------------------------------

def ensureDirectoryExists(path: str) -> str:
    """
    Checks if a directory exists, and creates it if it doesn't.
    """
    try:
        os.makedirs(path, exist_ok=True)
        return os.path.abspath(path)
    except OSError as e:
        print(f"Error creating directory {path}: {e}", file=sys.stderr)
        sys.exit(1)


def safeInput(prompt: str, targetType: type, validOptions: Union[List[str], None] = None) -> Any:
    """
    Prompts the user for input and validates its type and optional value.
    """
    while True:
        try:
            userInput = input(prompt).strip()

            if not userInput and targetType != str:
                raise ValueError("Input cannot be empty.")

            if targetType == int:
                value = int(userInput)
                if value < 0:
                    raise ValueError("Input must be a non-negative integer.")
                return value

            elif targetType == str:
                value = userInput.lower()
                if validOptions and value not in validOptions:
                    options_str = ', '.join(validOptions)
                    print(f"Invalid option. Must be one of: {options_str}")
                    continue
                return value

            # General case for float or other types
            return targetType(userInput)

        except ValueError as e:
            print(f"Invalid input: {e}. Please try again.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}. Please try again.")


# -----------------------------------------------------------------------------
# --- 3. PROCESS CLASS ---
# -----------------------------------------------------------------------------

class Process:
    """
    Represents a single process with essential metadata for simulation.
    """

    def __init__(self, processId: str, arrivalTime: int, burstTime: int, priority: int):
        self.processId = processId
        self.arrivalTime = arrivalTime
        self.originalBurstTime = burstTime
        self.burstTimeRemaining = burstTime
        self.priority = priority  # Lower number means higher priority

        # Metrics
        self.startTime = -1  # Time the process first starts execution
        self.finishTime = -1  # Time the process completes
        self.turnaroundTime = 0
        self.waitingTime = 0
        self.responseTime = 0
        self.lastExecutedTime = arrivalTime  # Used for calculating waiting time in preemptive
        self.executionHistory = []  # Stores (start_time, end_time) tuples for Gantt

    def __repr__(self) -> str:
        return (f"Process(ID={self.processId}, Arrival={self.arrivalTime}, "
                f"Burst={self.originalBurstTime}, Priority={self.priority}, Rem={self.burstTimeRemaining})")

    @property
    def isCompleted(self) -> bool:
        return self.burstTimeRemaining <= 0

    def calculateFinalMetrics(self):
        """Calculates final T.A.T, Waiting, and Response times."""
        if self.finishTime != -1:
            self.turnaroundTime = self.finishTime - self.arrivalTime
            # Waiting Time = Turnaround Time - Original Burst Time
            self.waitingTime = self.turnaroundTime - self.originalBurstTime
            # Response Time = Start Time - Arrival Time
            self.responseTime = self.startTime - self.arrivalTime
        else:
            # Should not happen in a complete simulation
            self.turnaroundTime = 0
            self.waitingTime = 0
            self.responseTime = 0


# -----------------------------------------------------------------------------
# --- 4. DATA GENERATOR ---
# -----------------------------------------------------------------------------

def generateProcesses(N: int) -> List[Process]:
    """
    Generates N random processes with controlled parameters.
    """
    random.seed(42)  # Ensure reproducible results

    processes = []

    # Process attributes ranges
    MAX_ARRIVAL = N * 2
    MAX_BURST = 20
    MAX_PRIORITY = 10

    # Generate random processes
    for i in range(1, N + 1):
        processId = f"P{i:02d}"
        # Arrival times increase slowly to avoid huge gaps
        arrivalTime = random.randint(0, MAX_ARRIVAL)
        burstTime = random.randint(1, MAX_BURST)
        # Priority: 1 (highest) to MAX_PRIORITY (lowest)
        priority = random.randint(1, MAX_PRIORITY)

        processes.append(Process(processId, arrivalTime, burstTime, priority))

    # Sort processes by arrival time to ensure a proper start
    processes.sort(key=lambda p: p.arrivalTime)

    return processes


# -----------------------------------------------------------------------------
# --- 5. SCHEDULING ALGORITHMS ---
# -----------------------------------------------------------------------------

def runScheduling(
        initial_processes: List[Process],
        algorithm_func: Callable[[List[Process], int], Tuple[List[Process], List[Tuple[int, str, int]]]],
        time_quantum: int = 0
) -> Tuple[List[Process], List[Tuple[int, str, int]]]:
    """
    Prepares copies of processes and runs the specified scheduling algorithm.
    Returns: (completed_processes, gantt_chart_data)
    """
    # Create deep copies of processes for the simulation to avoid side effects
    processes_copy = [
        Process(p.processId, p.arrivalTime, p.originalBurstTime, p.priority)
        for p in initial_processes
    ]

    # Check if the algorithm is Round Robin and requires a quantum
    if algorithm_func.__name__ == 'roundRobin' and time_quantum <= 0:
        raise ValueError("Round Robin algorithm requires a time_quantum > 0.")

    return algorithm_func(processes_copy, time_quantum)


def fcfs(processes: List[Process], time_quantum: int = 0) -> Tuple[List[Process], List[Tuple[int, str, int]]]:
    """First Come First Served (Non-Preemptive)"""
    # FCFS processes are already sorted by arrival time from the generator.
    current_time = 0
    completed_processes: List[Process] = []
    gantt_chart: List[Tuple[int, str, int]] = []  # (start_time, process_id, duration)

    # Wait until the first process arrives
    if processes:
        current_time = processes[0].arrivalTime

    for p in processes:
        # If CPU is idle, advance time to the process's arrival time
        if current_time < p.arrivalTime:
            gantt_chart.append((current_time, "Idle", p.arrivalTime - current_time))
            current_time = p.arrivalTime

        # Non-Preemptive: Run until completion
        p.startTime = current_time
        duration = p.originalBurstTime
        p.finishTime = current_time + duration
        p.burstTimeRemaining = 0

        gantt_chart.append((current_time, p.processId, duration))
        p.executionHistory.append((current_time, p.finishTime))

        current_time = p.finishTime
        completed_processes.append(p)

    return completed_processes, gantt_chart


def sjf(processes: List[Process], time_quantum: int = 0) -> Tuple[List[Process], List[Tuple[int, str, int]]]:
    """Shortest Job First (Non-Preemptive)"""
    processes.sort(key=lambda p: p.arrivalTime)  # Ensure initial order is by arrival
    n = len(processes)
    current_time = 0
    ready_queue: List[Process] = []
    completed_processes: List[Process] = []
    gantt_chart: List[Tuple[int, str, int]] = []

    process_index = 0

    while len(completed_processes) < n:
        # 1. Move arrived processes to the ready queue
        while process_index < n and processes[process_index].arrivalTime <= current_time:
            ready_queue.append(processes[process_index])
            process_index += 1

        # 2. Select next process
        if ready_queue:
            # Sort by remaining burst time (SJF)
            ready_queue.sort(key=lambda p: p.burstTimeRemaining)

            p = ready_queue.pop(0)

            # Execute (Non-Preemptive)
            if p.startTime == -1:
                p.startTime = current_time

            duration = p.originalBurstTime
            p.finishTime = current_time + duration
            p.burstTimeRemaining = 0

            gantt_chart.append((current_time, p.processId, duration))
            p.executionHistory.append((current_time, p.finishTime))

            current_time = p.finishTime
            completed_processes.append(p)

        else:
            # CPU is idle. Advance time to the next arrival.
            if process_index < n:
                next_arrival = processes[process_index].arrivalTime
                gantt_chart.append((current_time, "Idle", next_arrival - current_time))
                current_time = next_arrival
            else:
                # All processes are completed or in the queue (shouldn't happen here)
                break

    return completed_processes, gantt_chart


def srtf(processes: List[Process], time_quantum: int = 0) -> Tuple[List[Process], List[Tuple[int, str, int]]]:
    """Shortest Remaining Time First (Preemptive SJF)"""
    processes.sort(key=lambda p: p.arrivalTime)
    n = len(processes)
    current_time = 0
    ready_queue: List[Process] = []
    completed_processes: List[Process] = []
    gantt_chart: List[Tuple[int, str, int]] = []

    process_index = 0
    running_process: Union[Process, None] = None

    while len(completed_processes) < n:
        # 1. Move arrived processes to the ready queue
        while process_index < n and processes[process_index].arrivalTime <= current_time:
            ready_queue.append(processes[process_index])
            process_index += 1

        # 2. Add running process back to ready queue if preempted
        if running_process and not running_process.isCompleted:
            ready_queue.append(running_process)
            running_process = None

        # 3. Select next process from ready queue
        if ready_queue:
            # Find process with minimum remaining burst time. Break ties with arrival time.
            ready_queue.sort(key=lambda p: (p.burstTimeRemaining, p.arrivalTime))
            next_process = ready_queue.pop(0)
        else:
            next_process = None

        # 4. Check for preemption or idle time
        if next_process:
            # Check if preemption occurred
            preempted = running_process is not None and next_process != running_process

            running_process = next_process

            if running_process.startTime == -1:
                running_process.startTime = current_time
                running_process.lastExecutedTime = current_time
            else:
                # Update waiting time for the time it spent in the queue
                running_process.waitingTime += (current_time - running_process.lastExecutedTime)

            # Determine execution time (run for 1 time unit)
            execution_time = 1
            start_segment_time = current_time

            running_process.burstTimeRemaining -= execution_time
            current_time += execution_time

            running_process.lastExecutedTime = current_time

            # Add segment to Gantt chart
            if gantt_chart and gantt_chart[-1][1] == running_process.processId and not preempted:
                # Coalesce execution segment if the same process ran consecutively
                prev_start, prev_id, prev_duration = gantt_chart.pop()
                new_duration = prev_duration + execution_time
                gantt_chart.append((prev_start, running_process.processId, new_duration))
            else:
                gantt_chart.append((start_segment_time, running_process.processId, execution_time))

            # 5. Check for completion
            if running_process.isCompleted:
                running_process.finishTime = current_time
                completed_processes.append(running_process)
                running_process = None

        else:
            # CPU is idle. Advance time to the next arrival.
            if process_index < n:
                next_arrival = processes[process_index].arrivalTime
                gantt_chart.append((current_time, "Idle", next_arrival - current_time))
                current_time = next_arrival
            else:
                # All processes finished and no more arrivals
                break

    return completed_processes, gantt_chart


def priorityScheduling(processes: List[Process], time_quantum: int = 0) -> Tuple[
    List[Process], List[Tuple[int, str, int]]]:
    """Priority Scheduling (Preemptive) - Lower number means higher priority."""
    processes.sort(key=lambda p: p.arrivalTime)
    n = len(processes)
    current_time = 0
    ready_queue: List[Process] = []
    completed_processes: List[Process] = []
    gantt_chart: List[Tuple[int, str, int]] = []

    process_index = 0
    running_process: Union[Process, None] = None

    while len(completed_processes) < n:
        # 1. Move arrived processes to the ready queue
        while process_index < n and processes[process_index].arrivalTime <= current_time:
            ready_queue.append(processes[process_index])
            process_index += 1

        # 2. Add running process back to ready queue if preempted
        if running_process and not running_process.isCompleted:
            ready_queue.append(running_process)
            running_process = None

        # 3. Select next process from ready queue
        if ready_queue:
            # Find process with minimum priority (highest priority). Break ties with arrival time.
            ready_queue.sort(key=lambda p: (p.priority, p.arrivalTime))
            next_process = ready_queue.pop(0)
        else:
            next_process = None

        # 4. Check for preemption or idle time
        if next_process:
            preempted = running_process is not None and next_process != running_process

            running_process = next_process

            if running_process.startTime == -1:
                running_process.startTime = current_time
                running_process.lastExecutedTime = current_time
            else:
                # Update waiting time for the time it spent in the queue
                running_process.waitingTime += (current_time - running_process.lastExecutedTime)

            # Determine execution time (run for 1 time unit)
            execution_time = 1
            start_segment_time = current_time

            running_process.burstTimeRemaining -= execution_time
            current_time += execution_time

            running_process.lastExecutedTime = current_time

            # Add segment to Gantt chart
            if gantt_chart and gantt_chart[-1][1] == running_process.processId and not preempted:
                # Coalesce execution segment if the same process ran consecutively
                prev_start, prev_id, prev_duration = gantt_chart.pop()
                new_duration = prev_duration + execution_time
                gantt_chart.append((prev_start, running_process.processId, new_duration))
            else:
                gantt_chart.append((start_segment_time, running_process.processId, execution_time))

            # 5. Check for completion
            if running_process.isCompleted:
                running_process.finishTime = current_time
                completed_processes.append(running_process)
                running_process = None

        else:
            # CPU is idle. Advance time to the next arrival.
            if process_index < n:
                next_arrival = processes[process_index].arrivalTime
                gantt_chart.append((current_time, "Idle", next_arrival - current_time))
                current_time = next_arrival
            else:
                # All processes finished and no more arrivals
                break

    return completed_processes, gantt_chart


def roundRobin(processes: List[Process], time_quantum: int) -> Tuple[List[Process], List[Tuple[int, str, int]]]:
    """Round Robin Scheduling (Preemptive)"""
    processes.sort(key=lambda p: p.arrivalTime)
    n = len(processes)
    current_time = 0
    ready_queue: List[Process] = []
    completed_processes: List[Process] = []
    gantt_chart: List[Tuple[int, str, int]] = []

    process_index = 0  # Index for processes yet to arrive

    # Initialize the queue by advancing to the first arrival time
    if processes:
        current_time = processes[0].arrivalTime

    # Move initial arrived processes to the ready queue
    while process_index < n and processes[process_index].arrivalTime <= current_time:
        ready_queue.append(processes[process_index])
        process_index += 1

    while len(completed_processes) < n:

        if ready_queue:
            p = ready_queue.pop(0)

            # --- Execute process ---
            if p.startTime == -1:
                p.startTime = current_time
                p.lastExecutedTime = current_time
            else:
                # Update waiting time for the time it spent in the queue
                p.waitingTime += (current_time - p.lastExecutedTime)

            # Determine execution time (min of remaining burst or time quantum)
            execute_duration = min(p.burstTimeRemaining, time_quantum)

            start_segment_time = current_time
            current_time += execute_duration
            p.burstTimeRemaining -= execute_duration
            p.lastExecutedTime = current_time

            # Add segment to Gantt chart
            gantt_chart.append((start_segment_time, p.processId, execute_duration))

            # --- Check for new arrivals during this execution segment ---
            arrivals_during_execution = []
            while process_index < n and processes[process_index].arrivalTime <= current_time:
                arrivals_during_execution.append(processes[process_index])
                process_index += 1

            # Add newly arrived processes to the queue before the currently running process
            ready_queue.extend(arrivals_during_execution)

            # --- Check for completion ---
            if p.isCompleted:
                p.finishTime = current_time
                completed_processes.append(p)
            else:
                # Process is not finished, put it back at the end of the ready queue
                ready_queue.append(p)

        else:
            # CPU is idle. Advance time to the next arrival.
            if process_index < n:
                next_arrival = processes[process_index].arrivalTime
                gantt_chart.append((current_time, "Idle", next_arrival - current_time))
                current_time = next_arrival

                # Move arrived processes to the ready queue immediately
                while process_index < n and processes[process_index].arrivalTime <= current_time:
                    ready_queue.append(processes[process_index])
                    process_index += 1
            else:
                # All processes finished and no more arrivals
                break

    return completed_processes, gantt_chart


# -----------------------------------------------------------------------------
# --- 6. ENHANCED VISUALIZER & METRICS ---
# -----------------------------------------------------------------------------

def calculateMetrics(completed_processes: List[Process], algorithm_name: str) -> Dict[str, Any]:
    """Calculates and returns key performance metrics."""
    if not completed_processes:
        return {"Algorithm": algorithm_name, "Avg_TAT": 0, "Avg_Wait": 0, "Avg_Response": 0, "Throughput": 0}

    # Calculate individual metrics
    for p in completed_processes:
        p.calculateFinalMetrics()

    total_tat = sum(p.turnaroundTime for p in completed_processes)
    total_wait = sum(p.waitingTime for p in completed_processes)
    total_response = sum(p.responseTime for p in completed_processes)

    n = len(completed_processes)

    avg_tat = total_tat / n
    avg_wait = total_wait / n
    avg_response = total_response / n

    # Throughput: Total number of processes completed per unit of time
    # Max completion time is the last finish time
    max_finish_time = max(p.finishTime for p in completed_processes)
    throughput = n / max_finish_time if max_finish_time > 0 else 0

    return {
        "Algorithm": algorithm_name,
        "Avg_TAT": round(avg_tat, 2),
        "Avg_Wait": round(avg_wait, 2),
        "Avg_Response": round(avg_response, 2),
        "Throughput": round(throughput, 4),
        "Max_Finish_Time": max_finish_time,
        "Raw_Processes": completed_processes  # Keep processes for detail table
    }


def generateGanttChart(gantt_chart_data: List[Tuple[int, str, int]], algorithm_name: str, output_path: str):
    """
    Generates and saves a Gantt chart visualization.
    gantt_chart_data is List[(start_time, process_id, duration)]
    """
    if not gantt_chart_data:
        print(f"No Gantt data for {algorithm_name}. Skipping chart generation.")
        return

    # Prepare data for plotting
    processes = [item[1] for item in gantt_chart_data]
    start_times = [item[0] for item in gantt_chart_data]
    durations = [item[2] for item in gantt_chart_data]
    end_time = max(start_times) + max(durations)

    unique_processes = sorted(list(set(p for p in processes if p != "Idle")))

    # Assign unique colors to processes and a specific color for 'Idle'
    cmap = get_cmap('Set3' if len(unique_processes) <= 12 else 'tab20')
    norm = Normalize(vmin=0, vmax=len(unique_processes) - 1)

    color_map = {proc: cmap(norm(i)) for i, proc in enumerate(unique_processes)}
    color_map["Idle"] = 'lightgray'

    colors = [color_map[p] for p in processes]

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(max(12, end_time / 8), 5))

    # Plot bars
    bars = ax.barh(y=[0] * len(gantt_chart_data),
                   width=durations,
                   left=start_times,
                   height=0.6,
                   color=colors,
                   edgecolor='black',
                   linewidth=1.2,
                   alpha=0.8)

    # Add labels for process IDs
    for start, proc_id, duration in gantt_chart_data:
        if proc_id != "Idle":
            ax.text(start + duration / 2, 0, proc_id, ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')
        elif duration > 2:  # Only label idle if duration is noticeable
            ax.text(start + duration / 2, 0, "Idle", ha='center', va='center',
                    fontsize=9, style='italic', color='darkgray')

    # Set up the axes
    ax.set_title(f"Gantt Chart - {algorithm_name}", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Time Units", fontsize=14)
    ax.set_yticks([0])
    ax.set_yticklabels(["CPU"], fontsize=14)
    ax.set_xticks(np.arange(0, end_time + 1, max(1, math.ceil(end_time / 15))))
    ax.set_xlim(-1, end_time + 2)
    ax.set_ylim(-0.5, 0.5)
    ax.grid(axis='x', linestyle='--', alpha=0.3, color='gray')

    # Style improvements
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Save the chart
    filename = f"{algorithm_name.replace(' ', '_').replace('(', '').replace(')', '')}_Gantt.png"
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Gantt chart saved: {filename}")


def generateComparisonCharts(metrics: List[Dict[str, Any]], output_path: str):
    """
    Generates comprehensive comparison charts for all algorithms.
    """
    if len(metrics) < 2:
        print("Need at least 2 algorithms for comparison charts.")
        return

    # Prepare data for plotting
    algorithms = [m['Algorithm'] for m in metrics]
    avg_tat = [m['Avg_TAT'] for m in metrics]
    avg_wait = [m['Avg_Wait'] for m in metrics]
    avg_response = [m['Avg_Response'] for m in metrics]
    throughput = [m['Throughput'] for m in metrics]
    max_finish = [m['Max_Finish_Time'] for m in metrics]

    # Color palette
    colors = plt.cm.Set2(np.linspace(0, 1, len(algorithms)))

    # 1. Multi-metric Comparison Chart
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('CPU Scheduling Algorithms Performance Comparison', fontsize=18, fontweight='bold')

    # Average Turnaround Time
    bars1 = ax1.bar(algorithms, avg_tat, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax1.set_title('Average Turnaround Time', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Time Units', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    for i, v in enumerate(avg_tat):
        ax1.text(i, v + max(avg_tat) * 0.02, f'{v:.1f}', ha='center', fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)

    # Average Waiting Time
    bars2 = ax2.bar(algorithms, avg_wait, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax2.set_title('Average Waiting Time', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Time Units', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    for i, v in enumerate(avg_wait):
        ax2.text(i, v + max(avg_wait) * 0.02, f'{v:.1f}', ha='center', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)

    # Average Response Time
    bars3 = ax3.bar(algorithms, avg_response, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax3.set_title('Average Response Time', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Time Units', fontsize=12)
    ax3.grid(axis='y', alpha=0.3)
    for i, v in enumerate(avg_response):
        ax3.text(i, v + max(avg_response) * 0.02, f'{v:.1f}', ha='center', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)

    # Throughput
    bars4 = ax4.bar(algorithms, throughput, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax4.set_title('Throughput (Processes/Time Unit)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Processes/Time', fontsize=12)
    ax4.grid(axis='y', alpha=0.3)
    for i, v in enumerate(throughput):
        ax4.text(i, v + max(throughput) * 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    comparison_filename = os.path.join(output_path, "Algorithm_Performance_Comparison.png")
    plt.savefig(comparison_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Performance comparison chart saved: Algorithm_Performance_Comparison.png")

    # 2. Radar Chart for Multi-dimensional Comparison
    generateRadarChart(metrics, output_path)

    # 3. Side-by-side Gantt Chart Comparison
    generateCombinedGanttChart(metrics, output_path)


def generateRadarChart(metrics: List[Dict[str, Any]], output_path: str):
    """
    Generates a radar chart comparing all algorithms across multiple metrics.
    """
    algorithms = [m['Algorithm'] for m in metrics]

    # Normalize metrics for radar chart (0-1 scale)
    # For TAT, Wait, Response: lower is better, so we invert them
    max_tat = max(m['Avg_TAT'] for m in metrics)
    max_wait = max(m['Avg_Wait'] for m in metrics)
    max_response = max(m['Avg_Response'] for m in metrics)
    max_throughput = max(m['Throughput'] for m in metrics)

    # Categories for radar chart
    categories = ['Turnaround Time\n(Lower Better)', 'Waiting Time\n(Lower Better)',
                  'Response Time\n(Lower Better)', 'Throughput\n(Higher Better)']

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Number of variables
    N = len(categories)

    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle

    # Colors for each algorithm
    colors = plt.cm.Set2(np.linspace(0, 1, len(algorithms)))

    for i, metric in enumerate(metrics):
        # Normalize values (invert for metrics where lower is better)
        values = [
            1 - (metric['Avg_TAT'] / max_tat) if max_tat > 0 else 0,  # Invert TAT
            1 - (metric['Avg_Wait'] / max_wait) if max_wait > 0 else 0,  # Invert Wait
            1 - (metric['Avg_Response'] / max_response) if max_response > 0 else 0,  # Invert Response
            metric['Throughput'] / max_throughput if max_throughput > 0 else 0  # Keep Throughput as is
        ]
        values += values[:1]  # Complete the circle

        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, label=algorithms[i], color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])

    # Add category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.grid(True)

    # Add title and legend
    plt.title('Algorithm Performance Radar Chart\n(Outer = Better Performance)',
              size=16, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    radar_filename = os.path.join(output_path, "Algorithm_Radar_Comparison.png")
    plt.savefig(radar_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Radar comparison chart saved: Algorithm_Radar_Comparison.png")


def generateCombinedGanttChart(metrics: List[Dict[str, Any]], output_path: str):
    """
    Generates a combined view of all algorithm Gantt charts for easy comparison.
    """
    if len(metrics) < 2:
        return

    # This function would need gantt data to be passed, but for now we'll create
    # a placeholder that shows the completion times comparison
    algorithms = [m['Algorithm'] for m in metrics]
    completion_times = [m['Max_Finish_Time'] for m in metrics]

    fig, ax = plt.subplots(figsize=(14, 8))

    # Create a horizontal bar chart showing completion times
    y_positions = range(len(algorithms))
    bars = ax.barh(y_positions, completion_times,
                   color=plt.cm.Set2(np.linspace(0, 1, len(algorithms))),
                   alpha=0.7, edgecolor='black', linewidth=1)

    # Add completion time labels
    for i, (bar, time) in enumerate(zip(bars, completion_times)):
        ax.text(bar.get_width() + max(completion_times) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{time}', ha='left', va='center', fontweight='bold', fontsize=11)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(algorithms, fontsize=12)
    ax.set_xlabel('Total Execution Time', fontsize=14)
    ax.set_title('Algorithm Completion Time Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)

    # Style improvements
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    completion_filename = os.path.join(output_path, "Algorithm_Completion_Time_Comparison.png")
    plt.tight_layout()
    plt.savefig(completion_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Completion time comparison saved: Algorithm_Completion_Time_Comparison.png")


def generateProcessWiseComparison(metrics: List[Dict[str, Any]], output_path: str):
    """
    Generates process-wise performance comparison across algorithms.
    """
    if not metrics:
        return

    # Get process IDs from the first algorithm (assuming all have same processes)
    process_ids = [p.processId for p in metrics[0]['Raw_Processes']]
    algorithms = [m['Algorithm'] for m in metrics]

    # Prepare data for each metric
    process_data = {pid: {'TAT': [], 'Wait': [], 'Response': []} for pid in process_ids}

    for metric in metrics:
        for process in metric['Raw_Processes']:
            process_data[process.processId]['TAT'].append(process.turnaroundTime)
            process_data[process.processId]['Wait'].append(process.waitingTime)
            process_data[process.processId]['Response'].append(process.responseTime)

    # Create grouped bar charts
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 14))
    fig.suptitle('Process-wise Performance Comparison Across Algorithms',
                 fontsize=16, fontweight='bold')

    x = np.arange(len(process_ids))
    width = 0.15
    colors = plt.cm.Set2(np.linspace(0, 1, len(algorithms)))

    # Turnaround Time
    for i, alg in enumerate(algorithms):
        tat_values = [process_data[pid]['TAT'][i] for pid in process_ids]
        ax1.bar(x + i * width, tat_values, width, label=alg,
                color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)

    ax1.set_xlabel('Processes', fontsize=12)
    ax1.set_ylabel('Turnaround Time', fontsize=12)
    ax1.set_title('Process-wise Turnaround Time', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width * (len(algorithms) - 1) / 2)
    ax1.set_xticklabels(process_ids)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(axis='y', alpha=0.3)

    # Waiting Time
    for i, alg in enumerate(algorithms):
        wait_values = [process_data[pid]['Wait'][i] for pid in process_ids]
        ax2.bar(x + i * width, wait_values, width, label=alg,
                color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)

    ax2.set_xlabel('Processes', fontsize=12)
    ax2.set_ylabel('Waiting Time', fontsize=12)
    ax2.set_title('Process-wise Waiting Time', fontsize=14, fontweight='bold')
    ax2.set_xticks(x + width * (len(algorithms) - 1) / 2)
    ax2.set_xticklabels(process_ids)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(axis='y', alpha=0.3)

    # Response Time
    for i, alg in enumerate(algorithms):
        response_values = [process_data[pid]['Response'][i] for pid in process_ids]
        ax3.bar(x + i * width, response_values, width, label=alg,
                color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)

    ax3.set_xlabel('Processes', fontsize=12)
    ax3.set_ylabel('Response Time', fontsize=12)
    ax3.set_title('Process-wise Response Time', fontsize=14, fontweight='bold')
    ax3.set_xticks(x + width * (len(algorithms) - 1) / 2)
    ax3.set_xticklabels(process_ids)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    process_filename = os.path.join(output_path, "Process_Wise_Performance_Comparison.png")
    plt.savefig(process_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Process-wise comparison saved: Process_Wise_Performance_Comparison.png")


def generateEfficiencyMatrix(metrics: List[Dict[str, Any]], output_path: str):
    """
    Generates a heatmap showing relative efficiency of algorithms across different metrics.
    """
    algorithms = [m['Algorithm'] for m in metrics]
    metric_names = ['Avg Turnaround', 'Avg Waiting', 'Avg Response', 'Throughput']

    # Create efficiency matrix
    efficiency_matrix = np.zeros((len(algorithms), len(metric_names)))

    # Get raw values
    tat_values = [m['Avg_TAT'] for m in metrics]
    wait_values = [m['Avg_Wait'] for m in metrics]
    response_values = [m['Avg_Response'] for m in metrics]
    throughput_values = [m['Throughput'] for m in metrics]

    # Normalize to 0-100 scale (100 = best performance)
    # For TAT, Wait, Response: lower is better, so invert
    # For Throughput: higher is better

    if max(tat_values) > 0:
        efficiency_matrix[:, 0] = [(max(tat_values) - val) / max(tat_values) * 100 for val in tat_values]

    if max(wait_values) > 0:
        efficiency_matrix[:, 1] = [(max(wait_values) - val) / max(wait_values) * 100 for val in wait_values]

    if max(response_values) > 0:
        efficiency_matrix[:, 2] = [(max(response_values) - val) / max(response_values) * 100 for val in response_values]

    if max(throughput_values) > 0:
        efficiency_matrix[:, 3] = [val / max(throughput_values) * 100 for val in throughput_values]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(efficiency_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Efficiency Score (0-100)', rotation=270, labelpad=20, fontsize=12)

    # Set ticks and labels
    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, fontsize=12)
    ax.set_yticks(range(len(algorithms)))
    ax.set_yticklabels([alg.replace(' ', '\n') for alg in algorithms], fontsize=10)

    # Add text annotations
    for i in range(len(algorithms)):
        for j in range(len(metric_names)):
            text = ax.text(j, i, f'{efficiency_matrix[i, j]:.1f}',
                           ha="center", va="center", color="black", fontweight='bold')

    ax.set_title('Algorithm Efficiency Matrix\n(Higher Score = Better Performance)',
                 fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    heatmap_filename = os.path.join(output_path, "Algorithm_Efficiency_Heatmap.png")
    plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Efficiency heatmap saved: Algorithm_Efficiency_Heatmap.png")


def displayResults(metrics: List[Dict[str, Any]], process_list: List[Process], output_path: str):
    """Enhanced results display with comprehensive analysis."""
    if not metrics:
        return

    # --- 1. Comparison Table ---
    comparison_data = []
    for m in metrics:
        comparison_data.append({
            "Algorithm": m['Algorithm'],
            "Avg Turnaround Time": m['Avg_TAT'],
            "Avg Waiting Time": m['Avg_Wait'],
            "Avg Response Time": m['Avg_Response'],
            "Throughput (proc/time)": m['Throughput'],
            "Max Finish Time": m['Max_Finish_Time']
        })

    df_comparison = pd.DataFrame(comparison_data)

    print("\n" + "=" * 90)
    print("           ENHANCED CPU SCHEDULING ALGORITHM COMPARISON RESULTS")
    print("=" * 90)
    print(df_comparison.to_string(index=False, float_format='{:.3f}'.format))
    print("=" * 90 + "\n")

    # Save comparison to CSV
    csv_filename = os.path.join(output_path, "Enhanced_Scheduling_Comparison.csv")
    df_comparison.to_csv(csv_filename, index=False)
    print(f"Enhanced comparison table saved to: {csv_filename}")

    # --- 2. Performance Analysis ---
    print("\n" + "-" * 90)
    print("                              PERFORMANCE ANALYSIS")
    print("-" * 90)

    # Find best performers
    best_tat = min(metrics, key=lambda x: x['Avg_TAT'])
    best_wait = min(metrics, key=lambda x: x['Avg_Wait'])
    best_response = min(metrics, key=lambda x: x['Avg_Response'])
    best_throughput = max(metrics, key=lambda x: x['Throughput'])

    print(f"🏆 Best Average Turnaround Time: {best_tat['Algorithm']} ({best_tat['Avg_TAT']:.2f})")
    print(f"🏆 Best Average Waiting Time: {best_wait['Algorithm']} ({best_wait['Avg_Wait']:.2f})")
    print(f"🏆 Best Average Response Time: {best_response['Algorithm']} ({best_response['Avg_Response']:.2f})")
    print(f"🏆 Best Throughput: {best_throughput['Algorithm']} ({best_throughput['Throughput']:.4f})")

    # Overall recommendation
    print(f"\n📊 RECOMMENDATION:")
    scores = {}
    for m in metrics:
        # Calculate composite score (lower is better for most metrics)
        score = (m['Avg_TAT'] + m['Avg_Wait'] + m['Avg_Response']) / 3 - m['Throughput'] * 10
        scores[m['Algorithm']] = score

    best_overall = min(scores, key=scores.get)
    print(f"   Overall Best Performer: {best_overall}")
    print("-" * 90 + "\n")

    # --- 3. Generate Enhanced Visualizations ---
    print("Generating enhanced comparison visualizations...")
    generateComparisonCharts(metrics, output_path)
    generateProcessWiseComparison(metrics, output_path)
    generateEfficiencyMatrix(metrics, output_path)

    # --- 4. Individual Process Details ---
    for metric_set in metrics:
        alg_name = metric_set['Algorithm']
        process_data = []
        for p in metric_set['Raw_Processes']:
            process_data.append({
                "ID": p.processId,
                "Arrival": p.arrivalTime,
                "Burst": p.originalBurstTime,
                "Priority": p.priority,
                "Start Time": p.startTime,
                "Finish Time": p.finishTime,
                "Turnaround Time": p.turnaroundTime,
                "Waiting Time": p.waitingTime,
                "Response Time": p.responseTime,
            })

        df_details = pd.DataFrame(process_data)

        # Save details to CSV
        details_filename = os.path.join(output_path,
                                        f"{alg_name.replace(' ', '_').replace('(', '').replace(')', '')}_Process_Details.csv")
        df_details.to_csv(details_filename, index=False)
        print(f"Detailed results for {alg_name} saved to: {details_filename}")


# -----------------------------------------------------------------------------
# --- 7. MAIN EXECUTION ---
# -----------------------------------------------------------------------------

def main():
    """Main function to run the enhanced simulator."""

    print("\n" + "#" * 80)
    print("     ENHANCED CPU SCHEDULING SIMULATOR WITH COMPARISON CHARTS")
    print("     FCFS, SJF, SRTF, Priority, Round Robin")
    print("#" * 80)

    # --- Configuration ---
    N = safeInput("Enter number of processes (N, e.g., 10): ", int)
    quantum = safeInput("Enter Time Quantum for Round Robin (e.g., 4): ", int)
    output_dir = "enhanced_scheduling_results"

    output_path = ensureDirectoryExists(output_dir)
    print(f"\nResults will be saved in: {output_path}")

    # --- Data Generation ---
    initial_processes = generateProcesses(N)
    print(f"\nGenerated {N} processes:")
    for p in initial_processes:
        print(f"  - {p}")

    # --- Scheduling Execution ---

    algorithms = {
        "FCFS": fcfs,
        "SJF (Non-Preemptive)": sjf,
        "SRTF (Preemptive)": srtf,
        "Priority (Preemptive)": priorityScheduling,
        "Round Robin": roundRobin,
    }

    all_metrics: List[Dict[str, Any]] = []

    print("\nRunning scheduling algorithms...")

    for name, func in algorithms.items():
        try:
            # Pass the quantum only to Round Robin, others ignore it
            q = quantum if name == "Round Robin" else 0

            completed_procs, gantt_data = runScheduling(initial_processes, func, q)

            # Calculate metrics
            metrics = calculateMetrics(completed_procs, name)
            all_metrics.append(metrics)

            # Generate individual Gantt Chart
            generateGanttChart(gantt_data, name, output_path)

            print(f"✓ Completed: {name}")

        except Exception as e:
            print(f"\n❌ Error in '{name}': {e}", file=sys.stderr)
            continue  # Continue to the next algorithm

    # --- Final Results and Visualizations ---
    if all_metrics:
        # Display the enhanced results with comprehensive analysis
        displayResults(all_metrics, initial_processes, output_path)
    else:
        print("\n❌ No successful simulations completed.")

    print("\n" + "#" * 80)
    print("🎉 ENHANCED SIMULATION COMPLETE!")
    print(f"📁 Check the '{output_dir}' directory for:")
    print("   • Individual Gantt Charts")
    print("   • Performance Comparison Charts")
    print("   • Radar Chart Analysis")
    print("   • Process-wise Comparisons")
    print("   • Efficiency Heatmap")
    print("   • Detailed CSV Reports")
    print("#" * 80)


if __name__ == "__main__":
    main()