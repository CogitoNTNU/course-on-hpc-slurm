import multiprocessing
import os
import time


work = (["A", 5], ["B", 2], ["C", 1], ["D", 3])


def work_log(work_data):
    print(f"Worker process ID: {os.getpid()}")
    print(f"Worker CPU core: {os.sched_getaffinity(0)}")
    print(f"Worker CPU count: {multiprocessing.cpu_count()}")

    print(" Process %s waiting %s seconds" % (work_data[0], work_data[1]))
    time.sleep(int(work_data[1]))
    print(" Process %s Finished." % work_data[0])


if __name__ == "__main__":
    print(f"Main process ID: {os.getpid()}")
    print(f"Main CPU core: {os.sched_getaffinity(0)}")
    print(f"Main CPU count: {multiprocessing.cpu_count()}")
    print(f"Total of CPU core: {os.cpu_count()}")

    print("Simulating work...")
    p = multiprocessing.Pool(len(work))
    p.map(work_log, work)
