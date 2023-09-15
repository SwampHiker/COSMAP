#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import time
import numpy as np
from functools import partial
from typing import Optional as O, Union as U
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


def default_print_task_function(idx, idx_num, step_, task_name: str, level=1):
    level_pref = '\t' * level
    if step_ is not None:
        if (idx % step_) == 0:
            print('{}[{}/{}] (l{}) ({})'.format(level_pref, idx, idx_num, level, task_name))


def task_function_helper_default(pdata, funtion_ptr):
    idx, idx_num, print_fun, params_ = pdata
    ret = funtion_ptr(*params_)
    print_fun(idx, idx_num)
    return ret


def parallel_tasks_run(task_function, task_data: list,
                       num_prints: O[int] = None,
                       print_function=default_print_task_function,
                       level=1,
                       task_name: str = 'unnamed-task',
                       num_threads: int = 1,
                       use_process: bool = True) -> list:
    num_task = len(task_data)
    print(':: start ({}) #data/#threads = {}/{}, use-process({})'
                 .format(task_name, num_task, num_threads, use_process))
    if num_prints is not None:
        step_print = int(np.ceil(num_task / num_prints))
    else:
        step_print = None
    print_function = partial(print_function, step_=step_print, task_name=task_name, level=level)
    task_data = [[xi, len(task_data), print_function, x if isinstance(x, list) else [x]]
                 for xi, x in enumerate(task_data)]
    if (len(task_data) < 2) or (num_threads < 2):
        ret = [task_function(x) for x in task_data]
    else:
        if use_process:
            pool_ = ProcessPoolExecutor(max_workers=num_threads)
        else:
            pool_ = ThreadPoolExecutor(max_workers=num_threads)
        ret = list(pool_.map(task_function, task_data))
        pool_.shutdown()
    return ret


def parallel_tasks_run_def(task_function, task_data: list,
                           num_prints: O[int] = 10,
                           print_function=default_print_task_function,
                           level=1,
                           task_name: str = 'unnamed-task',
                           num_workers: int = 1,
                           use_process: bool = True,
                           print_timing=True):
    task_function_def_ = partial(task_function_helper_default, funtion_ptr=task_function)
    t1 = time.time()
    ret = parallel_tasks_run(task_function=task_function_def_,
                             task_data=task_data,
                             num_prints=num_prints,
                             print_function=print_function,
                             level=level,
                             task_name=task_name,
                             num_threads=num_workers,
                             use_process=use_process)
    dt = time.time() - t1
    if print_timing:
        speed_ = len(task_data) / dt
        print('\t{}... done, dt ~ {:0.2f} (s), speed={:0.2f} (samples/s) [{}]'.format('\t' * level, dt, speed_, task_name))
    return ret
