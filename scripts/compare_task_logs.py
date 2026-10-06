#!/usr/bin/env python3
import argparse
from evaluation.analysis import compare_runs

def main():
    p=argparse.ArgumentParser(description='Compare tasks and durations; identify worst-case actuator requirements')
    p.add_argument('logs',nargs='+');p.add_argument('--output',default='task_comparison')
    p.add_argument('--include-transitions',action='store_true')
    a=p.parse_args()
    try:compare_runs(a.logs,a.output,a.include_transitions)
    except (ValueError,KeyError,OSError) as exc:p.error(str(exc))
if __name__=='__main__':main()
