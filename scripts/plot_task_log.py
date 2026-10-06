#!/usr/bin/env python3
import argparse
from evaluation.analysis import plot_run

def main():
    p=argparse.ArgumentParser(description='Plot task log, including its motion duration and actuator metrics')
    p.add_argument('--log',required=True);p.add_argument('--output');p.add_argument('--show',action='store_true')
    p.add_argument('--include-transitions',action='store_true')
    a=p.parse_args()
    try:plot_run(a.log,a.output,a.include_transitions,a.show)
    except (ValueError,KeyError,OSError) as exc:p.error(str(exc))
if __name__=='__main__':main()
