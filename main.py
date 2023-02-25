"""
CS6140 Project 3
Yihan Xu
Jake Van Meter
"""

import sys
import preprocessing as pp
import classification as cls


def main():
    argv = sys.argv
    # If desired task is not specified, run all tasks
    if len(argv) == 1:
        argv.append("all")
    #################### Preprocessing ####################
    if "all" in argv or "1" in argv:
        pp.main()
    #################### Classification with iteration ####################
    if "all" in argv or "2" in argv:
        cls.main()


if __name__ == "__main__":
    main()
