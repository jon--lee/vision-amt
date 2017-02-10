#!/bin/sh
source '~/.bashrc'
alias tpython='LD_LIBRARY_PATH=/home/annal/tflow/libc6_2.17/lib/x86_64-linux-gnu/ /home/annal/tflow/libc6_2.17/lib/x86_64-linux-gnu/ld-2.17.so /usr/bin/python'

tpython scripts/smoothing_differences.py -n Jonathan -tg smoothings.txt -ho stats.txt -f 0 -l 60 -sm 5
tpython scripts/smoothing_differences.py -n Sona -tg smoothings.txt -ho stats.txt -f 0 -l 60 -sm 5
tpython scripts/smoothing_differences.py -n Aimee -tg smoothings.txt -ho stats.txt -f 0 -l 60 -sm 5
tpython scripts/smoothing_differences.py -n Dave -tg smoothings.txt -ho stats.txt -f 0 -l 60 -sm 5
tpython scripts/smoothing_differences.py -n Jacky -tg smoothings.txt -ho stats.txt -f 0 -l 60 -sm 5
tpython scripts/smoothing_differences.py -n Sherdil -tg smoothings.txt -ho stats.txt -f 0 -l 60 -sm 5
tpython scripts/smoothing_differences.py -n Richard -tg smoothings.txt -ho stats.txt -f 0 -l 60 -sm 5
tpython scripts/smoothing_differences.py -n Johan -tg smoothings.txt -ho stats.txt -f 0 -l 60 -sm 5
