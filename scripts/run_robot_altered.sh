#!/bin/sh
source '~/.bashrc'
alias tpython='LD_LIBRARY_PATH=/home/annal/tflow/libc6_2.17/lib/x86_64-linux-gnu/ /home/annal/tflow/libc6_2.17/lib/x86_64-linux-gnu/ld-2.17.so /usr/bin/python'

python scripts/write_delta_corrections.py -n Aimee -o full_deltas.txt -d new_robot_alter.txt -f 0 -l 60
tpython scripts/train_izzy_net.py -n Aimee -f 0 -l 60 -ho stats.txt -dn full_deltas.txt -s

python scripts/write_delta_corrections.py -n Johan -o full_deltas.txt -d new_robot_alter.txt -f 0 -l 60
tpython scripts/train_izzy_net.py -n Johan -f 0 -l 60 -ho stats.txt -dn full_deltas.txt -s

python scripts/write_delta_corrections.py -n Richard -o full_deltas.txt -d new_robot_alter.txt -f 0 -l 60
tpython scripts/train_izzy_net.py -n Richard -f 0 -l 60 -ho stats.txt -dn full_deltas.txt -s

python scripts/write_delta_corrections.py -n Jonathan -o full_deltas.txt -d new_robot_alter.txt -f 0 -l 60
tpython scripts/train_izzy_net.py -n Jonathan -f 0 -l 60 -ho stats.txt -dn full_deltas.txt -s

python scripts/write_delta_corrections.py -n Dave -o full_deltas.txt -d new_deltas.txt -f 0 -l 60
tpython scripts/train_izzy_net.py -n Dave -f 0 -l 60 -ho stats.txt -dn full_deltas.txt -s

python scripts/write_delta_corrections.py -n Sherdil -o full_deltas.txt -d new_deltas.txt -f 0 -l 60
tpython scripts/train_izzy_net.py -n Sherdil -f 0 -l 60 -ho stats.txt -dn full_deltas.txt -s

python scripts/write_delta_corrections.py -n Jacky -o full_deltas.txt -d new_deltas.txt -f 0 -l 60
tpython scripts/train_izzy_net.py -n Jacky -f 0 -l 60 -ho stats.txt -dn full_deltas.txt -s

python scripts/write_delta_corrections.py -n Sona -o full_deltas.txt -d new_deltas.txt -f 0 -l 60
tpython scripts/train_izzy_net.py -n Sona -f 0 -l 60 -ho stats.txt -dn full_deltas.txt -s

python scripts/write_delta_corrections.py -n Dave -o full_deltas.txt -d new_deltas.txt -f 0 -l 60
tpython scripts/train_izzy_net.py -n Dave -f 0 -l 60 -ho stats.txt -dn full_deltas.txt -s -sm 5

python scripts/write_delta_corrections.py -n Sherdil -o full_deltas.txt -d new_deltas.txt -f 0 -l 60
tpython scripts/train_izzy_net.py -n Sherdil -f 0 -l 60 -ho stats.txt -dn full_deltas.txt -s -sm 5

python scripts/write_delta_corrections.py -n Jacky -o full_deltas.txt -d new_deltas.txt -f 0 -l 60
tpython scripts/train_izzy_net.py -n Jacky -f 0 -l 60 -ho stats.txt -dn full_deltas.txt -s -sm 5

python scripts/write_delta_corrections.py -n Sona -o full_deltas.txt -d new_deltas.txt -f 0 -l 60
tpython scripts/train_izzy_net.py -n Sona -f 0 -l 60 -ho stats.txt -dn full_deltas.txt -s -sm 5
