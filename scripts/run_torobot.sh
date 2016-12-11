#!/bin/sh
source '~/.bashrc'
alias tpython='LD_LIBRARY_PATH=/home/annal/tflow/libc6_2.17/lib/x86_64-linux-gnu/ /home/annal/tflow/libc6_2.17/lib/x86_64-linux-gnu/ld-2.17.so /usr/bin/python'

tpython scripts/correct_error.py -n Aimee
tpython scripts/correct_error.py -n Johan
tpython scripts/correct_error.py -n Jonathan
tpython scripts/correct_error.py -n Richard
tpython scripts/correct_error.py -n Jacky
tpython scripts/correct_error.py -n Sona
tpython scripts/correct_error.py -n Chris
tpython scripts/correct_error.py -n Lauren
tpython scripts/correct_error.py -n Dave
tpython scripts/correct_error.py -n ChJon
tpython scripts/correct_error.py -n Sherdil