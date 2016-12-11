#!/bin/sh

alias tpython='LD_LIBRARY_PATH=/home/annal/tflow/libc6_2.17/lib/x86_64-linux-gnu/ /home/annal/tflow/libc6_2.17/lib/x86_64-linux-gnu/ld-2.17.so /usr/bin/python'

echo "Sona"
echo "sup60"
tpython scripts/error_cleaner.py -n Sona -o comparisons.txt -f 0 -l 60

echo "Jonathan"
echo "sup60"
tpython scripts/write_delta_corrections.py -n Jonathan -o full_deltas.txt -d new_remove.txt -f 0 -l 60
tpython scripts/train_izzy_net.py -n Jonathan -s -f 0 -l 60 -sm 5 -t 5 -ho stats.txt -dn full_deltas.txt
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value

echo "ChJon"
echo "sup60"
tpython scripts/write_delta_corrections.py -n ChJon -o full_deltas.txt -d new_remove.txt -f 0 -l 60
tpython scripts/train_izzy_net.py -n ChJon -s -f 0 -l 60 -sm 5 -t 5 -ho stats.txt -dn full_deltas.txt
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value

echo "Jacky"
echo "sup60"
tpython scripts/write_delta_corrections.py -n Jacky -o full_deltas.txt -d new_remove.txt -f 0 -l 60
tpython scripts/train_izzy_net.py -n Jacky -s -f 0 -l 60 -sm 5 -t 5 -ho stats.txt -dn full_deltas.txt
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value

echo "Aimee"
echo "sup60"
tpython scripts/write_delta_corrections.py -n Aimee -o full_deltas.txt -d new_remove.txt -f 0 -l 60
tpython scripts/train_izzy_net.py -n Aimee -s -f 0 -l 60 -sm 5 -t 5 -ho stats.txt -dn full_deltas.txt
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value

echo "Chris"
echo "sup60"
tpython scripts/write_delta_corrections.py -n Chris -o full_deltas.txt -d new_remove.txt -f 0 -l 60
tpython scripts/train_izzy_net.py -n Chris -s -f 0 -l 60 -sm 5 -t 5 -ho stats.txt -dn full_deltas.txt
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value

echo "Dave"
echo "sup60"
tpython scripts/write_delta_corrections.py -n Dave -o full_deltas.txt -d new_remove.txt -f 0 -l 60
tpython scripts/train_izzy_net.py -n Dave -s -f 0 -l 60 -sm 5 -t 5 -ho stats.txt -dn full_deltas.txt
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value

echo "Lauren"
echo "sup60"
tpython scripts/write_delta_corrections.py -n Lauren -o full_deltas.txt -d new_remove.txt -f 0 -l 60
tpython scripts/train_izzy_net.py -n Lauren -s -f 0 -l 60 -sm 5 -t 5 -ho stats.txt -dn full_deltas.txt
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value

echo "Johan"
echo "sup60"
tpython scripts/write_delta_corrections.py -n Johan -o full_deltas.txt -d new_remove.txt -f 0 -l 60
tpython scripts/train_izzy_net.py -n Johan -s -f 0 -l 60 -sm 5 -t 5 -ho stats.txt -dn full_deltas.txt
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value

echo "Sherdil"
echo "sup60"
tpython scripts/write_delta_corrections.py -n Sherdil -o full_deltas.txt -d new_remove.txt -f 0 -l 60
tpython scripts/train_izzy_net.py -n Sherdil -s -f 0 -l 60 -sm 5 -t 5 -ho stats.txt -dn full_deltas.txt
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value

echo "Sona"
echo "sup60"
tpython scripts/write_delta_corrections.py -n Sona -o full_deltas.txt -d new_remove.txt -f 0 -l 60
tpython scripts/train_izzy_net.py -n Sona -s -f 0 -l 60 -sm 5 -t 5 -ho stats.txt -dn full_deltas.txt
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value

echo "Richard"
echo "sup60"
tpython scripts/write_delta_corrections.py -n Richard -o full_deltas.txt -d new_remove.txt -f 0 -l 60
tpython scripts/train_izzy_net.py -n Richard -s -f 0 -l 60 -sm 5 -t 5 -ho stats.txt -dn full_deltas.txt
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
