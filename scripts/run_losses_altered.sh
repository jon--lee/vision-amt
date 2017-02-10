#!/bin/sh
source '~/.bashrc'
alias tpython='LD_LIBRARY_PATH=/home/annal/tflow/libc6_2.17/lib/x86_64-linux-gnu/ /home/annal/tflow/libc6_2.17/lib/x86_64-linux-gnu/ld-2.17.so /usr/bin/python'
echo "Smoothing and error corrected"

# echo "Jonathan"
# echo "sup60"
# python scripts/write_delta_corrections.py -n Jonathan -o full_deltas.txt -d new_deltas.txt -f 0 -l 60
# tpython scripts/train_izzy_net.py -n Jonathan -f 0 -l 60 -ho stats.txt -dn full_deltas.txt -s -nn -sm 5
# tpython scripts/visualizer_supervised.py -nn /media/1tb/Izzy/nets/net6_11-29-2016_14h26m58s.ckpt -n Jonathan

# echo "Jacky"
# echo "sup60"
# python scripts/write_delta_corrections.py -n Jacky -o full_deltas.txt -d new_deltas.txt -f 0 -l 60
# tpython scripts/train_izzy_net.py -n Jacky -f 0 -l 60 -ho stats.txt -dn full_deltas.txt -s -nn -sm 5
# tpython scripts/visualizer_supervised.py -nn /media/1tb/Izzy/nets/net6_12-11-2016_16h10m45s.ckpt -n Jacky

# echo "Aimee"
# echo "sup60"
# python scripts/write_delta_corrections.py -n Aimee -o full_deltas.txt -d new_deltas.txt -f 0 -l 60
# tpython scripts/train_izzy_net.py -n Aimee -f 0 -l 60 -ho stats.txt -dn full_deltas.txt -s -nn -sm 5
# tpython scripts/visualizer_supervised.py -nn /media/1tb/Izzy/nets/net6_11-28-2016_20h45m31s.ckpt -n Aimee

# echo "Dave"
# echo "sup60"
# python scripts/write_delta_corrections.py -n Dave -o full_deltas.txt -d new_deltas.txt -f 0 -l 60
# tpython scripts/train_izzy_net.py -n Dave -f 0 -l 60 -ho stats.txt -dn full_deltas.txt -s -nn -sm 5
# tpython scripts/visualizer_supervised.py -nn /media/1tb/Izzy/nets/net6_12-11-2016_15h53m40s.ckpt -n Dave

# echo "Johan"
# echo "sup60"
# python scripts/write_delta_corrections.py -n Johan -o full_deltas.txt -d new_deltas.txt -f 0 -l 60
# tpython scripts/train_izzy_net.py -n Johan -f 0 -l 60 -ho stats.txt -dn full_deltas.txt -s -nn -sm 5
# tpython scripts/visualizer_supervised.py -nn /media/1tb/Izzy/nets/net6_11-30-2016_11h53m30s.ckpt -n Johan

# echo "Sherdil"
# echo "sup60"
# python scripts/write_delta_corrections.py -n Sherdil -o full_deltas.txt -d new_deltas.txt -f 0 -l 60
# tpython scripts/train_izzy_net.py -n Sherdil -f 0 -l 60 -ho stats.txt -dn full_deltas.txt -s -nn -sm 5
# tpython scripts/visualizer_supervised.py -nn /media/1tb/Izzy/nets/net6_12-11-2016_16h02m12s.ckpt -n Sherdil

# echo "Sona"
# echo "sup60"
# python scripts/write_delta_corrections.py -n Sona -o full_deltas.txt -d new_deltas.txt -f 0 -l 60
# tpython scripts/train_izzy_net.py -n Sona -f 0 -l 60 -ho stats.txt -dn full_deltas.txt -s -nn -sm 5
# tpython scripts/visualizer_supervised.py -nn /media/1tb/Izzy/nets/net6_12-11-2016_16h19m17s.ckpt -n Sona

# echo "Richard"
# echo "sup60"
# python scripts/write_delta_corrections.py -n Richard -o full_deltas.txt -d new_deltas.txt -f 0 -l 60
# tpython scripts/train_izzy_net.py -n Richard -f 0 -l 60 -ho stats.txt -dn full_deltas.txt -s -nn -sm 5
# tpython scripts/visualizer_supervised.py -nn /media/1tb/Izzy/nets/net6_11-29-2016_15h42m11s.ckpt -n Richard

echo "Error corrected"

echo "Jonathan"
echo "sup60"
python scripts/write_delta_corrections.py -n Jonathan -o full_deltas.txt -d new_deltas.txt -f 0 -l 60
tpython scripts/train_izzy_net.py -n Jonathan -f 0 -l 60 -ho stats.txt -dn full_deltas.txt -s
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py -nn $value -n Jonathan

echo "Jacky"
echo "sup60"
python scripts/write_delta_corrections.py -n Jacky -o full_deltas.txt -d new_deltas.txt -f 0 -l 60
tpython scripts/train_izzy_net.py -n Jacky -f 0 -l 60 -ho stats.txt -dn full_deltas.txt -s
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py -nn $value -n Jacky

echo "Aimee"
echo "sup60"
python scripts/write_delta_corrections.py -n Aimee -o full_deltas.txt -d new_deltas.txt -f 0 -l 60
tpython scripts/train_izzy_net.py -n Aimee -f 0 -l 60 -ho stats.txt -dn full_deltas.txt -s
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py -nn $value -n Aimee

echo "Dave"
echo "sup60"
python scripts/write_delta_corrections.py -n Dave -o full_deltas.txt -d new_deltas.txt -f 0 -l 60
tpython scripts/train_izzy_net.py -n Dave -f 0 -l 60 -ho stats.txt -dn full_deltas.txt -s
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py -nn $value -n Dave

echo "Johan"
echo "sup60"
python scripts/write_delta_corrections.py -n Johan -o full_deltas.txt -d new_deltas.txt -f 0 -l 60
tpython scripts/train_izzy_net.py -n Johan -f 0 -l 60 -ho stats.txt -dn full_deltas.txt -s
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py -nn $value -n Johan

echo "Sherdil"
echo "sup60"
python scripts/write_delta_corrections.py -n Sherdil -o full_deltas.txt -d new_deltas.txt -f 0 -l 60
tpython scripts/train_izzy_net.py -n Sherdil -f 0 -l 60 -ho stats.txt -dn full_deltas.txt -s
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py -nn $value -n Sherdil

echo "Sona"
echo "sup60"
python scripts/write_delta_corrections.py -n Sona -o full_deltas.txt -d new_deltas.txt -f 0 -l 60
tpython scripts/train_izzy_net.py -n Sona -f 0 -l 60 -ho stats.txt -dn full_deltas.txt -s
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py -nn $value -n Sona

echo "Richard"
echo "sup60"
python scripts/write_delta_corrections.py -n Richard -o full_deltas.txt -d new_deltas.txt -f 0 -l 60
tpython scripts/train_izzy_net.py -n Richard -f 0 -l 60 -ho stats.txt -dn full_deltas.txt -s
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py -nn $value -n Richard

