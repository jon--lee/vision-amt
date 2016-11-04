#!/bin/sh
source '~/.bashrc'
alias tpython='LD_LIBRARY_PATH=/home/annal/tflow/libc6_2.17/lib/x86_64-linux-gnu/ /home/annal/tflow/libc6_2.17/lib/x86_64-linux-gnu/ld-2.17.so /usr/bin/python'
echo "Jonathan"
echo "DAg40"
tpython scripts/train_izzy_net.py -n Jonathan -d -f 0 -l 40
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value

echo "ChJon"
echo "DAg40"
tpython scripts/train_izzy_net.py -n ChJon -d -f 0 -l 40
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value

echo "Jacky"
echo "DAg40"
tpython scripts/train_izzy_net.py -n Jacky -d -f 0 -l 40
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value

echo "Aimee"
echo "DAg40"
tpython scripts/train_izzy_net.py -n Aimee -d -f 0 -l 40
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value

echo "Chris"
echo "DAg40"
tpython scripts/train_izzy_net.py -n Chris -d -f 0 -l 40
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value

echo "Dave"
echo "DAg40"
tpython scripts/train_izzy_net.py -n Dave -d -f 0 -l 40
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value

echo "Lauren"
echo "DAg40"
tpython scripts/train_izzy_net.py -n Lauren -d -f 0 -l 40
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value

echo "Johan"
echo "DAg40"
tpython scripts/train_izzy_net.py -n Johan -d -f 0 -l 40
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value

echo "Sherdil"
echo "DAg40"
tpython scripts/train_izzy_net.py -n Sherdil -d -f 0 -l 40
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value


echo "Sona"
echo "DAg40"
tpython scripts/train_izzy_net.py -n Sona -d -f 0 -l 40
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
