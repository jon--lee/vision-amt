#!/bin/sh
source '~/.bashrc'
alias tpython='LD_LIBRARY_PATH=/home/annal/tflow/libc6_2.17/lib/x86_64-linux-gnu/ /home/annal/tflow/libc6_2.17/lib/x86_64-linux-gnu/ld-2.17.so /usr/bin/python'
echo "Jonathan"
echo "sup20"
tpython scripts/train_izzy_net.py -n Jonathan -s -f 0 -l 20 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "sup40"
tpython scripts/train_izzy_net.py -n Jonathan -s -f 0 -l 40 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "sup60"
tpython scripts/train_izzy_net.py -n Jonathan -s -f 0 -l 60 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "DAg20"
tpython scripts/train_izzy_net.py -n Jonathan -d -f 0 -l 20 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "DAg40"
tpython scripts/train_izzy_net.py -n Jonathan -d -f 0 -l 40 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value

echo "ChJon"
echo "sup20"
tpython scripts/train_izzy_net.py -n ChJon -s -f 0 -l 20 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "sup40"
tpython scripts/train_izzy_net.py -n ChJon -s -f 0 -l 40 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "sup60"
tpython scripts/train_izzy_net.py -n ChJon -s -f 0 -l 60 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "DAg20"
tpython scripts/train_izzy_net.py -n ChJon -d -f 0 -l 20 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "DAg40"
tpython scripts/train_izzy_net.py -n ChJon -d -f 0 -l 40 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value

echo "Jacky"
echo "sup20"
tpython scripts/train_izzy_net.py -n Jacky -s -f 0 -l 20 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "sup40"
tpython scripts/train_izzy_net.py -n Jacky -s -f 0 -l 40 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "sup60"
tpython scripts/train_izzy_net.py -n Jacky -s -f 0 -l 60 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "DAg20"
tpython scripts/train_izzy_net.py -n Jacky -d -f 0 -l 20 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "DAg40"
tpython scripts/train_izzy_net.py -n Jacky -d -f 0 -l 40 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value

echo "Aimee"
echo "sup20"
tpython scripts/train_izzy_net.py -n Aimee -s -f 0 -l 20 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "sup40"
tpython scripts/train_izzy_net.py -n Aimee -s -f 0 -l 40 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "sup60"
tpython scripts/train_izzy_net.py -n Aimee -s -f 0 -l 60 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "DAg20"
tpython scripts/train_izzy_net.py -n Aimee -d -f 0 -l 20 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "DAg40"
tpython scripts/train_izzy_net.py -n Aimee -d -f 0 -l 40 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value

echo "Chris"
echo "sup20"
tpython scripts/train_izzy_net.py -n Chris -s -f 0 -l 20 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "sup40"
tpython scripts/train_izzy_net.py -n Chris -s -f 0 -l 40 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "sup60"
tpython scripts/train_izzy_net.py -n Chris -s -f 0 -l 60 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "DAg20"
tpython scripts/train_izzy_net.py -n Chris -d -f 0 -l 20 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "DAg40"
tpython scripts/train_izzy_net.py -n Chris -d -f 0 -l 40 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value

echo "Dave"
echo "sup20"
tpython scripts/train_izzy_net.py -n Dave -s -f 0 -l 20 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "sup40"
tpython scripts/train_izzy_net.py -n Dave -s -f 0 -l 40 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "sup60"
tpython scripts/train_izzy_net.py -n Dave -s -f 0 -l 60 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "DAg20"
tpython scripts/train_izzy_net.py -n Dave -d -f 0 -l 20 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "DAg40"
tpython scripts/train_izzy_net.py -n Dave -d -f 0 -l 40 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value

echo "Lauren"
echo "sup20"
tpython scripts/train_izzy_net.py -n Lauren -s -f 0 -l 20 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "sup40"
tpython scripts/train_izzy_net.py -n Lauren -s -f 0 -l 40 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "sup60"
tpython scripts/train_izzy_net.py -n Lauren -s -f 0 -l 60 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "DAg20"
tpython scripts/train_izzy_net.py -n Lauren -d -f 0 -l 20 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "DAg40"
tpython scripts/train_izzy_net.py -n Lauren -d -f 0 -l 40 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value

echo "Johan"
echo "sup20"
tpython scripts/train_izzy_net.py -n Johan -s -f 0 -l 20 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "sup40"
tpython scripts/train_izzy_net.py -n Johan -s -f 0 -l 40 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "sup60"
tpython scripts/train_izzy_net.py -n Johan -s -f 0 -l 60 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "DAg20"
tpython scripts/train_izzy_net.py -n Johan -d -f 0 -l 20 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "DAg40"
tpython scripts/train_izzy_net.py -n Johan -d -f 0 -l 40 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value

echo "Sherdil"
echo "sup20"
tpython scripts/train_izzy_net.py -n Sherdil -s -f 0 -l 20 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "sup40"
tpython scripts/train_izzy_net.py -n Sherdil -s -f 0 -l 40 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "sup60"
tpython scripts/train_izzy_net.py -n Sherdil -s -f 0 -l 60 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "DAg20"
tpython scripts/train_izzy_net.py -n Sherdil -d -f 0 -l 20 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "DAg40"
tpython scripts/train_izzy_net.py -n Sherdil -d -f 0 -l 40 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value

echo "Aimee"
echo "sup20"
tpython scripts/train_izzy_net.py -n Aimee -s -f 0 -l 20 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "sup40"
tpython scripts/train_izzy_net.py -n Aimee -s -f 0 -l 40 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "sup60"
tpython scripts/train_izzy_net.py -n Aimee -s -f 0 -l 60 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "DAg20"
tpython scripts/train_izzy_net.py -n Aimee -d -f 0 -l 20 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "DAg40"
tpython scripts/train_izzy_net.py -n Aimee -d -f 0 -l 40 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value


echo "Sona"
echo "sup20"
tpython scripts/train_izzy_net.py -n Sona -s -f 0 -l 20 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "sup40"
tpython scripts/train_izzy_net.py -n Sona -s -f 0 -l 40 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "sup60"
tpython scripts/train_izzy_net.py -n Sona -s -f 0 -l 60 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "DAg20"
tpython scripts/train_izzy_net.py -n Sona -d -f 0 -l 20 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "DAg40"
tpython scripts/train_izzy_net.py -n Sona -d -f 0 -l 40 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value

echo "Richard"
echo "sup20"
tpython scripts/train_izzy_net.py -n Richard -s -f 0 -l 20 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "sup40"
tpython scripts/train_izzy_net.py -n Richard -s -f 0 -l 40 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "sup60"
tpython scripts/train_izzy_net.py -n Richard -s -f 0 -l 60 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "DAg20"
tpython scripts/train_izzy_net.py -n Richard -d -f 0 -l 20 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
echo "DAg40"
tpython scripts/train_izzy_net.py -n Richard -d -f 0 -l 40 -sm 5
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py $value
