#!/bin/sh
source '~/.bashrc'
alias tpython='LD_LIBRARY_PATH=/home/annal/tflow/libc6_2.17/lib/x86_64-linux-gnu/ /home/annal/tflow/libc6_2.17/lib/x86_64-linux-gnu/ld-2.17.so /usr/bin/python'
# echo "Jonathan"
# echo "sup60"
# tpython scripts/train_izzy_net.py -n Jonathan -s -f 0 -l 60  -t 5 -ho stats.txt
# value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
# tpython scripts/visualizer_supervised.py -nn $value

# # echo "ChJon"
# # echo "sup60"
# # tpython scripts/train_izzy_net.py -n ChJon -s -f 0 -l 60  -t 5 -ho stats.txt
# # value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
# # tpython scripts/visualizer_supervised.py -nn $value

# echo "Jacky"
# echo "sup60"
# tpython scripts/train_izzy_net.py -n Jacky -s -f 0 -l 60  -t 5 -ho stats.txt
# value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
# tpython scripts/visualizer_supervised.py -nn $value

# echo "Aimee"
# echo "sup60"
# tpython scripts/train_izzy_net.py -n Aimee -s -f 0 -l 60  -t 5 -ho stats.txt
# value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
# tpython scripts/visualizer_supervised.py -nn $value

# echo "Chris"
# echo "sup60"
# tpython scripts/train_izzy_net.py -n Chris -s -f 0 -l 60  -t 5 -ho stats.txt
# value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
# tpython scripts/visualizer_supervised.py -nn $value

# echo "Dave"
# echo "sup60"
# tpython scripts/train_izzy_net.py -n Dave -s -f 0 -l 60  -t 5 -ho stats.txt
# value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
# tpython scripts/visualizer_supervised.py -nn $value

# echo "Lauren"
# echo "sup60"
# tpython scripts/train_izzy_net.py -n Lauren -s -f 0 -l 60  -t 5 -ho stats.txt
# value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
# tpython scripts/visualizer_supervised.py -nn $value

# echo "Johan"
# echo "sup60"
# tpython scripts/train_izzy_net.py -n Johan -s -f 0 -l 60  -t 5 -ho stats.txt
# value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
# tpython scripts/visualizer_supervised.py -nn $value

# echo "Sherdil"
# echo "sup60"
# tpython scripts/train_izzy_net.py -n Sherdil -s -f 0 -l 60  -t 5 -ho stats.txt
# value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
# tpython scripts/visualizer_supervised.py -nn $value

# echo "Sona"
# echo "sup60"
# tpython scripts/train_izzy_net.py -n Sona -s -f 0 -l 60 -t 5 -ho stats.txt
# value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
# tpython scripts/visualizer_supervised.py -nn $value

echo "Richard"
echo "sup60"
tpython scripts/train_izzy_net.py -n Richard -s -f 0 -l 60 -t 5 -ho stats.txt
value=`cat /home/annal/Izzy/vision_amt/data/amt/last_net.txt`
tpython scripts/visualizer_supervised.py -nn $value
