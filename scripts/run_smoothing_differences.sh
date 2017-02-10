#!/bin/sh
source '~/.bashrc'
alias tpython='LD_LIBRARY_PATH=/home/annal/tflow/libc6_2.17/lib/x86_64-linux-gnu/ /home/annal/tflow/libc6_2.17/lib/x86_64-linux-gnu/ld-2.17.so /usr/bin/python'

echo "evaluating holdout differences"
tpython scripts/train_izzy_net.py -n Jonathan -s -nn -f 0 -l 60 -t 5 -ho stats.txt
tpython scripts/visualizer_supervised.py /media/1tb/Izzy/nets/net6_11-27-2016_18h55m18s.ckpt -n Jonathan

tpython scripts/train_izzy_net.py -n ChJon -s -nn -f 0 -l 60 -t 5 -ho stats.txt
tpython scripts/visualizer_supervised.py /media/1tb/Izzy/nets/net6_11-27-2016_19h04m23s.ckpt -n ChJon

tpython scripts/train_izzy_net.py -n Jacky -s -nn -f 0 -l 60 -t 5 -ho stats.txt
tpython scripts/visualizer_supervised.py /media/1tb/Izzy/nets/net6_11-27-2016_19h13m32s.ckpt -n Jacky

tpython scripts/train_izzy_net.py -n Aimee -s -nn -f 0 -l 60 -t 5 -ho stats.txt
tpython scripts/visualizer_supervised.py /media/1tb/Izzy/nets/net6_11-27-2016_19h22m36s.ckpt -n Aimee

tpython scripts/train_izzy_net.py -n Chris -s -nn -f 0 -l 60 -t 5 -ho stats.txt
tpython scripts/visualizer_supervised.py /media/1tb/Izzy/nets/net6_11-27-2016_19h04m23s.ckpt -n ChJon

tpython scripts/train_izzy_net.py -n ChJon -s -nn -f 0 -l 60 -t 5 -ho stats.txt
tpython scripts/visualizer_supervised.py /media/1tb/Izzy/nets/net6_11-27-2016_19h04m23s.ckpt -n ChJon

tpython scripts/train_izzy_net.py -n ChJon -s -nn -f 0 -l 60 -t 5 -ho stats.txt
tpython scripts/visualizer_supervised.py /media/1tb/Izzy/nets/net6_11-27-2016_19h04m23s.ckpt -n ChJon

tpython scripts/train_izzy_net.py -n ChJon -s -nn -f 0 -l 60 -t 5 -ho stats.txt
tpython scripts/visualizer_supervised.py /media/1tb/Izzy/nets/net6_11-27-2016_19h04m23s.ckpt -n ChJon

tpython scripts/train_izzy_net.py -n ChJon -s -nn -f 0 -l 60 -t 5 -ho stats.txt
tpython scripts/visualizer_supervised.py /media/1tb/Izzy/nets/net6_11-27-2016_19h04m23s.ckpt -n ChJon

tpython scripts/train_izzy_net.py -n ChJon -s -nn -f 0 -l 60 -t 5 -ho stats.txt
tpython scripts/visualizer_supervised.py /media/1tb/Izzy/nets/net6_11-27-2016_19h04m23s.ckpt -n ChJon

tpython scripts/train_izzy_net.py -n ChJon -s -nn -f 0 -l 60 -t 5 -ho stats.txt
tpython scripts/visualizer_supervised.py /media/1tb/Izzy/nets/net6_11-27-2016_19h04m23s.ckpt -n ChJon
