import sys, os
sys.path.append('/home/annal/Izzy/vision_amt/')
from Net.tensor import net3
from Net.tensor import net4
from Net.tensor import net6,net6_c,net8
from Net.tensor import inputdata
from options import AMTOptions
import numpy as np, argparse
from scripts import compile_supervisor, merge_supervised

def copy_over(infile, outfile, first, last):
    lines = infile.readlines()
    rol_num = lambda x: int(x[x.find('_rollout') + len('_rollout'):x.find('_rollout') + len('_rollout') + 1])
    for line in lines:
    	if rol_num(line) >= first and rol_num(line) <= last: 
    		print rol_num(line), first, last
    		outfile.write(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--DAgger", help="trains a dagger net",
                    action="store_true")
    parser.add_argument("-s", "--supervisor", help="Runs in limited experiment mode, requires other flags",
                    action="store_true")
    parser.add_argument("-n", "--name", type=str,
                        help="run experiment, will prompt for name")
    parser.add_argument("-f", "--first", type=int,
                        help="enter the starting value of rollouts to be used for training")
    parser.add_argument("-l", "--last", type=int,
                        help="enter the last value of the rollouts to be used for training")
    args = parser.parse_args()
    if args.name is not None:
        person = args.name 
    else:
        person = None

    if args.first is not None:
        first = args.first
    else:
        print "please enter a first value with -f"
        sys.exit()
    if args.last is not None:
        last = args.last
    else:
        print "please enter a last value with -l (not inclusive)"
        sys.exit()

    sup = True
    if args.DAgger:
        sup = False
    elif args.supervisor:
        sup = True
    else:
        print "specify type"
        sys.exit()
    outfile = open(AMTOptions.amt_dir + 'deltas.txt', 'w+')
    if sup:
        failure = merge_supervised.load_rollouts(False, False, (first,last-1), (0,-1), outfile, name = person)
        if failure:
            print "did not have the sufficient rollouts specified"
        outfile.close()
    else:
        failure = merge_supervised.load_rollouts(False, False, (0,20), (0,-1), outfile, name = person)
        if failure:
            print "did not have the sufficient rollouts specified... Do you have at least 20 supervised rollouts"
            sys.exit()
        f = []
        for (dirpath, dirnames, filenames) in os.walk(AMTOptions.rollouts_dir + person + "_rollouts/"):
            f.extend(filenames)
        for filename in f:
            read_path = AMTOptions.rollouts_dir + person + "_rollouts/" + filename
            if read_path.find("retroactive_feedback") != -1:
            	print filename
                index = read_path.find("retroactive_feedback") + len("retroactive_feedback")
                start = int(read_path[index:index + 1])
                end = int(read_path[index+2:index + 3])
                print start, end
                if start < last and first <= end:
                    infile = open(read_path, 'r')

                    copy_over(infile, outfile, max(start, first), min(last-1, end))
                    infile.close()
            

        
        outfile.close()
    compile_supervisor.compile_reg()

    data = inputdata.AMTData(AMTOptions.train_file, AMTOptions.test_file,channels=3)
    net = net6.NetSix()
    # net.optimize(300,data, batch_size=200)

    # path = '//media/1tb/Izzy/nets/net6_07-01-2016_16h15m32s.ckpt'
    # net.optimize(200,data,path = path, batch_size=200)
