import sys, os
sys.path.append('/home/annal/Izzy/vision_amt/')
from Net.tensor import net3
from Net.tensor import net4
from Net.tensor import net6,net6_c,net8
from Net.tensor import inputdata
from options import AMTOptions
import numpy as np, argparse
from scripts import compile_supervisor, merge_supervised

def copy_over(infile, outfile, first, last, frame_first=0, frame_last=100):
    lines = infile.readlines()
    rol_num = lambda x: int(x[x.find('_rollout') + len('_rollout'):x.find('_frame_')])
    frame_num = lambda x: int(x[x.find('_frame_') + len('_frame_'):x.find('.jpg')])
    for line in lines:
        # print line, rol_num(line), first, last
        if rol_num(line) >= first and rol_num(line) <= last:
            if frame_num(line) >= frame_first and frame_num(line) <= frame_last:
                outfile.write(line)
                outfile.write(line)
            else:
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
    parser.add_argument("-i", "--initial", type=int,
                        help="enter the initial value of frames to be used for training")
    parser.add_argument("-e", "--end", type=int,
                        help="enter the ending value of the frames to be used for training")
    parser.add_argument("-sm", "--smooth", type=int,
                        help="determines if a butterworth filter should be applied over the data, of order N")
    parser.add_argument("-t", "--test_size", type=int,
                        help="determines the size of the test set")
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
    test_size = args.test_size if args.test_size is not None else 10
    initial = args.initial if args.initial is not None else 0
    end = args.end if args.end is not None else 100

    sup = True
    smooth = args.smooth
    if args.DAgger:
        sup = False
    elif args.supervisor:
        sup = True
    else:
        print "specify type"
        sys.exit()
    outfile = open(AMTOptions.amt_dir + 'deltas.txt', 'w+')
    if sup:
        failure = merge_supervised.load_rollouts(False, False, (first,last), (initial,end), outfile, name = person)
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
            if read_path.find("retroactive_feedback") != -1 and read_path.find("~") == -1:
                print filename
                paths = filename.split('_')
                index = len('feedback')
                end_index = paths[2].find(".txt")
                start = int(paths[1][index:])
                end = int(paths[2][:end_index])
                print start, end
                if start < last and first <= end:
                    infile = open(read_path, 'r')

                    copy_over(infile, outfile, max(start, first), min(last, end), frame_first =initial, frame_last=end)
                    infile.close()
            

        
        outfile.close()
    skipped = compile_supervisor.compile_reg(smooth=smooth, num=test_size)

    data = inputdata.AMTData(AMTOptions.train_file, AMTOptions.test_file,channels=3)
    net = net6.NetSix()
    path = '/media/1tb/Izzy/nets/net6_10-10-2016_13h57m13s.ckpt'
    net_name = net.optimize(300,data, path=path, batch_size=200)
    outf = open(AMTOptions.amt_dir + 'last_net.txt', 'w')
    outf.write(net_name)
    outf.close()
    # path = '/media/1tb/Izzy/nets/net6_08-19-2016_12h28m11s.ckpt'
    # net.optimize(700,data,path = path, batch_size=200)
    outf = open(AMTOptions.amt_dir + 'testing_outputs.txt', 'a+')
    outf.write(person + ', is supervised data: ' + str(sup) + ', range: ' + str(first) +  ', ' +str(last) + '\n')
    outf.write(str(skipped) + '\n')
    outf.write(net_name + '\n')
    outf.close()
