from options import AMTOptions
import random


def scale(deltas):
    deltas[0] = float(deltas[0])/0.2
    deltas[1] = float(deltas[1])/0.01
    deltas[2] = float(deltas[2])/0.005
    deltas[3] = float(deltas[3])/0.2
    return deltas


def compile():
    train_path = AMTOptions.train_file
    test_path = AMTOptions.test_file
    deltas_path = AMTOptions.deltas_file

    print "Moving deltas from " + deltas_path + " to train: " + train_path + " and test: " + test_path

    train_file = open(train_path, 'w+')
    test_file = open(test_path, 'w+')
    deltas_file = open(deltas_path, 'r')

    for line in deltas_file:            
        #path = AMTOptions.originals_dir
        path = AMTOptions.colors_dir 
        labels = line.split()
        deltas = scale(labels[1:5])
        line = labels[0]+" "+str(deltas[0])+" "+str(deltas[1])+" "+str(deltas[2])+" "+str(deltas[3])+"\n"
        if random.random() > .2:
            train_file.write(path + line)
        else:
            test_file.write(path + line)
    
    print "Done moving images and labels."


if __name__ == '__main__':
    compile()
