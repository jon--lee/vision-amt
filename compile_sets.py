from options import AMTOptions
import random

def compile():
    train_path = AMTOptions.train_file
    test_path = AMTOptions.test_file
    deltas_path = AMTOptions.deltas_file

    print "Moving deltas from " + deltas_path + " to train: " + train_path + " and test: " + test_path

    train_file = open(train_path, 'w+')
    test_file = open(test_path, 'w+')
    deltas_file = open(deltas_path, 'r')

    for line in deltas_file:            
        path = AMTOptions.originals_dir
        #path = AMTOptions.binaries_dir
        if random.random() > .2:
            train_file.write(path + line)
        else:
            test_file.write(path + line)
    
    print "Done moving images and labels."


if __name__ == '__main__':
        compile()
