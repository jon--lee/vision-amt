from random import shuffle
import numpy as np
def parse(filename, stop=-1):
    """
    Parse the file which is inteded to contain lines in the form of
    "path/to/image/file.jpg label1 label2 label3 label4" with no quotes.
    
    Returns a list of tuples where the first el is the string path
    and the second element is a list of float labels
    """
    f = open(filename, 'r')
    data = []
    lines = [ x for x in f ]
    shuffle(lines)
    
    for i, line in enumerate(lines):
        split = line.split(' ')
        path = split[0]
        labels = np.array([ float(x) for x in split[1:] ])
        data.append((path, labels))
        if (not stop < 0) and i > stop:
            break
    return data

