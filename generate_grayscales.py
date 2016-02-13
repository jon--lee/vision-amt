from options import AMTOptions

"""
   NOT FINISHED
"""


if __name__ == "__main__":
    deltas_path = AMTOptions.deltas_file
    f = open(deltas_path, 'r')
    for line in f:
        split = line.split(' ')
        name = split[0]

