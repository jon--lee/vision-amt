import matplotlib.pyplot as plt
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-n', '--net', required=True)
args = vars(ap.parse_args())
path = args['net'] + '/train.log'

f = open(path, 'r')

iterations = []
losses = []
i = 0
for line in f:
    if '[' in line:
        iter_string = line.split('Iteration ')[1]
        iteration = i + int(iter_string.split(' ]')[0])
        loss = float(iter_string.split('loss: ')[1])
        iterations.append(iteration)
        losses.append(loss)
    else:
        i = iterations[-1] + 1


plt.plot(iterations, losses, 'b')
plt.show()
f.close()
