from Net.tensor import net3, net2, net4, inputdata
from options import AMTOptions
import tensorflow as tf



if __name__ == '__main__':
    data = inputdata.AMTData(AMTOptions.train_file, AMTOptions.test_file)
    nets = [net2.NetTwo(), net3.NetThree(), net4.NetFour()]

    for net in nets:
        net.optimize(200, data, batch_size=200)

