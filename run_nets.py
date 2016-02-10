from Net.tensor import net3, net2, inputdata
from options import AMTOptions



if __name__ == '__main__':
    net_classes = [net2.NetTwo, net3.NetThree]

    for net_class in nets:
        data = inputdata.AMTData(AMTOptions.train_file, AMTOptions.test_file)
        n = net_class()
        net.optimize(350, data, batch_size=200)
