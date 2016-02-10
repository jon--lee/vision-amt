from Net.tensor import net3
from Net.tensor import inputdata
from options import AMTOptions

data = inputdata.AMTData(AMTOptions.train_file, AMTOptions.test_file)
net = net3.NetThree()

net.optimize(400, data, batch_size=200)

