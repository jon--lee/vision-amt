from Net.tensor import net3
from Net.tensor import net4
from Net.tensor import inputdata
from options import AMTOptions

data = inputdata.AMTData(AMTOptions.train_file, AMTOptions.test_file,channels=3)
net = net4.NetFour()

net.optimize(500, data,batch_size=200)

