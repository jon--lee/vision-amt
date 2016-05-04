from Net.tensor import net3
from Net.tensor import net4
from Net.tensor import net6,net8
from Net.tensor import inputdata
from options import AMTOptions

data = inputdata.AMTData(AMTOptions.train_file, AMTOptions.test_file,channels=3)
net = net6.NetSix()
# path = '/media/1tb/Izzy/nets/net6_03-27-2016_11h16m08s.ckpt'
net.optimize(4000,data, batch_size=200)

