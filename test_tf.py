from Net.tensor import net3
from Net.tensor import net4
from Net.tensor import net6
from Net.tensor import inputdata
from options import AMTOptions

data = inputdata.AMTData(AMTOptions.train_file, AMTOptions.test_file,channels=3)
net = net6.NetSix()

net.optimize(500, data,path = '/media/1tb/Izzy/nets/net6_02-19-2016_17h14m04s.ckpt',batch_size=200)

