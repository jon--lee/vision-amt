from Net.tensor import net3
from Net.tensor import net4
from Net.tensor import net6,net8
from Net.tensor import inputdata
from options import AMTOptions

data = inputdata.AMTData(AMTOptions.train_file, AMTOptions.test_file,channels=3)
net = net6.NetSix()
path = '/media/1tb/Izzy/nets/net6_05-04-2016_15h59m29s.ckpt'
#path = '/media/1tb/Izzy/nets/net6_05-04-2016_14h10m59s.ckpt'
# net.optimize(200,data,path = path, batch_size=200)
net.optimize(500,data, batch_size=200)
