import sys
sys.path.append('/home/annal/Izzy/vision_amt/')
from Net.tensor import net3
from Net.tensor import net4
from Net.tensor import net6,net6_c,net8
from Net.tensor import inputdata
from options import AMTOptions
import numpy as np

data = inputdata.AMTData(AMTOptions.train_file, AMTOptions.test_file,channels=3)
net = net6.NetSix()
# print np.array(data.next_train_batch(100)[1]).shape
# net = net6.NetSix()
path = '//media/1tb/Izzy/nets/net6_07-01-2016_16h15m32s.ckpt'
# path = '/media/1tb/Izzy/nets/net6_05-10-2016_16h51m48s.ckpt'
#path = '/media/1tb/Izzy/nets/net6_05-04-2016_14h10m59s.ckpt'
# net.optimize(200,data,path = path, batch_size=200)
net.optimize(300,data, batch_size=200)
