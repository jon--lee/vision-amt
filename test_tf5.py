from Net.tensor import inputdata, net5, net6, net7, net8
from options import AMTOptions

data = inputdata.AMTData(AMTOptions.train_file, AMTOptions.test_file,channels=3)
#net = net5.NetFive()
#path = AMTOptions.tf_dir + 'net5/net5_02-14-2016_18h42m49s.ckpt'
#net.optimize(200, data=data, path=path, batch_size=100)

#path = AMTOptions.tf_dir + 'net6/net6_02-15-2016_13h48m35s.ckpt' 
net = net6.NetSix()
net.optimize(400, data=data, batch_size=100)

#path = AMTOptions.tf_dir + 'net7/'
#net = net7.NetSeven()
#net.optimize(400, data=data, batch_size=150)

#net = net8.NetEight()
#net.optimize(400, data=data, batch_size=150)
