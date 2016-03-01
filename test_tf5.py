from Net.tensor import inputdata, net5, net6, net7, net8, net9, net10, net11
from options import AMTOptions
from tensorflow.python.framework import ops
import transfer_weights
data = inputdata.AMTData(AMTOptions.train_file, AMTOptions.test_file,channels=3)
#net = net5.NetFive()
#path = AMTOptions.tf_dir + 'net5/net5_02-14-2016_18h42m49s.ckpt'
#net.optimize(200, data=data, path=path, batch_size=100)

#path = AMTOptions.tf_dir + 'net6/net6_02-15-2016_13h48m35s.ckpt' 
#net = net6.NetSix()
#net.optimize(400, data=data, batch_size=100)

#path = AMTOptions.tf_dir + 'net7/'
#net = net7.NetSeven()
#net.optimize(400, data=data, batch_size=150)
"""
source_path = '/media/1tb/Izzy/nets/net6_02-24-2016_19h32m42s.ckpt'
target_path = '/media/1tb/Izzy/nets/net10_02-26-2016_11h04m06s.ckpt'

weights, biases = transfer_weights.get_conv1_variables(net6.NetSix, source_path)
ops.reset_default_graph()

new_net = net10.NetTen()
sess = new_net.load(var_path=target_path)

transfer_weights.assign_variables(sess, new_net.w_conv1, new_net.b_conv1, weights, biases)
save_path = new_net.save(sess, save_path='merged_net10.ckpt')

ops.reset_default_graph()

net = net10.NetTen()
net.optimize(500, data=data, path=save_path, batch_size=100)
"""
#path = AMTOptions.nets_dir + 'net8_02-26-2016_12h45m28s.ckpt'
#path = '/media/1tb/Izzy/nets/net8_02-26-2016_13h51m21s.ckpt'
#net = net8.NetEight()
#net.optimize(900, data=data, path=path, batch_size=80)

#net = net10.NetTen()
#net.optimize(900, data=data, batch_size=100)

#path = '/media/1tb/Izzy/nets/net6_02-26-2016_17h58m15s.ckpt'
#net = net6.NetSix()
#net.optimize(500, data=data, path=path, batch_size=100)

net  = net11.NetEleven()
net.optimize(2000, data=data, batch_size=150)



