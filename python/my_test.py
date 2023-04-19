import oneflow as flow
import numpy as np
import torch

N = 1024    #N不是1024就不可以
x = np.random.randint(0,100,size=(N,N))
print('x = {}'.format(x))
b = np.random.randint(0,100,size=N)
print('b = {}'.format(b))
w = np.random.randint(0,100,size=(N,N))

x_tensor = flow.Tensor(x).to(dtype=flow.int,device="cuda")
w_tensor = flow.Tensor(w).to(dtype=flow.int,device="cuda")
b_tensor = flow.Tensor(b).to(dtype=flow.int,device="cuda")
print(b_tensor.size())

flow_y = flow._C.my_tiled_aggregate(x_tensor,w_tensor,b_tensor).detach().cpu()
print('矩阵X:{0}\n矩阵W:{1}\n向量b:{2}\n结果Y:{3}\n'.format(x_tensor,w_tensor,b_tensor,flow_y))

#verify result
torch_x = torch.from_numpy(x).to(dtype=torch.int)
torch_w = torch.from_numpy(w).to(dtype=torch.int)
torch_b = torch.from_numpy(b).to(dtype=torch.int)

torch_tmp_b = np.expand_dims(b,0).repeat(N,0)

print('torch_tmp_b = {}'.format(torch_tmp_b))

torch_tmp_numpy = torch.mm(torch_w,torch_x).numpy()
print("torch_tmp_numpy = :",torch_tmp_numpy)
torch_y = torch_tmp_numpy + torch_tmp_b
print("torch_y = :",torch_y)
print("flow_y",flow_y.numpy())
result = np.allclose(flow_y, torch_y, 1e-05, 1e-05)
print(result)
