import oneflow as flow
import numpy as np
import torch



# N = np.random.randint(0,512)   
N = 33
print('N = ', N)
w = np.random.randint(0,100,size=(N,N))
print('w = {}'.format(w))
x = np.random.randint(0,100,size=(N,N))
print('x = {}'.format(x))
b = np.random.randint(0,100,size=N)
print('b = {}'.format(b))


# # compute in oneflow（GPU运行）
# x_tensor = flow.Tensor(x).to(dtype=flow.int,device="cuda")
# w_tensor = flow.Tensor(w).to(dtype=flow.int,device="cuda")
# b_tensor = flow.Tensor(b).to(dtype=flow.int,device="cuda")
# flow_y = flow._C.my_tiled_aggregate(x_tensor,w_tensor,b_tensor).detach().cpu()

#compute in oneflow（cpu运行）
x_tensor = flow.Tensor(x).to(dtype=flow.int32)
w_tensor = flow.Tensor(w).to(dtype=flow.int32)
b_tensor = flow.Tensor(b).to(dtype=flow.int32)
flow_y = flow._C.my_tiled_aggregate(x_tensor,w_tensor,b_tensor)


#compute in pytorch
torch_x = torch.from_numpy(x).to(dtype=torch.float32).to('cuda')
torch_w = torch.from_numpy(w).to(dtype=torch.float32).to('cuda')
torch_tmp_b = np.expand_dims(b,0).repeat(N,0)
torch_tmp_b= torch.from_numpy(b).to(dtype=torch.float32).to('cuda')
print('torch_tmp_b = {}'.format(torch_tmp_b))
torch_tmp_numpy = torch.mm(torch_w,torch_x)
print("torch_tmp_numpy = :",torch_tmp_numpy)
torch_y = torch.add(torch_tmp_numpy, torch_tmp_b).detach().cpu()


#verify result
print("torch_y = :",torch_y)
print("flow_y",flow_y)
result = np.allclose(flow_y, torch_y, 1e-05, 1e-05)
print(result)
