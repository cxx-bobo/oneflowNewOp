# import oneflow as flow
import numpy as np
import torch

crow_indices = torch.tensor([0,2,3,3,3,6,6,7])
col_indices = torch.tensor([0,2,2,2,3,4,3])
values = torch.tensor([8,2,5,7,1,2,9])
csr = torch.sparse_csr_tensor(crow_indices, col_indices, values, dtype=torch.float64)
print("csr = ",csr)
print("csr.shape() = ",csr.shape())
v = csr.col_indices()
print("v = ",v)
v_num = v.size(0)
print("v_num = ",v_num)
e = csr.crow_indices()
print("e = ",e)
e_num=e[csr.crow_indices().size(0)-1]
print("e_num = ",e_num.item())
