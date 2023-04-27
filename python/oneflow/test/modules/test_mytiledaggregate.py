"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import unittest

import numpy as np
from oneflow.test_utils.automated_test_util import *

import oneflow as flow
import torch

@flow.unittest.skip_unless_1n1d()
class TestMyTiledAggregate(flow.unittest.TestCase):  
    def test_flow_tensor_matmul_with_random_int_data(test_case):
        
        # initial data
        N = np.random.randint(1, 300)
        print('N = {}'.format(N))
        x = np.random.randint(0, 100, size=(N, N))
        #print('x = {}'.format(x))
        w = np.random.randint(0, 100, size=(N, N))
        #print('w = {}'.format(w))
        b = np.random.randint(0, 100, size=N)
        #print('b = {}'.format(b))


        #compute in pytorch
        torch_x = torch.from_numpy(x).to(dtype=torch.float).to('cuda')
        torch_w = torch.from_numpy(w).to(dtype=torch.float).to('cuda')
        torch_tmp_b = np.expand_dims(b,0).repeat(N,0)
        torch_tmp_b= torch.from_numpy(b).to(dtype=torch.float).to('cuda')
        #print('torch_tmp_b = {}'.format(torch_tmp_b))
        torch_tmp_numpy = torch.mm(torch_w,torch_x)
        #print("torch_tmp_numpy = :",torch_tmp_numpy)
        torch_y = torch.add(torch_tmp_numpy, torch_tmp_b).detach().cpu()
        #print("torch_output_numpy = :",torch_y)
        

        #compute in oneflow(gpu)
        print("Compute in oneflow(GPU)...")
        flow_x_gpu = flow.tensor(x).to(dtype=flow.int,device="cuda")
        flow_w_gpu = flow.tensor(w).to(dtype=flow.int,device="cuda")
        flow_b_gpu = flow.tensor(b).to(dtype=flow.int,device="cuda")
        flow_y_gpu = flow._C.my_tiled_aggregate(flow_x_gpu, flow_w_gpu, flow_b_gpu).detach().cpu().numpy()
        #print('flow_y_gpu = {}'.format(flow_y_gpu))
        #verify the result
        print("Compare the results of onflow(GPU) and torch...")
        test_case.assertTrue( np.allclose(flow_y_gpu, torch_y.numpy(), 1e-05, 1e-05))
        print("Correct!")

        #compute in oneflow(cpu)
        #cpu kernel下 若数据类型为int，则必须指定为int64，否则很有可能出现数据溢出的情况
        #但gpu kernel下 如数据类型为int，则不用指定为int64，为什么？
        print("Compute in oneflow(CPU)...")
        flow_x_cpu = flow.tensor(x).to(dtype=flow.int64,device="cpu")
        flow_w_cpu = flow.tensor(w).to(dtype=flow.int64,device="cpu")
        flow_b_cpu = flow.tensor(b).to(dtype=flow.int64,device="cpu")
        flow_y_cpu = flow._C.my_tiled_aggregate(flow_x_cpu, flow_w_cpu, flow_b_cpu).numpy()
        #print('flow_y_cpu = {}'.format(flow_y_cpu))
        #verify the result
        print("Compare the results of onflow(CPU) and torch...")
        test_case.assertTrue( np.allclose(flow_y_cpu, torch_y.numpy(), 1e-05, 1e-05))
        print("Correct!")


        

if __name__ == "__main__":
    unittest.main()