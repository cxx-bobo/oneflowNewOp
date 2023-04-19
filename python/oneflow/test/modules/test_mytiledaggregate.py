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
# from collections import OrderedDict

import numpy as np
from oneflow.test_utils.automated_test_util import *

import oneflow as flow
# import oneflow.unittest
import torch

@flow.unittest.skip_unless_1n1d()
class TestMyTiledAggregate(flow.unittest.TestCase):
    # @autotest(check_graph=False,auto_backward=False)    #autotest的参数?
    def test_flow_tensor_matmul_with_random_int_data(test_case):
        #initial data
        # k = np.random.randint(1, 1024)
        k = 16
        print('k = {}'.format(k))
        b = np.random.randint(0, 100, size=k)
        print('b = {}'.format(b))
        x = np.random.randint(0, 100, size=(k, k))
        print('x = {}'.format(x))
        w = np.random.randint(0, 100, size=(k, k))
        print('w = {}'.format(w))

        b_tmp = np.expand_dims(b,0).repeat(k,0)
        print('b_tmp = {}'.format(b_tmp))

        #tiled_aggregate in pytorch
        torch_x = torch.from_numpy(x).to(dtype=torch.int)
        torch_w = torch.from_numpy(w).to(dtype=torch.int)
        torch_b = torch.from_numpy(b_tmp).to(dtype=torch.int)
        torch_tmp_numpy = torch_x.matmul(torch_w)
        print("torch_tmp_numpy = :",torch_tmp_numpy)
        

        torch_output_numpy = torch_tmp_numpy.add(torch_b)
        print("torch_output_numpy = :",torch_output_numpy)
        #tiled_aggregate in oneflow
        flow_x = flow.tensor(x).to(dtype=flow.int,device="cuda")
        flow_w = flow.tensor(w).to(dtype=flow.int,device="cuda")
        flow_b = flow.tensor(b).to(dtype=flow.int,device="cuda")

        

        flow_output_numpy = flow._C.my_tiled_aggregate(flow_x, flow_w, flow_b).detach().cpu().numpy()

        print('flow_output_numpy = {}'.format(flow_output_numpy))

        #verify the result
        test_case.assertTrue(
            np.allclose(flow_output_numpy, torch_output_numpy.numpy(), 1e-05, 1e-05)
        )

if __name__ == "__main__":
    unittest.main()