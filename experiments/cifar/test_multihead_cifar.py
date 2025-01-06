import unittest
import random

import torch

from multihead_cifar_classification import Adaptor, MultiHeadCNN

batch_size = 2
in_channels = 3
img_size = 112

class TestUniheadResnet18(unittest.TestCase):
    model_name = "resnet18"
    output_dim = 512
    def test_resnet18_str_num_heads_none(self):
        x = torch.randn(batch_size, in_channels, img_size, img_size)
        model = MultiHeadCNN(self.model_name, 10)
        y = model(x).squeeze()
        self.assertEqual(tuple(y.shape), (batch_size, self.output_dim))

    def test_resnet18_list_num_heads_none(self):
        x = torch.randn(batch_size, in_channels, img_size, img_size)
        model = MultiHeadCNN([self.model_name], 10)
        y = model(x).squeeze()
        self.assertEqual(tuple(y.shape), (batch_size, self.output_dim))

    def test_resnet18_str_num_heads_1(self):
        x = torch.randn(batch_size, in_channels, img_size, img_size)
        model = MultiHeadCNN(self.model_name, 10, 1)
        y = model(x).squeeze()
        self.assertEqual(tuple(y.shape), (batch_size, self.output_dim))

    def test_resnet18_list_num_heads_1(self):
        x = torch.randn(batch_size, in_channels, img_size, img_size)
        model = MultiHeadCNN([self.model_name], 10, 1)
        y = model(x).squeeze()
        self.assertEqual(tuple(y.shape), (batch_size, self.output_dim))

class TestUniheadResnet34(unittest.TestCase):
    model_name = "resnet34"
    output_dim = 512
    def test_resnet18_str_num_heads_none(self):
        x = torch.randn(batch_size, in_channels, img_size, img_size)
        model = MultiHeadCNN(self.model_name, 10)
        y = model(x).squeeze()
        self.assertEqual(tuple(y.shape), (batch_size, self.output_dim))

    def test_resnet18_list_num_heads_none(self):
        x = torch.randn(batch_size, in_channels, img_size, img_size)
        model = MultiHeadCNN([self.model_name], 10)
        y = model(x).squeeze()
        self.assertEqual(tuple(y.shape), (batch_size, self.output_dim))

    def test_resnet18_str_num_heads_1(self):
        x = torch.randn(batch_size, in_channels, img_size, img_size)
        model = MultiHeadCNN(self.model_name, 10, 1)
        y = model(x).squeeze()
        self.assertEqual(tuple(y.shape), (batch_size, self.output_dim))

    def test_resnet18_list_num_heads_1(self):
        x = torch.randn(batch_size, in_channels, img_size, img_size)
        model = MultiHeadCNN([self.model_name], 10, 1)
        y = model(x).squeeze()
        self.assertEqual(tuple(y.shape), (batch_size, self.output_dim))

class TestUniheadResnet50(unittest.TestCase):
    model_name = "resnet50"
    output_dim = 2048
    def test_resnet18_str_num_heads_none(self):
        x = torch.randn(batch_size, in_channels, img_size, img_size)
        model = MultiHeadCNN(self.model_name, 10)
        y = model(x).squeeze()
        self.assertEqual(tuple(y.shape), (batch_size, self.output_dim))

    def test_resnet18_list_num_heads_none(self):
        x = torch.randn(batch_size, in_channels, img_size, img_size)
        model = MultiHeadCNN([self.model_name], 10)
        y = model(x).squeeze()
        self.assertEqual(tuple(y.shape), (batch_size, self.output_dim))

    def test_resnet18_str_num_heads_1(self):
        x = torch.randn(batch_size, in_channels, img_size, img_size)
        model = MultiHeadCNN(self.model_name, 10, 1)
        y = model(x).squeeze()
        self.assertEqual(tuple(y.shape), (batch_size, self.output_dim))

    def test_resnet18_list_num_heads_1(self):
        x = torch.randn(batch_size, in_channels, img_size, img_size)
        model = MultiHeadCNN([self.model_name], 10, 1)
        y = model(x).squeeze()
        self.assertEqual(tuple(y.shape), (batch_size, self.output_dim))

class TestUniheadResnet101(unittest.TestCase):
    model_name = "resnet101"
    output_dim = 2048
    def test_resnet18_str_num_heads_none(self):
        x = torch.randn(batch_size, in_channels, img_size, img_size)
        model = MultiHeadCNN(self.model_name, 10)
        y = model(x).squeeze()
        self.assertEqual(tuple(y.shape), (batch_size, self.output_dim))

    def test_resnet18_list_num_heads_none(self):
        x = torch.randn(batch_size, in_channels, img_size, img_size)
        model = MultiHeadCNN([self.model_name], 10)
        y = model(x).squeeze()
        self.assertEqual(tuple(y.shape), (batch_size, self.output_dim))

    def test_resnet18_str_num_heads_1(self):
        x = torch.randn(batch_size, in_channels, img_size, img_size)
        model = MultiHeadCNN(self.model_name, 10, 1)
        y = model(x).squeeze()
        self.assertEqual(tuple(y.shape), (batch_size, self.output_dim))

    def test_resnet18_list_num_heads_1(self):
        x = torch.randn(batch_size, in_channels, img_size, img_size)
        model = MultiHeadCNN([self.model_name], 10, 1)
        y = model(x).squeeze()
        self.assertEqual(tuple(y.shape), (batch_size, self.output_dim))

class TestUniheadResnet152(unittest.TestCase):
    model_name = "resnet152"
    output_dim = 2048
    def test_resnet18_str_num_heads_none(self):
        x = torch.randn(batch_size, in_channels, img_size, img_size)
        model = MultiHeadCNN(self.model_name, 10)
        y = model(x).squeeze()
        self.assertEqual(tuple(y.shape), (batch_size, self.output_dim))

    def test_resnet18_list_num_heads_none(self):
        x = torch.randn(batch_size, in_channels, img_size, img_size)
        model = MultiHeadCNN([self.model_name], 10)
        y = model(x).squeeze()
        self.assertEqual(tuple(y.shape), (batch_size, self.output_dim))

    def test_resnet18_str_num_heads_1(self):
        x = torch.randn(batch_size, in_channels, img_size, img_size)
        model = MultiHeadCNN(self.model_name, 10, 1)
        y = model(x).squeeze()
        self.assertEqual(tuple(y.shape), (batch_size, self.output_dim))

    def test_resnet18_list_num_heads_1(self):
        x = torch.randn(batch_size, in_channels, img_size, img_size)
        model = MultiHeadCNN([self.model_name], 10, 1)
        y = model(x).squeeze()
        self.assertEqual(tuple(y.shape), (batch_size, self.output_dim))

class TestMultiheadResnet18(unittest.TestCase):
    model_names = "resnet18"
    output_dim = 512
    num_heads = 2
    def test_resnet18_str_num_heads_2(self):
        x = torch.randn(batch_size, in_channels, img_size, img_size)
        model = MultiHeadCNN(self.model_names, 10, self.num_heads)
        y = model(x).squeeze()
        self.assertEqual(tuple(y.shape), (batch_size, self.num_heads * self.output_dim))

    def test_resnet18_list_num_heads_none(self):
        x = torch.randn(batch_size, in_channels, img_size, img_size)
        model = MultiHeadCNN([self.model_names] * self.num_heads, 10)
        y = model(x).squeeze()
        self.assertEqual(tuple(y.shape), (batch_size, self.num_heads * self.output_dim))

    def test_resnet18_list_num_heads_1(self):
        x = torch.randn(batch_size, in_channels, img_size, img_size)
        model = MultiHeadCNN([self.model_names] * self.num_heads, 10, 1)
        y = model(x).squeeze()
        self.assertEqual(tuple(y.shape), (batch_size, self.num_heads * self.output_dim))

    def test_resnet18_list_num_heads_2(self):
        x = torch.randn(batch_size, in_channels, img_size, img_size)
        model = MultiHeadCNN([self.model_names] * self.num_heads, 10, 2)
        y = model(x).squeeze()
        self.assertEqual(tuple(y.shape), (batch_size, self.num_heads * self.output_dim))

    def test_resnet18_list_num_heads_3(self):
        x = torch.randn(batch_size, in_channels, img_size, img_size)
        model = MultiHeadCNN([self.model_names] * self.num_heads, 10, 3)
        y = model(x).squeeze()
        self.assertEqual(tuple(y.shape), (batch_size, self.num_heads * self.output_dim))

class TestMultiheadResnet18_34(unittest.TestCase):
    model_names = ["resnet18", "resnet34"]
    output_dim = 512 + 512
    def test_resnet18_34_str_num_heads_none(self):
        x = torch.randn(batch_size, in_channels, img_size, img_size)
        model = MultiHeadCNN(self.model_names, 10)
        y = model(x).squeeze()
        self.assertEqual(tuple(y.shape), (batch_size, self.output_dim))

class TestMultiheadResnet18_50(unittest.TestCase):
    model_names = ["resnet18", "resnet50"]
    output_dim = 512 + 2048
    def test_resnet18_50_str_num_heads_none(self):
        x = torch.randn(batch_size, in_channels, img_size, img_size)
        model = MultiHeadCNN(self.model_names, 10)
        y = model(x).squeeze()
        self.assertEqual(tuple(y.shape), (batch_size, self.output_dim))

class TestRandomResnetCombination(unittest.TestCase):
    def test_random_resnet_config(self):
        model_feat_dim_dict = {
            "resnet18": 512,
            "resnet34": 512,
            "resnet50": 2048,
            "resnet101": 2048,
            "resnet152": 2048
        }
        model_names = list()
        output_dim = 0
        num_models = random.randint(2, 5)
        for _ in range(num_models):
            model_name = random.choice(list(model_feat_dim_dict.keys()))
            model_names.append(model_name)
            output_dim += model_feat_dim_dict[model_name]
        x = torch.randn(batch_size, in_channels, img_size, img_size)
        model = MultiHeadCNN(model_names, 10)
        y = model(x).squeeze()
        self.assertEqual(tuple(y.shape), (batch_size, output_dim))

class TestCopyHeads(unittest.TestCase):
    def test_copy_heads_resnet18_x2(self):
        num_heads = 2
        model_name = "resnet18"
        model_names = [model_name] * num_heads
        model = MultiHeadCNN(model_names, 10, copy_heads=True)
        main_head_state_dict = model.heads[0].state_dict()
        other_heads_state_dicts = [head.state_dict() for head in model.heads[1:]]
        for param_name in main_head_state_dict.keys():
            for head_state_dict in other_heads_state_dicts:
                self.assertTrue(
                    torch.equal(
                        main_head_state_dict[param_name], 
                        head_state_dict[param_name]
                        )
                    )

    def test_copy_heads_resnet50_x3(self):
        num_heads = 3
        model_name = "resnet50"
        model_names = [model_name] * num_heads
        model = MultiHeadCNN(model_names, 10, copy_heads=True)
        main_head_state_dict = model.heads[0].state_dict()
        other_heads_state_dicts = [head.state_dict() for head in model.heads[1:]]
        for param_name in main_head_state_dict.keys():
            for head_state_dict in other_heads_state_dicts:
                self.assertTrue(
                    torch.equal(
                        main_head_state_dict[param_name], 
                        head_state_dict[param_name]
                        )
                    )


def make_random_resnet_suite():
    resnet_config_test = [TestRandomResnetCombination("test_random_resnet_config")] * 10
    return unittest.TestSuite(tests=resnet_config_test)

def make_copy_heads_suite():
    copy_heads_test = [
        TestCopyHeads("test_copy_heads_resnet18_x2"),
        TestCopyHeads("test_copy_heads_resnet50_x3")
        ]
    return unittest.TestSuite(tests=copy_heads_test)

if __name__ == "__main__":
    suite = make_copy_heads_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)