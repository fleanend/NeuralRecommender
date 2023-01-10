from unittest import TestCase
from neuralrecommender import model
import torch

class TestModel(TestCase):
    """Test Cases for Model primitives"""


    def test_local_kernel_sad(self):
        """Test local kernel for bogus inputs"""
        # dim should be > 2
        u, v = torch.zeros(1), torch.zeros(1)
        self.assertRaises(IndexError, model.local_kernel, u, v)
        u, v = torch.zeros((1,1)), torch.zeros(1)
        self.assertRaises(IndexError, model.local_kernel, u, v)

    def test_local_kernel_happy(self):
        """Test local kernel for apt range of inputs"""
        u = torch.zeros((1,1,1))
        v = torch.zeros((1,1,1))
        results = model.local_kernel(u,v)
        self.assertEqual(results.item(),1)
        u = torch.zeros((1,1,1))
        v = torch.ones((1,1,1))
        results = model.local_kernel(u,v)
        self.assertEqual(results.item(),0)
        u = 2*torch.ones((1,1,1))
        v = 2*torch.zeros((1,1,1))
        results = model.local_kernel(u,v)
        self.assertEqual(results.item(),0)
        u = 100000*torch.ones((1,1,1))
        v = -100000*torch.ones((1,1,1))
        results = model.local_kernel(u,v)
        self.assertEqual(results.item(),0)

    def test_kernel_layer_happy(self):
        """Test kernel layer with well formed input"""
        n_in, n_hid, n_dim, lambda_s, lambda_2 = 20, 30, 40, 1, 0.006
        n_user, n_movies = 100, 20
        kl = model.KernelLayer(n_in, n_hid, n_dim, lambda_s, lambda_2)
        matrix = torch.zeros((n_user,n_movies))
        output, _ = kl(matrix)
        self.assertEqual(output.shape[0], matrix.shape[0])
        self.assertEqual(output.shape[1], n_hid)

    def test_kernel_layer_sad(self):
        """Test kernel layer with badly formed input"""
        n_in, n_hid, n_dim, lambda_s, lambda_2 = 20, 30, 40, 1, 0.006
        n_user, n_movies = 100, 30
        kl = model.KernelLayer(n_in, n_hid, n_dim, lambda_s, lambda_2)
        matrix = torch.zeros((n_user,n_movies))
        with self.assertRaises(RuntimeError):
           kl(matrix)

    def test_kernel_net_happy(self):
        """Test kernel net with well formed input"""
        n_in, n_hid, n_dim, n_layers, lambda_s, lambda_2 = 20, 30, 40, 2, 1, 0.006
        n_user, n_movies = 100, 20
        kn = model.KernelNet(n_in, n_hid, n_dim, n_layers, lambda_s, lambda_2)
        self.assertEqual(len(kn.layers), n_layers+1)

        matrix = torch.zeros((n_user,n_movies))
        output, _ = kn(matrix)
        self.assertEqual(output.shape[0], matrix.shape[0])
        self.assertEqual(output.shape[1], n_movies)

    def test_kernel_net_sad(self):
        """Test kernel net with badly formed input"""
        n_in, n_hid, n_dim, n_layers, lambda_s, lambda_2 = 20, 30, 40, 2, 1, 0.006
        n_user, n_movies = 100, 60
        kn = model.KernelNet(n_in, n_hid, n_dim, n_layers, lambda_s, lambda_2)
        matrix = torch.zeros((n_user,n_movies))
        with self.assertRaises(RuntimeError):
           kn(matrix)

    def test_complete_net_happy(self):
        """Test complete net with well formed input"""
        n_in, n_hid, n_dim, n_layers, lambda_s, lambda_2 = 20, 30, 40, 2, 1, 0.006
        n_user, n_movies = 100, 20
        kn = model.KernelNet(n_in, n_hid, n_dim, n_layers, lambda_s, lambda_2)
        cn = model.CompleteNet(kn, n_user, n_movies, n_hid, n_dim, n_layers, lambda_s, lambda_2, 3, 1)

        matrix = torch.zeros((n_user, n_movies))
        output, _ = cn(matrix)
        self.assertEqual(output.shape[0], matrix.shape[0])
        self.assertEqual(output.shape[1], n_movies)

    def test_complete_net_sad(self):
        """Test complete net with badly formed input"""
        n_in, n_hid, n_dim, n_layers, lambda_s, lambda_2 = 20, 30, 40, 2, 1, 0.006
        n_user, n_movies = 100, 60
        kn = model.KernelNet(n_in, n_hid, n_dim, n_layers, lambda_s, lambda_2)
        cn = model.CompleteNet(kn, n_user, n_movies, n_hid, n_dim, n_layers, lambda_s, lambda_2, 3, 1)
        matrix = torch.zeros((n_user, n_movies))
        with self.assertRaises(RuntimeError):
           cn(matrix)

    def test_loss(self):
        """Test loss results"""
        n_in, n_hid, n_dim, n_layers, lambda_s, lambda_2 = 20, 30, 40, 2, 1, 0.006
        n_user, n_movies = 100, 20
        kn = model.KernelNet(n_in, n_hid, n_dim, n_layers, lambda_s, lambda_2)
        cn = model.CompleteNet(kn, n_user, n_movies, n_hid, n_dim, n_layers, lambda_s, lambda_2, 3, 1)

        matrix = torch.zeros((n_user, n_movies))
        mask = torch.ones_like(matrix)
        output, reg = cn(matrix)
        loss = model.Loss()(mask, reg, mask, matrix)
        self.assertAlmostEqual(loss.item(), (reg+1).item())