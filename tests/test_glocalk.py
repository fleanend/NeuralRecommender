from unittest import TestCase
from neuralrecommender import glocalk
import numpy as np
import unittest.mock
import io
import os

class TestInterface(TestCase):
    """Test Cases for training/inference interface"""

    def setUp(self) -> None:
        self.GL = glocalk.GlocalK()

    def test_get_lds_happy(self):
        """Test function to automagically get three hyperparamateres based on the matrix to reconstruct"""
        matrix = np.identity(100)
        s = matrix.sum()/(matrix.shape[0]*matrix.shape[1])

        # All is defined
        lambda_2, lambda_s, dot_scale = 20, 0.006, 1
        lambda_2_o, lambda_s_o, dot_scale_o = self.GL.get_lds(lambda_2, lambda_s, dot_scale, matrix)
        self.assertEqual([lambda_2, lambda_s, dot_scale],[lambda_2_o, lambda_s_o, dot_scale_o])

        # lambda_2 missing
        lambda_2, lambda_s, dot_scale = None, 0.006, 1
        lambda_2_o, lambda_s_o, dot_scale_o = self.GL.get_lds(lambda_2, lambda_s, dot_scale, matrix)
        self.assertEqual([lambda_s, dot_scale],[lambda_s_o, dot_scale_o])
        self.assertAlmostEqual((dot_scale_o*lambda_2_o/lambda_s_o)/s, 55_000)

        # lambda_s missing
        lambda_2, lambda_s, dot_scale = 20, None, 1
        lambda_2_o, lambda_s_o, dot_scale_o = self.GL.get_lds(lambda_2, lambda_s, dot_scale, matrix)
        self.assertEqual([lambda_2, dot_scale],[lambda_2_o, dot_scale_o])
        self.assertAlmostEqual((dot_scale_o*lambda_2_o/lambda_s_o)/s, 55_000)

        # dot_scale missing
        lambda_2, lambda_s, dot_scale = 20, 0.006, None
        lambda_2_o, lambda_s_o, dot_scale_o = self.GL.get_lds(lambda_2, lambda_s, dot_scale, matrix)
        self.assertEqual([lambda_2, lambda_s],[lambda_2_o, lambda_s_o])
        self.assertAlmostEqual((dot_scale_o*lambda_2_o/lambda_s_o)/s, 55_000)

        # lambda_2, lambda_s missing
        lambda_2, lambda_s, dot_scale = None, None, 1
        lambda_2_o, lambda_s_o, dot_scale_o = self.GL.get_lds(lambda_2, lambda_s, dot_scale, matrix)
        self.assertEqual(dot_scale, dot_scale_o)
        self.assertAlmostEqual((dot_scale_o*lambda_2_o/lambda_s_o)/s, 55_000)

        # lambda_2, dot_scale missing
        lambda_2, lambda_s, dot_scale = None, 0.006, None
        lambda_2_o, lambda_s_o, dot_scale_o = self.GL.get_lds(lambda_2, lambda_s, dot_scale, matrix)
        self.assertEqual(lambda_s, lambda_s_o)
        self.assertAlmostEqual((dot_scale_o*lambda_2_o/lambda_s_o)/s, 55_000)

        # lambda_s, dot_scale missing
        lambda_2, lambda_s, dot_scale = 20, None, None
        lambda_2_o, lambda_s_o, dot_scale_o = self.GL.get_lds(lambda_2, lambda_s, dot_scale, matrix)
        self.assertEqual(lambda_2, lambda_2_o)
        self.assertAlmostEqual((dot_scale_o*lambda_2_o/lambda_s_o)/s, 55_000)

        # Nothing is defined
        lambda_2, lambda_s, dot_scale = None, None, None
        lambda_2_o, lambda_s_o, dot_scale_o = self.GL.get_lds(lambda_2, lambda_s, dot_scale, matrix)
        self.assertAlmostEqual((dot_scale_o*lambda_2_o/lambda_s_o)/s, 55_000)


    def test_lds_sad(self):
        """Test zero matrix error handling for hyperparameter search"""
        matrix = np.zeros((100,100))
        lambda_2, lambda_s, dot_scale = None, None, None
        with self.assertRaises(ValueError) as ctx:
            lambda_2_o, lambda_s_o, dot_scale_o = self.GL.get_lds(lambda_2, lambda_s, dot_scale, matrix)
        self.assertEqual("Matrix must have at least one rating.", str(ctx.exception))

    def test_predict_happy(self):
        """Test prediction with proper index"""

        train_mask = np.ones((5,7))
        val_mask = np.zeros((5,7))
        train_mask[2,2] = 0
        val_mask[2,2] = 1
        train_matrix = 5*train_mask
        val_matrix = 5*val_mask
        self.GL.train_mask=train_mask
        self.GL.val_mask=val_mask
        self.GL.train_matrix=train_matrix
        self.GL.val_matrix=val_matrix
        self.GL.fit(train_matrix+val_matrix, max_epoch_p=1, max_epoch_f=1)
        prediction = self.GL.predict(0)
        self.assertEqual(len(prediction.shape),1)
        self.assertEqual(prediction.shape[0],5)

    def test_predict_before_fit(self):
        """Test calling predict before having fitted a model"""
        with self.assertRaises(IndexError) as ctx:
            self.GL.predict(0)
        self.assertEqual("Fit must be called before predicting.", str(ctx.exception))

    def test_fit_train_max_epochs(self):
        """Test fit and train function running smoothly to max epochs"""

        # model not yet crated
        with self.assertRaises(AttributeError) as ctx:
            _ = self.GL.model

        matrix = np.identity(100)
        matrix[2,2] = 0
        metrics = self.GL.fit(matrix, max_epoch_f=1, max_epoch_p=1, verbose=2, print_each=1)
        self.assertEqual(metrics['epochs_p'],1)
        self.assertEqual(metrics['epochs_f'],1)

        # model exists
        self.assertTrue(self.GL.model)
        
        metrics = self.GL.fit(matrix, max_epoch_f=2, max_epoch_p=2, verbose=2, print_each=1)
        self.assertEqual(metrics['epochs_p'],2)
        self.assertEqual(metrics['epochs_f'],2)

    def test_fit_train_patience(self):
        """Test fit and train function running smoothly to patience"""

        # model not yet created
        with self.assertRaises(AttributeError) as ctx:
            _ = self.GL.model

        matrix = np.identity(100)
        matrix[2,2] = 0
        metrics = self.GL.fit(matrix, max_epoch_f=1000, max_epoch_p=1000, tol_f=10, tol_p=10, patience_f=0, patience_p=0, verbose=2, print_each=1)

        self.assertEqual(metrics['epochs_p'],1)
        self.assertEqual(metrics['epochs_f'],1)

        # model exists
        self.assertTrue(self.GL.model)

        metrics = self.GL.fit(matrix, max_epoch_f=1000, max_epoch_p=1000, tol_f=10, tol_p=10, patience_f=1, patience_p=1, verbose=2, print_each=1)
        self.assertEqual(metrics['epochs_p'],2)
        self.assertEqual(metrics['epochs_f'],2)

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_fit_train_verbose(self, mock_stdout):
        """Test fit and train function verbosity > 1 output"""

        # model not yet crated
        with self.assertRaises(AttributeError) as ctx:
            _ = self.GL.model

        matrix = np.identity(100)
        matrix[2,2] = 0
        self.GL.fit(matrix, max_epoch_f=1000, max_epoch_p=1000, tol_f=10, tol_p=10, patience_f=0, patience_p=0, verbose=2, print_each=1)

        self.assertEqual(len(mock_stdout.getvalue().split('\n')), 13)

        # model exists
        self.assertTrue(self.GL.model)
        
        self.GL.fit(matrix, max_epoch_f=1000, max_epoch_p=1000, tol_f=10, tol_p=10, patience_f=1, patience_p=1, verbose=2, print_each=1)
        self.assertEqual(len(mock_stdout.getvalue().split('\n')), 37)

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_fit_train_less_verbose(self, mock_stdout):
        """Test fit and train function verbosity == 1 output"""

        # model not yet crated
        with self.assertRaises(AttributeError) as ctx:
            _ = self.GL.model

        matrix = np.identity(100)
        matrix[2,2] = 0
        self.GL.fit(matrix, max_epoch_f=1000, max_epoch_p=1000, tol_f=10, tol_p=10, patience_f=0, patience_p=0, verbose=1, print_each=1)

        self.assertEqual(len(mock_stdout.getvalue().split('\n')), 3)


    def test_fit_train_new_folder(self):
        """Test fit and train function creating new folder"""

        # model not yet created
        with self.assertRaises(AttributeError) as ctx:
            _ = self.GL.model

        path = "tests/models/"
        self.assertFalse(os.path.exists(path))

        matrix = np.identity(100)
        metrics = self.GL.fit(matrix, max_epoch_f=1, max_epoch_p=1, tol_f=10, tol_p=10, patience_f=0, patience_p=0, verbose=2, print_each=1, save_folder=path)
        
        self.assertTrue(os.path.exists("tests/models/"))
        os.remove(os.path.join(path,"best_model_weights.pth"))
        os.rmdir(path)

    def test_load_model(self):
        """Test loading model"""

        # model not yet created
        with self.assertRaises(AttributeError) as ctx:
            _ = self.GL.model

        matrix = np.ones((5,7))
        self.GL.load(matrix, "tests/fixtures/best_model_weights.pth")

        self.assertTrue(self.GL.model)