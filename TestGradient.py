import unittest
import GradientDescend
import numpy as np


class TestGradient(unittest.TestCase):
    
    def test_1dim_1iter(self):
        def test_fun(x):
            return 1/2 * x**2
        
        def test_grad(x):
            return x

        x0 = 1

        res = GradientDescend.GradientDescend(test_fun, test_grad, x0, 0.01,1,1)

        self.assertEqual(res.iat[-1,0], 0)

    def test_1dim_2(self):
        def test_fun(x):
            return np.exp(-x)
        
        def test_grad(x):
            return -np.exp(-x)
        
        x0 = 1

        res = GradientDescend.GradientDescend(test_fun,test_grad,x0,0,1,10000)

        self.assertLessEqual(res.iat[10000,2], 0.0001)


if __name__ == '__main__':
    unittest.main()