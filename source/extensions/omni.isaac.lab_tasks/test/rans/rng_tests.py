from omni.isaac.lab.app import AppLauncher, run_tests

# launch omniverse app
config = {"headless": True}
simulation_app = AppLauncher(config).app
import unittest
import torch
from omni.isaac.lab_tasks.rans.utils import PerEnvSeededRNG


class TestRandomNumberGenerator(unittest.TestCase):
    def __init__(self, methodName="runTest"):
        super().__init__(methodName)

    ############################################################
    # Test Seed Initialization
    ############################################################

    def test_set_seeds(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        self.assertTrue(torch.equal(pesrng_1.seeds_torch, torch.arange(1000, dtype=torch.int32, device="cuda")))

    ############################################################
    # Test Uniform Sampling
    ############################################################

    def test_uniform_randomness(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        output = pesrng_1.sample_uniform_torch(0.0, 1.0, 1000)
        test = torch.mean(torch.square(torch.diff(output, dim=0)))
        self.assertTrue(test > 0.0)

    def test_uniform_reproducibility_1D(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        pesrng_2 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_2.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        output_1 = pesrng_1.sample_uniform_torch(0.0, 1.0, 1)
        output_2 = pesrng_2.sample_uniform_torch(0.0, 1.0, (1,))
        self.assertTrue(torch.equal(output_1, output_2))

    def test_uniform_reproducibility_2D(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        pesrng_2 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_2.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        output_1 = pesrng_1.sample_uniform_torch(0.0, 1.0, 10)
        output_2 = pesrng_2.sample_uniform_torch(0.0, 1.0, (10,))
        self.assertTrue(torch.equal(output_1, output_2))

    def test_uniform_reproducibility_3D(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        pesrng_2 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_2.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        output_1 = pesrng_1.sample_uniform_torch(0.0, 1.0, (10, 5))
        output_2 = pesrng_2.sample_uniform_torch(0.0, 1.0, (10, 5))
        self.assertTrue(torch.equal(output_1, output_2))

    def test_uniform_index_sampling(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        pesrng_2 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_2.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        pesrng_2 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_2.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        ids = torch.arange(10, dtype=torch.int32, device="cuda")
        for i in range(100):
            output_1 = pesrng_1.sample_uniform_torch(0.0, 1.0, 10, ids=ids)
            output_2 = pesrng_2.sample_uniform_torch(0.0, 1.0, (10,), ids=ids)
            self.assertTrue(torch.equal(output_1, output_2))

    def test_uniform_bounds(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        for i in range(100):
            output = pesrng_1.sample_uniform_torch(0.0, 1.0, 1)
            self.assertTrue(torch.all(output >= 0.0) and torch.all(output <= 1.0))

    def test_uniform_different_for_each_run(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        output_1 = pesrng_1.sample_uniform_torch(0.0, 1.0, 1)
        output_2 = pesrng_1.sample_uniform_torch(0.0, 1.0, 1)
        self.assertFalse(torch.equal(output_1, output_2))

    ############################################################
    # Test Sign Sampling
    ############################################################

    def test_sign_randomness(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        output = pesrng_1.sample_sign_torch("float", 100)
        test = torch.mean(torch.square(torch.diff(output, dim=0)))
        self.assertTrue(test > 0.0)

    def test_sign_int_reproducibility_1D(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        pesrng_2 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_2.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        output_1 = pesrng_1.sample_sign_torch("int", 1)
        output_2 = pesrng_2.sample_sign_torch("int", (1,))
        self.assertTrue(torch.equal(output_1, output_2))

    def test_sign_int_reproducibility_2D(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        pesrng_2 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_2.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        output_1 = pesrng_1.sample_sign_torch("int", (10))
        output_2 = pesrng_2.sample_sign_torch("int", (10,))
        self.assertTrue(torch.equal(output_1, output_2))

    def test_sign_int_reproducibility_3D(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        pesrng_2 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_2.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        output_1 = pesrng_1.sample_sign_torch("int", (10, 5))
        output_2 = pesrng_2.sample_sign_torch("int", (10, 5))
        self.assertTrue(torch.equal(output_1, output_2))

    def test_sign_int_index_sampling(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        pesrng_2 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_2.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        ids = torch.arange(10, dtype=torch.int32, device="cuda")
        for i in range(100):
            output_1 = pesrng_1.sample_sign_torch("int", 10, ids=ids)
            output_2 = pesrng_2.sample_sign_torch("int", (10,), ids=ids)
            self.assertTrue(torch.equal(output_1, output_2))

    def test_sign_int_different_for_each_run(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        output_1 = pesrng_1.sample_sign_torch("int", 1)
        output_2 = pesrng_1.sample_sign_torch("int", 1)
        self.assertFalse(torch.equal(output_1, output_2))

    def test_sign_float_reproducibility_1D(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        pesrng_2 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_2.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        output_1 = pesrng_1.sample_sign_torch("float", 1)
        output_2 = pesrng_2.sample_sign_torch("float", (1,))
        self.assertTrue(torch.equal(output_1, output_2))

    def test_sign_float_reproducibility_2D(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        pesrng_2 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_2.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        output_1 = pesrng_1.sample_sign_torch("float", 10)
        output_2 = pesrng_2.sample_sign_torch("float", (10,))
        self.assertTrue(torch.equal(output_1, output_2))

    def test_sign_float_reproducibility_3D(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        pesrng_2 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_2.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        output_1 = pesrng_1.sample_sign_torch("float", (10, 5))
        output_2 = pesrng_2.sample_sign_torch("float", (10, 5))
        self.assertTrue(torch.equal(output_1, output_2))

    def test_sign_float_index_sampling(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        pesrng_2 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_2.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        ids = torch.arange(10, dtype=torch.int32, device="cuda")
        for i in range(100):
            output_1 = pesrng_1.sample_sign_torch("float", 10, ids=ids)
            output_2 = pesrng_2.sample_sign_torch("float", (10,), ids=ids)
            self.assertTrue(torch.equal(output_1, output_2))

    def test_sign_float_different_for_each_run(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        output_1 = pesrng_1.sample_sign_torch("float", 1)
        output_2 = pesrng_1.sample_sign_torch("float", 1)
        self.assertFalse(torch.equal(output_1, output_2))

    ############################################################
    # Test Integer Sampling
    ############################################################

    def test_integer_reproducibility_1D(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        pesrng_2 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_2.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        output_1 = pesrng_1.sample_integer_torch(0, 100, 1)
        output_2 = pesrng_2.sample_integer_torch(0, 100, (1,))
        self.assertTrue(torch.equal(output_1, output_2))

    def test_integer_reproducibility_2D(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        pesrng_2 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_2.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        output_1 = pesrng_1.sample_integer_torch(0, 100, 10)
        output_2 = pesrng_2.sample_integer_torch(0, 100, (10,))
        self.assertTrue(torch.equal(output_1, output_2))

    def test_integer_reproducibility_3D(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        pesrng_2 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_2.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        output_1 = pesrng_1.sample_integer_torch(0, 100, (10, 5))
        output_2 = pesrng_2.sample_integer_torch(0, 100, (10, 5))
        self.assertTrue(torch.equal(output_1, output_2))

    def test_integer_index_sampling(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        pesrng_2 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_2.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        ids = torch.arange(10, dtype=torch.int32, device="cuda")
        for i in range(100):
            output_1 = pesrng_1.sample_integer_torch(0, 100, 10, ids=ids)
            output_2 = pesrng_2.sample_integer_torch(0, 100, (10,), ids=ids)
            self.assertTrue(torch.equal(output_1, output_2))

    def test_integer_bounds(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        for i in range(100):
            output = pesrng_1.sample_integer_torch(0, 100, 1)
            self.assertTrue(torch.all(output >= 0) and torch.all(output <= 100))

    def test_integer_different_for_each_run(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        output_1 = pesrng_1.sample_integer_torch(0, 100, 1)
        output_2 = pesrng_1.sample_integer_torch(0, 100, 1)
        self.assertFalse(torch.equal(output_1, output_2))

    ############################################################
    # Test Normal Sampling
    ############################################################

    def test_normal_randomness(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        output = pesrng_1.sample_normal_torch(0.0, 1.0, 1000)
        test = torch.mean(torch.square(torch.diff(output, dim=0)))
        self.assertTrue(test > 0.0)

    def test_normal_reproducibility_1D(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        pesrng_2 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_2.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        output_1 = pesrng_1.sample_normal_torch(0.0, 1.0, 1)
        output_2 = pesrng_2.sample_normal_torch(0.0, 1.0, (1,))
        self.assertTrue(torch.equal(output_1, output_2))

    def test_normal_reproducibility_2D(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        pesrng_2 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_2.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        output_1 = pesrng_1.sample_normal_torch(0.0, 1.0, 10)
        output_2 = pesrng_2.sample_normal_torch(0.0, 1.0, (10,))
        self.assertTrue(torch.equal(output_1, output_2))

    def test_normal_reproducibility_3D(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        pesrng_2 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_2.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        output_1 = pesrng_1.sample_normal_torch(0.0, 1.0, (10, 5))
        output_2 = pesrng_2.sample_normal_torch(0.0, 1.0, (10, 5))
        self.assertTrue(torch.equal(output_1, output_2))

    def test_normal_index_sampling(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        pesrng_2 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_2.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )

        ids = torch.arange(10, dtype=torch.int32, device="cuda")
        for i in range(100):
            output_1 = pesrng_1.sample_normal_torch(0.0, 1.0, 10, ids=ids)
            output_2 = pesrng_2.sample_normal_torch(0.0, 1.0, (10,), ids=ids)
            self.assertTrue(torch.equal(output_1, output_2))

    def test_normal_mean_variance(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        values = pesrng_1.sample_normal_torch(0.0, 1.0, 100)
        mean = torch.mean(values)
        std = torch.std(values)
        self.assertAlmostEqual(mean, 0.0, delta=0.05)
        self.assertAlmostEqual(std, 1.0, delta=0.05)
        values = pesrng_1.sample_normal_torch(3.0, 4.0, 100)
        mean = torch.mean(values)
        std = torch.std(values)
        self.assertAlmostEqual(mean, 3.0, delta=0.05)
        self.assertAlmostEqual(std, 4.0, delta=0.05)

    def test_normal_different_for_each_run(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        output_1 = pesrng_1.sample_normal_torch(0.0, 1.0, 1)
        output_2 = pesrng_1.sample_normal_torch(0.0, 1.0, 1)
        self.assertFalse(torch.equal(output_1, output_2))

    ############################################################
    # Test Poisson Sampling
    ############################################################

    def test_poisson_reproducibility_1D(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        pesrng_2 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_2.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        output_1 = pesrng_1.sample_poisson_torch(1.0, 1)
        output_2 = pesrng_2.sample_poisson_torch(1.0, (1,))
        self.assertTrue(torch.equal(output_1, output_2))

    def test_poisson_reproducibility_2D(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        pesrng_2 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_2.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        output_1 = pesrng_1.sample_poisson_torch(1.0, 10)
        output_2 = pesrng_2.sample_poisson_torch(1.0, (10,))
        self.assertTrue(torch.equal(output_1, output_2))

    def test_poisson_reproducibility_3D(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        pesrng_2 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_2.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        output_1 = pesrng_1.sample_poisson_torch(1.0, (10, 5))
        output_2 = pesrng_2.sample_poisson_torch(1.0, (10, 5))
        self.assertTrue(torch.equal(output_1, output_2))

    def test_poisson_index_sampling(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        pesrng_2 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_2.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        ids = torch.arange(10, dtype=torch.int32, device="cuda")
        for i in range(100):
            output_1 = pesrng_1.sample_poisson_torch(1.0, 10, ids=ids)
            output_2 = pesrng_2.sample_poisson_torch(1.0, (10,), ids=ids)
            self.assertTrue(torch.equal(output_1, output_2))

    def test_poisson_different_for_each_run(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        output_1 = pesrng_1.sample_poisson_torch(1.0, 1)
        output_2 = pesrng_1.sample_poisson_torch(1.0, 1)
        self.assertFalse(torch.equal(output_1, output_2))

    ############################################################
    # Test Quaternion Sampling
    ############################################################

    def test_quaternion_reproducibility_1D(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        pesrng_2 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_2.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        output_1 = pesrng_1.sample_quaternion_torch(1)
        output_2 = pesrng_2.sample_quaternion_torch((1,))
        self.assertTrue(torch.equal(output_1, output_2))

    def test_quaternion_reproducibility_2D(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        pesrng_2 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_2.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        output_1 = pesrng_1.sample_quaternion_torch(10)
        output_2 = pesrng_2.sample_quaternion_torch((10,))
        self.assertTrue(torch.equal(output_1, output_2))

    def test_quaternion_reproducibility_3D(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        pesrng_2 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_2.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        output_1 = pesrng_1.sample_quaternion_torch((10, 5))
        output_2 = pesrng_2.sample_quaternion_torch((10, 5))
        self.assertTrue(torch.equal(output_1, output_2))

    def test_quaternion_index_sampling(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        pesrng_2 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_2.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )

        ids = torch.arange(10, dtype=torch.int32, device="cuda")
        for i in range(100):
            output_1 = pesrng_1.sample_quaternion_torch(10, ids=ids)
            output_2 = pesrng_2.sample_quaternion_torch((10,), ids=ids)
            self.assertTrue(torch.equal(output_1, output_2))

    def test_quaternion_different_for_each_run(self):
        pesrng_1 = PerEnvSeededRNG(42, 1000, "cuda")
        pesrng_1.set_seeds(
            torch.arange(1000, dtype=torch.int32, device="cuda"),
            torch.arange(1000, dtype=torch.int32, device="cuda"),
        )
        output_1 = pesrng_1.sample_quaternion_torch(1)
        output_2 = pesrng_1.sample_quaternion_torch(1)
        self.assertFalse(torch.equal(output_1, output_2))


if __name__ == "__main__":
    run_tests()
