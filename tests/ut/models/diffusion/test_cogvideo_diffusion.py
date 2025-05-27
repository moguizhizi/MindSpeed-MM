import torch
import pytest
import numpy as np

from mindspeed_mm.models.diffusion.cogvideo_diffusion import append_dims, default, append_zero, \
    generate_roughly_equally_spaced_steps, EpsWeighting, make_beta_schedule
from tests.ut.utils import judge_expression


class TestCogvideoDiffusion:

    def test_append_dims_no_append_needed(self):
        """Test when no dimensions need to be appended."""
        x = torch.tensor([1, 2, 3])
        result = append_dims(x, 1)
        judge_expression(result.shape == (3,))
        judge_expression(torch.equal(result, x))

    def test_append_dims_append_one_dim(self):
        """Test appending one dimension."""
        x = torch.tensor([1, 2, 3])
        result = append_dims(x, 2)
        expected_shape = (3, 1)
        judge_expression(result.shape == expected_shape)

    def test_append_dims_append_multiple_dims(self):
        """Test appending multiple dimensions."""
        x = torch.tensor([1, 2, 3])
        target_dims = 5
        result = append_dims(x, target_dims)
        expected_shape = (3,) + (1,) * (target_dims - x.ndim)
        judge_expression(result.shape == expected_shape)

    def test_append_dims_already_higher_dims(self):
        """Test when the input already has more dimensions than target_dims."""
        x = torch.rand(2, 3, 4)
        with pytest.raises(ValueError):
            append_dims(x, 2)

    def test_append_dims_target_dims_equal_input_dims(self):
        """Test when target_dims is equal to the number of dimensions in the input."""
        x = torch.rand(2, 3, 4)
        result = append_dims(x, 3)
        judge_expression(result.shape == (2, 3, 4))
        judge_expression(torch.equal(result, x))

    def test_default_val_not_none(self):
        """Test when val is not None."""
        result = default(5, lambda: 10)
        judge_expression(result == 5)

    def test_default_val_none_d_is_function(self):
        """Test when val is None and d is a function."""
        def func():
            return "default_value"

        result = default(None, func)
        judge_expression(result == "default_value")

    def test_default_val_none_d_is_none(self):
        """Test when both val and d are None."""
        result = default(None, None)
        judge_expression(result is None)

    def test_append_zero_regular_tensor(self):
        """Test appending zero to a regular tensor."""
        x = torch.tensor([1, 2, 3])
        result = append_zero(x)
        expected = torch.tensor([1, 2, 3, 0])
        judge_expression(torch.equal(result, expected))

    def test_append_zero_multidimensional_tensor(self):
        """Test appending zero to a multidimensional tensor."""
        x = torch.tensor([[1, 2], [3, 4]])
        with pytest.raises(RuntimeError):
            append_zero(x)

    def test_append_zero_empty_tensor(self):
        """Test appending zero to an empty tensor."""
        x = torch.tensor([])
        result = append_zero(x)
        expected = torch.tensor([0])
        judge_expression(torch.equal(result, expected))

    def test_generate_roughly_equally_spaced_steps_case(self):
        """Test with a normal case."""
        num_substeps = 5
        max_step = 10
        result = generate_roughly_equally_spaced_steps(num_substeps, max_step)
        expected = np.array([1, 3, 5, 7, 9])
        judge_expression(np.array_equal(result, expected))

    def test_eps_weighting_positive_input(self):
        """Test with a positive input."""
        weighting = EpsWeighting()
        sigma = 2.0
        result = weighting(sigma)
        expected = sigma ** -2.0
        judge_expression(np.isclose(result, expected))

    def test_make_beta_schedule_linear(self):
        """Test with linear schedule."""
        n_timestep = 10
        betas = make_beta_schedule("linear", n_timestep)
        expected = np.linspace(1e-4 ** 0.5, 2e-2 ** 0.5, n_timestep) ** 2
        judge_expression(np.allclose(betas, expected))

    def test_make_beta_schedule_custom_params(self):
        """Test with custom linear_start and linear_end."""
        n_timestep = 10
        linear_start = 1e-3
        linear_end = 5e-2
        betas = make_beta_schedule("linear", n_timestep, linear_start, linear_end)
        expected = np.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep) ** 2
        judge_expression(np.allclose(betas, expected))

    def test_make_beta_schedule_no_linear(self):
        """Test with no linear schedule."""
        n_timestep = 10
        with pytest.raises(NotImplementedError):
            make_beta_schedule("cosine", n_timestep)
