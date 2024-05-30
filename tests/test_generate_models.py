import numpy as np

from dugs_simulation_examples.generate_models import generate_poro_normalized_distribution, upscale_porosity


class TestGenerateModel:

    def test_generate_poro_normalized_distribution(self):
        # Arrange
        task_id = 1
        expected_result = np.array([0.2312, 0.1194, 0.1235, 0.0963, 0.1932, 0.2372, 0.1119])
        # Analysis
        test_poro = generate_poro_normalized_distribution(task_id=task_id)
        # Assert
        np.testing.assert_allclose(expected_result, test_poro[:7], atol=1e-4)

    def test_upscale_porosity(self):
        # Arrange
        input_poro = np.linspace(0.01, 0.1, num=10)
        upscaling_factor = 4
        expected_result = np.array([0.01, 0.04, 0.07, 0.1])
        # Analysis
        test_poro = upscale_porosity(input_poro, upscaling_factor)
        # Assert
        np.testing.assert_allclose(expected_result, test_poro, atol=1e-5)
