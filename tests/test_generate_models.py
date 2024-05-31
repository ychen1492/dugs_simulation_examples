from unittest.mock import patch

import numpy as np
from darts.physics.geothermal.physics import Geothermal
from darts.reservoirs.struct_reservoir import StructReservoir

from dugs_simulation_examples.generate_models import (
    generate_poro_normalized_distribution,
    homogeneous_model_simulation,
    stratified_model_simulation,
    upscale_porosity,
)
from dugs_simulation_examples.model import Model
from dugs_simulation_examples.xarray_api import XarrayApi


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

    @patch.object(Model, 'init')
    def test_generate_homogeneous_model_without_overburden(self, mock_init):
        # Arrange
        test_nx = 20
        test_ny = 20
        test_nz = 10
        test_dx = 5
        test_dy = 5
        test_dz = 5
        test_perms = np.ones(test_nx * test_ny * test_nz) * 800
        test_poros = np.ones(test_nx * test_ny * test_nz) * 0.2
        expected_attributes = ['reservoir', 'physics', 'xrdata', 'params']
        # Analysis
        test_model = homogeneous_model_simulation(test_nx, test_ny, test_nz, test_dx, test_dy, test_dz, log_file='')

        # Assert
        np.testing.assert_allclose(test_nx, test_model.nx, atol=1e-5)
        np.testing.assert_allclose(test_ny, test_model.ny, atol=1e-5)
        np.testing.assert_allclose(test_nz, test_model.nz, atol=1e-5)

        np.testing.assert_allclose(test_dx, test_model.dx, atol=1e-5)
        np.testing.assert_allclose(test_dy, test_model.dy, atol=1e-5)
        np.testing.assert_allclose(test_dz, test_model.dz, atol=1e-5)

        np.testing.assert_allclose(test_perms, test_model.perm, atol=1e-5)
        np.testing.assert_allclose(test_poros, test_model.poro, atol=1e-5)

        # Check the existence of attributes
        # Check the attribute is the instance of the certain class
        for attr in expected_attributes:
            assert hasattr(test_model, attr)

        assert type(test_model.reservoir) is StructReservoir
        assert type(test_model.physics) is Geothermal
        assert type(test_model.xrdata) is XarrayApi

    @patch.object(Model, 'init')
    def test_generate_stratified_model(self, mock_init):
        # Arrange
        test_nx = 20
        test_ny = 20
        test_nz = 10
        test_dx = 5
        test_dy = 5
        test_dz = 5

        expected_attributes = ['reservoir', 'physics', 'xrdata', 'params']
        # Analysis
        test_model = stratified_model_simulation(test_nx, test_ny, test_nz, test_dx, test_dy, test_dz, log_file='')

        # Assert
        np.testing.assert_allclose(test_nx, test_model.nx, atol=1e-5)
        np.testing.assert_allclose(test_ny, test_model.ny, atol=1e-5)
        np.testing.assert_allclose(test_nz, test_model.nz, atol=1e-5)

        np.testing.assert_allclose(test_dx, test_model.dx, atol=1e-5)
        np.testing.assert_allclose(test_dy, test_model.dy, atol=1e-5)
        np.testing.assert_allclose(test_dz, test_model.dz, atol=1e-5)

        # np.testing.assert_allclose(test_perms, test_model.perm, atol=1e-5)
        # np.testing.assert_allclose(test_poros, test_model.poro, atol=1e-5)

        # Check the existence of attributes
        # Check the attribute is the instance of the certain class
        for attr in expected_attributes:
            assert hasattr(test_model, attr)

        assert type(test_model.reservoir) is StructReservoir
        assert type(test_model.physics) is Geothermal
        assert type(test_model.xrdata) is XarrayApi

    def test_generate_heterogeneous_model(self):
        pass
