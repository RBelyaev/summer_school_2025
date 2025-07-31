import torch
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Engine
from notebook_tests.summer_school_notebook import (
    IgniteTrainer,
    MLPDynamicModel,
    MLPDynamicWithRegularizationModel,
)
from torch import nn


class TestNotebook:
    """Tests for Jupyter Notebook"""

    def test_mlp_model_classes(self):
        """Checking model classes"""
        # Checking the MLPDynamicModel
        model1 = MLPDynamicModel([10, 20, 1])
        assert isinstance(model1, nn.Module)

        # Checking the MLPDynamicWithRegularizationModel
        model2 = MLPDynamicWithRegularizationModel([10, 20, 30, 1])
        assert isinstance(model2, nn.Module)

    def test_model_forward_pass(self):
        """Checking forward pass models"""
        # Checking the MLPDynamicModel
        model1 = MLPDynamicModel([5, 10, 1])
        output1 = model1(torch.randn(32, 5))
        assert output1.shape == (32, 1)

        # Checking the MLPDynamicWithRegularizationModel
        model2 = MLPDynamicWithRegularizationModel([5, 10, 1])
        output2 = model2(torch.randn(32, 5))
        assert output2.shape == (32, 1)

    def test_ignite_trainer_initialization(self):
        """Test IgniteTrainer initialization and basic functionality"""
        # Create a simple model and components
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.MSELoss()

        # Initialize trainer
        trainer = IgniteTrainer(
            model_name="test_model",
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            dataset_name="test_data",
            regularization="l2",
            min_delta=0.01,
        )

        # Check basic attributes
        assert hasattr(trainer, "trainer")
        assert isinstance(trainer.trainer, Engine)
        assert hasattr(trainer, "evaluator")
        assert isinstance(trainer.evaluator, Engine)
        assert hasattr(trainer, "logger")
        assert isinstance(trainer.logger, TensorboardLogger)

    def test_ignite_trainer_fit(self):
        """Test IgniteTrainer fit method with mock data"""
        # Create a simple model and components
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.MSELoss()

        # Initialize trainer
        trainer = IgniteTrainer(
            model_name="test_model",
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            dataset_name="test_data",
            regularization="dropout",
            min_delta=0.001,
        )

        # Create mock data loaders
        X = torch.randn(100, 10)
        y = torch.randn(100, 1)
        dataset = torch.utils.data.TensorDataset(X, y)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=10)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=10)

        # Test fitting
        trainer.fit(train_loader, val_loader, max_epochs=2)

        # Check that training completed (basic smoke test)
        assert True  # If we get here without errors, the test passed
