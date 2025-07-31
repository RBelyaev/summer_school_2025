import torch
import torch.nn as nn
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping
from ignite.metrics import Loss


class MLPDynamicModel(nn.Module):
    def __init__(self, layer_sizes):
        """Initialize a Multi-Layer Perceptron (MLP) with dynamic architecture.

        Args:
            layer_sizes (list): List of integers specifying the number of neurons in each layer.
                               e.g. [input_dim, hidden1_dim, ..., output_dim]
        """
        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.relu(x)

        x = self.layers[len(self.layers) - 1](x)

        return x


class MLPDynamicWithRegularizationModel(nn.Module):
    def __init__(self, layer_sizes, dropout_rate=0.1):
        """Initialize an MLP with dropout regularization for improved generalization.

        Args:
            layer_sizes (list): Specifies the architecture, e.g., [input_dim, hidden1_dim, ..., output_dim].
            dropout_rate (float, optional): Probability of dropping a neuron during training. Defaults to 0.1.
        """
        super().__init__()

        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.relu = nn.ReLU()

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

            if i < len(layer_sizes) - 2:
                self.dropouts.append(nn.Dropout(dropout_rate))

    def forward(self, x):
        """Forward pass with ReLU activation and dropout (during training only).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        for i in range(len(self.layers)):
            x = self.layers[i](x)

            if i < len(self.layers) - 1:
                x = self.relu(x)
                x = self.dropouts[i](x)

        return x


class IgniteTrainer:
    def __init__(
        self,
        model_name,
        model,
        optimizer,
        criterion,
        dataset_name,
        regularization,
        min_delta,
    ):
        """Initialize the trainer with model, optimizer, and training configurations.

        Args:
            model_name (str): Identifier for the model (used in logging paths).
            model (nn.Module): PyTorch model to train.
            optimizer (torch.optim): Optimizer (e.g., Adam, SGD).
            criterion (nn.Module): Loss function (e.g., MSELoss, CrossEntropyLoss).
            dataset_name (str): Name of the dataset (for organized logging).
            regularization (str): Type of regularization (e.g., "dropout", "l2").
            min_delta (float): Minimum change in validation loss to qualify as improvement for early stopping.
        """

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self._init_engines(model_name, dataset_name, regularization, min_delta)

    def _init_engines(self, model_name, dataset_name, regularization, min_delta):
        """Initialize the training and validation engines with event handlers."""

        def train_step(engine, batch):
            """Single training step: forward pass, loss computation, backward pass, and optimizer update."""

            self.model.train()
            x, y = batch
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
            loss.backward()
            self.optimizer.step()
            return loss.item()

        self.trainer = Engine(train_step)

        def validation_step(engine, batch):
            """Single validation step: forward pass without gradients."""

            self.model.eval()
            with torch.no_grad():
                x, y = batch
                y_pred = self.model(x)
            return y_pred, y

        self.evaluator = Engine(validation_step)

        metric = Loss(self.criterion)
        metric.attach(self.evaluator, "val_loss")

        def add_early_stopping(engine):
            """Add early stopping handler after 150 epochs to prevent overfitting."""
            if engine.state.epoch == 150:
                self.early_stopping = EarlyStopping(
                    patience=20,
                    score_function=lambda e: -e.state.metrics["val_loss"],
                    trainer=self.trainer,
                    min_delta=min_delta,
                )
                self.evaluator.add_event_handler(Events.COMPLETED, self.early_stopping)

        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, add_early_stopping)

        self.logger = TensorboardLogger(
            log_dir="logs/" + dataset_name + "/" + model_name[-1] + "/" + regularization + "/" + model_name[:-1]
        )
        self.logger.attach_output_handler(
            self.trainer,
            event_name=Events.EPOCH_COMPLETED,
            tag="training",
            output_transform=lambda x: {"loss": x},
        )

        self.logger.attach_output_handler(
            self.evaluator,
            event_name=Events.COMPLETED,
            tag="validation",
            metric_names=["val_loss"],
            global_step_transform=global_step_from_engine(self.trainer),
        )

    def fit(self, train_loader, val_loader, max_epochs=10):
        """Run training with validation after each epoch.

        Args:
            train_loader (DataLoader): Training data iterator.
            val_loader (DataLoader): Validation data iterator.
            max_epochs (int): Maximum number of training epochs.
        """

        @self.trainer.on(Events.EPOCH_COMPLETED)
        def run_validation(engine):
            self.evaluator.run(val_loader)

        self.trainer.run(train_loader, max_epochs=max_epochs)
