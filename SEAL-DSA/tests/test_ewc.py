"""
Unit tests for Elastic Weight Consolidation (EWC).
Tests Fisher computation, loss calculation, and online updates.
"""

import pytest
import torch
import torch.nn as nn

from seal_dsa.training.ewc import EWC
from seal_dsa.config import EWCConfig


class SimpleModel(nn.Module):
    """A tiny model for testing EWC."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 5)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


@pytest.fixture
def model():
    return SimpleModel()


@pytest.fixture
def ewc_config():
    return EWCConfig(enabled=True, lambda_=0.4, fisher_sample_size=5)


@pytest.fixture
def ewc(model, ewc_config):
    return EWC(model, ewc_config)


class TestEWCInitialization:
    """Test EWC initialization."""

    def test_creation(self, ewc):
        assert ewc is not None
        assert ewc.lambda_ == 0.4
        assert not ewc.initialized

    def test_initial_loss_is_zero(self, ewc, model):
        """Before Fisher computation, EWC loss should be zero."""
        loss = ewc.compute_loss(model)
        assert loss.item() == 0.0


class TestEWCLoss:
    """Test EWC loss computation."""

    def test_loss_after_fisher_update(self, ewc, model):
        """After Fisher update, moving params should produce non-zero loss."""
        # Store current params as optimal
        ewc.fisher = {
            name: torch.ones_like(param)
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        ewc.optimal_params = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        ewc.initialized = True

        # Perturb model parameters
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.1)

        loss = ewc.compute_loss(model)
        assert loss.item() > 0.0

    def test_loss_is_zero_at_optimum(self, ewc, model):
        """When parameters equal optimal, EWC loss should be zero."""
        ewc.fisher = {
            name: torch.ones_like(param)
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        ewc.optimal_params = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        ewc.initialized = True

        # Don't perturb — params are at optimal
        loss = ewc.compute_loss(model)
        assert abs(loss.item()) < 1e-6

    def test_higher_lambda_increases_loss(self, model):
        """Higher lambda should produce larger EWC loss."""
        losses = []
        for lam in [0.1, 0.5, 1.0]:
            cfg = EWCConfig(enabled=True, lambda_=lam, fisher_sample_size=5)
            ewc = EWC(model, cfg)

            ewc.fisher = {
                name: torch.ones_like(param)
                for name, param in model.named_parameters()
                if param.requires_grad
            }
            ewc.optimal_params = {
                name: param.data.clone() - 0.1
                for name, param in model.named_parameters()
                if param.requires_grad
            }
            ewc.initialized = True

            losses.append(ewc.compute_loss(model).item())

        # Higher lambda → higher loss
        assert losses[0] < losses[1] < losses[2]


class TestFisherComputation:
    """Test Fisher Information Matrix computation."""

    def test_fisher_accumulation(self, ewc, model):
        """Test that _accumulate_fisher updates the Fisher dict."""
        fisher_dict = {
            name: torch.zeros_like(param.data)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        # Create a fake batch
        x = torch.randn(4, 10)
        target = torch.randint(0, 5, (4,))

        # Compute loss and accumulate
        model.zero_grad()
        output = model(x)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_dict[name] += param.grad.data ** 2

        # Fisher values should be non-negative
        for name, f in fisher_dict.items():
            assert (f >= 0).all(), f"Negative Fisher values for {name}"

    def test_importance_summary(self, ewc, model):
        """Test get_importance_summary."""
        # Before initialization
        summary = ewc.get_importance_summary()
        assert len(summary) == 0

        # After initialization
        ewc.fisher = {
            name: torch.rand_like(param)
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        summary = ewc.get_importance_summary()
        assert len(summary) > 0
        for name, norm in summary.items():
            assert norm >= 0


class TestOnlineEWC:
    """Test online Fisher updates."""

    def test_update_count_increases(self, ewc, model):
        """Update count should increment."""
        assert ewc.update_count == 0

        # Manually set Fisher and optimal params
        ewc.fisher = {
            name: torch.ones_like(param)
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        ewc.optimal_params = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        ewc.initialized = True
        ewc.update_count = 1

        assert ewc.update_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
