from pathlib import Path

from src.train import train_model, save_model


def test_model_accuracy():
    """Test the accuracy of the trained model."""
    model, accuracy, n_estimators = train_model()

    assert accuracy >= 0.90


def test_model_exists():
    """Test if the model file is created after training."""
    model, accuracy, n_estimators = train_model()

    model_path = save_model(model)

    assert Path(model_path).exists()
