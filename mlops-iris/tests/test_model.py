from src.train import train_model

def test_model_accuracy():
    """Test the accuracy of the trained model."""
    model, accuracy = train_model()
    assert accuracy >= 0.80
def test_model_exists():
    """Test if the model file is created after training."""
    model, accuracy = train_model()
    assert model is not None
    