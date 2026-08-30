from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_model():
    """Train a Random Forest Model using the Iris dataset."""

    # Load Dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Create Model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Train Model
    model.fit(X_train, y_train)

    # Make Predictions
    predictions = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy


def save_model(model):
    """Save the trained model to a file."""

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / "iris_model.pkl"

    joblib.dump(model, model_path)

    return model_path


if __name__ == "__main__":
    model, accuracy = train_model()

    print(f"Model trained with accuracy: {accuracy:.2f}")

    model_path = save_model(model)

    print(f"Model saved to: {model_path}")