from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



def train_model():
    """Train a Random Forest model."""

    iris = load_iris()

    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Model parameters
    n_estimators = 50

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy, n_estimators


def save_model(model):
    """Save trained model."""

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / "iris_model.pkl"

    joblib.dump(model, model_path)

    return model_path


if __name__ == "__main__":

    # Start MLflow experiment
    mlflow.set_experiment("Iris Classification")

    with mlflow.start_run():

        model, accuracy, n_estimators = train_model()

        print(f"Model accuracy: {accuracy:.2f}")

        # Log parameters
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", n_estimators)

        # Log metric
        mlflow.log_metric("accuracy", accuracy)

        # Save model locally
        model_path = save_model(model)

        # Log model to MLflow
        mlflow.sklearn.log_model(
            model,
            "iris_model"
        )

        print(f"Model saved to: {model_path}")