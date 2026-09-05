# MLOps-iris

GitHub Actions is used to automate CI/CD tasks such as testing and building, while MLflow is used to track ML experiments, parameters, metrics, artifacts and models. Together they help automate and manage the ML lifecycle.

This is an important MLOps step because CI/CD tells us whether the code works, while MLflow helps us track how the ML model performs.

** Add MLflow to Our MLOps Project**

Our pipeline will become:

Developer
   ↓
GitHub
   ↓
GitHub Actions
   ↓
Run Tests
   ↓
Train Model
   ↓
MLflow
   ├── Parameters
   ├── Accuracy
   └── Model
5.1 What problem does MLflow solve?

Imagine tomorrow we change our model:

Run 1
Random Forest
100 trees
Accuracy = 0.95

Run 2
Random Forest
200 trees
Accuracy = 0.97

Run 3
Random Forest
50 trees
Accuracy = 0.93

Without experiment tracking, you might have to remember all of this manually.

MLflow records it for you.

5.2 Install MLflow

In your GitHub Codespace, make sure you're at the project root:

pwd

You should be in something like:

/workspaces/MLOps-iris/mlops-iris

Then update requirements.txt.

It should now contain:

scikit-learn
joblib
pytest
mlflow

Install:

python -m pip install -r requirements.txt

Verify:

mlflow --version

You should get an MLflow version number.

5.3 Modify src/train.py

We're going to add MLflow to our existing training code.

Replace your current src/train.py with:

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
    n_estimators = 100

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
5.4 What did we add?

The important new part is:

mlflow.set_experiment("Iris Classification")

We're creating an experiment called:

Iris Classification

Then:

with mlflow.start_run():

means:

Start recording this particular training run.

Parameters

We record:

mlflow.log_param("model_type", "RandomForest")

and:

mlflow.log_param("n_estimators", n_estimators)

So MLflow knows:

Model Type = RandomForest
Trees = 100
Metric

We record:

mlflow.log_metric("accuracy", accuracy)

So MLflow knows:

Accuracy = 0.97
Model

We also record the trained model:

mlflow.sklearn.log_model(
    model,
    "iris_model"
)

Now MLflow can keep track of the model produced by that run.

5.5 Run your model

From the project root:

python src/train.py

You should see something similar to:

Model accuracy: 0.97
Model saved to: models/iris_model.pkl

You should also see a new directory:

mlruns/

Check:

ls

You should now have something like:

.github
mlruns
models
src
tests
requirements.txt
README.md
.gitignore
What is mlruns?

For our beginner local setup, MLflow uses it to store experiment/run information.

Think of it as our local experiment database.

5.6 Start the MLflow UI

Now run:

mlflow ui

You should see something similar to:

Listening at: http://127.0.0.1:5000

Because you're using GitHub Codespaces, don't just open 127.0.0.1:5000 on your own computer.

Codespaces can forward the port.

Look at the PORTS tab in VS Code.

You should see:

5000

Then open the forwarded port.

You'll get the MLflow interface.

5.7 What you'll see in MLflow

You'll have an experiment:

Iris Classification

Inside it:

Run
│
├── Parameters
│     ├── model_type = RandomForest
│     └── n_estimators = 100
│
├── Metrics
│     └── accuracy = 0.97
│
└── Model
      └── iris_model

🎉 This is  first real MLOps experiment tracking.

5.8 Now let's make MLflow useful

Change:

n_estimators = 100

to:

n_estimators = 200

Run:

python src/train.py

Now MLflow will have another run.

Conceptually:

Iris Classification
│
├── Run 1
│   ├── Trees: 100
│   └── Accuracy: 0.97
│
└── Run 2
    ├── Trees: 200
    └── Accuracy: 0.97

Now change it to:

n_estimators = 50

Run again.

You'll have:

Run 1 → 100 trees → accuracy
Run 2 → 200 trees → accuracy
Run 3 → 50 trees  → accuracy

This is exactly why experiment tracking is useful.

5.9 Update .gitignore

We should not commit the local MLflow database/artifacts into our Git repository for this beginner setup.

Open .gitignore and add:

mlruns/

So your .gitignore becomes:

__pycache__/
*.pyc
.venv/
venv/
models/*.pkl
.pytest_cache/
mlruns/
5.10 Run your tests again

Before committing:

pytest -v

You should see:

2 passed

Then:

git status

You should see your changed files.

Commit them:

git add .
git commit -m "Add MLflow experiment tracking"
git push

Your existing GitHub Actions pipeline should run again.

So now we have:

              GitHub
                 ↓
           GitHub Actions
                 ↓
              pytest
                 ↓
             Training
                 ↓
              MLflow
            ↙    ↓    ↘
       Parameters Metric Model
🧠 The key MLOps distinction

You now have two different systems doing two different jobs:

GitHub Actions

Answers:

"Does our code pass the automated checks?"

Code
 ↓
Tests
 ↓
PASS / FAIL
MLflow

Answers:

"How did our ML experiments/models perform?"

Experiment
 ↓
Parameters
 ↓
Metrics
 ↓
Model

