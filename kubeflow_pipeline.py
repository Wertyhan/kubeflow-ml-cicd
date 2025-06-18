from typing import Dict, List
from kfp import dsl
from kfp import compiler
from kfp.dsl import Input, Output, Dataset, Model, component

# Step 1: Load Dataset
@dsl.component(base_image="python:3.9")
def load_data(output_csv: Output[Dataset]):
    import subprocess
    subprocess.run(["pip", "install", "pandas", "scikit-learn"], check=True)

    from sklearn.datasets import load_breast_cancer
    import pandas as pd

    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    df.to_csv(output_csv.path, index=False)

# Step 2: Preprocess Data
@dsl.component(base_image="python:3.9")
def preprocess_data(input_csv: Input[Dataset], output_train: Output[Dataset], output_test: Output[Dataset], 
                    output_ytrain: Output[Dataset], output_ytest: Output[Dataset]):
    import subprocess
    subprocess.run(["pip", "install", "pandas", "scikit-learn"], check=True)

    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(input_csv.path)
    df = df.dropna()

    features = df.drop(columns=['target'])
    target = df['target']

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.2, random_state=42)

    X_train_df = pd.DataFrame(X_train, columns=features.columns)
    y_train_df = pd.DataFrame(y_train)
    X_test_df = pd.DataFrame(X_test, columns=features.columns)
    y_test_df = pd.DataFrame(y_test)

    X_train_df.to_csv(output_train.path, index=False)
    X_test_df.to_csv(output_test.path, index=False)
    y_train_df.to_csv(output_ytrain.path, index=False)
    y_test_df.to_csv(output_ytest.path, index=False)

# Step 3: Train Model (KNN)
@dsl.component(base_image="python:3.9")
def train_model(train_data: Input[Dataset], ytrain_data: Input[Dataset], model_output: Output[Model]):
    import subprocess
    subprocess.run(["pip", "install", "pandas", "scikit-learn", "joblib"], check=True)

    import pandas as pd
    from sklearn.neighbors import KNeighborsClassifier
    from joblib import dump

    X_train = pd.read_csv(train_data.path)
    y_train = pd.read_csv(ytrain_data.path).values.ravel()

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    dump(model, model_output.path)

# Step 4: Evaluate Model
@dsl.component(base_image="python:3.9")
def evaluate_model(test_data: Input[Dataset], ytest_data: Input[Dataset], model: Input[Model], metrics_output: Output[Dataset]):
    import subprocess
    subprocess.run(["pip", "install", "pandas", "scikit-learn", "matplotlib", "joblib"], check=True)

    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    from joblib import load

    X_test = pd.read_csv(test_data.path)
    y_test = pd.read_csv(ytest_data.path).values.ravel()

    model = load(model.path)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # Save metrics report
    metrics_path = metrics_output.path
    with open(metrics_path, 'w') as f:
        f.write(str(report))

    # Save confusion matrix as image
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(metrics_path.replace('.csv', '.png').replace('.txt', '.png'))

# Pipeline definition
@dsl.pipeline(name="breast-cancer-knn-pipeline")
def ml_pipeline():
    load_op = load_data()
    preprocess_op = preprocess_data(input_csv=load_op.outputs["output_csv"])
    train_op = train_model(train_data=preprocess_op.outputs["output_train"], ytrain_data=preprocess_op.outputs["output_ytrain"])
    evaluate_op = evaluate_model(test_data=preprocess_op.outputs["output_test"], ytest_data=preprocess_op.outputs["output_ytest"], model=train_op.outputs["model_output"])

# Compile the pipeline
if __name__ == "__main__":
    compiler.Compiler().compile(pipeline_func=ml_pipeline, package_path="breast_cancer_knn_pipeline.yaml")
