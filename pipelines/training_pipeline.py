from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.evaluation import evaluate_model

@pipeline(enable_cache=True)
def train_pipeline(ingest_data, clean_data, model_train, evaluation):
    df = ingest_data()
    x_train, x_test, y_train, y_test = clean_data(df)
    model = model_train(x_train, x_test, y_train, y_test)
    mse, rmse = evaluation(model, x_test, y_test)
