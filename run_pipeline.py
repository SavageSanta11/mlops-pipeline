from pipelines.training_pipeline import train_pipeline
from steps.clean_data import clean_data
from steps.evaluation import evaluation
from steps.ingest_data import ingest_data
from steps.model_train import train_model

from zenml.client import Client 
if __name__ == "__main__":

    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    training = train_pipeline(
        ingest_data(),
        clean_data(),
        train_model(),
        evaluation(),
    )

    training.run()