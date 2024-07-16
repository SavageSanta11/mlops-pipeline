from abc import ABC, abstractmethod
import numpy as np
import logging

from sklearn.metrics import mean_squared_error, r2_score
class Evaluation(ABC):
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass


class MSE(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("MSE: {}".format(e))
            return mse
        except Exception as e:
            logging.error("Error in calculatating MSE {}".format(e))
            raise(e)
        
class R2(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating R2 Score")
            r2 = r2_score(y_true, y_pred)
            logging.info("R2 Score: {}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error in calculating r2 Score: {}".format(e))
            raise e

class RMSE(Evaluation):
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Entered the calculate_score method of the RMSE class")
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info("The root mean squared error value is: " + str(rmse))
            return rmse
        except Exception as e:
            logging.error(
                "Exception occurred in calculate_score method of the RMSE class. Exception message:  "
                + str(e)
            )
            raise e
