import os
from pathlib import Path
import pymsteams

from logger import logger
from utils.common import load_json
from constants import THRESHOLDS

class FailureDetection:
    def __init__(self):
        eval_path = Path('evaluation/metrics/base-scores.json')
        bias_path = Path('evaluation/metrics/bias-scores.json')
        self.eval_metrics = load_json(eval_path)
        self.bias_metrics = load_json(bias_path)


    def calculate_eval_mean(self, data):
        means = {}
        for key, values in data.items():
            means[key] = sum(values.values()) / len(values)
        return means
    

    def calculate_bias_mean(self, data):
        total_bias_detected_count = 0
        total_num_reviews = 0

        for product_id, data in data.items():
            total_bias_detected_count += data["bias_detected_count"]
            total_num_reviews += data["num_reviews"]
        
        metrics = {
            'total_bias_detected_count': total_bias_detected_count,
            'total_num_reviews': total_num_reviews
        }
        return metrics


    def check_metrics_and_send_message(self, metrics, thresholds):
        failure = False
        failed_metrics = []

        if ((metrics['total_bias_detected_count'] / metrics['total_num_reviews']) >= 0.5):
            failure = True
            failed_metrics.append(f"{'total_bias_detected_count'}: {metrics['total_bias_detected_count']}")
            failed_metrics.append(f"{'total_num_reviews'}: {metrics['total_num_reviews']}")
            failed_metrics.append(f"{'total_bias_percent'}: {metrics['total_bias_detected_count']/metrics['total_num_reviews']}")

        # Check if any metric is below the threshold
        for metric, threshold in thresholds.items():
            if metrics[metric] < threshold:
                failure = True
                failed_metrics.append(f"{metric}: {metrics[metric]}")

        if failure:
            self.send_failure_notif(failed_metrics)
        else:
            self.send_success_notif(metrics)


    def send_failure_notif(self, failed_metrics):
        try:
            myTeamsMessage = pymsteams.connectorcard(hookurl=os.getenv("MS_TEAMS_WEBHOOK_URL"))
            myTeamsMessage.title("Failure Notification: Metrics Below Threshold")
            myTeamsMessage.text(f"The following metrics are below the threshold:\n\n{(", \n\n".join(failed_metrics))}")
            myTeamsMessage.send()
            logger.info(f"Failure Alert sent successfully")
        except Exception as e:
            logger.error(f"Error sending Failure Alert: {e}")
    
    
    def send_success_notif(self, metrics):
        try:
            success_metrics = [f"{key}: {value}" for key, value in metrics.items()]
            myTeamsMessage = pymsteams.connectorcard(hookurl=os.getenv("MS_TEAMS_WEBHOOK_URL"))
            myTeamsMessage.title("Success Notification: Metrics Above Threshold")
            myTeamsMessage.text(f"The metrics are:\n\n{(", \n\n".join(success_metrics))}")
            myTeamsMessage.send()
            logger.info(f"Success Alert sent successfully")
        except Exception as e:
            logger.error(f"Error sending Success Alert: {e}")

    
    def detect(self):
        mean_eval = self.calculate_eval_mean(self.eval_metrics)
        mean_bias = self.calculate_bias_mean(self.bias_metrics)

        total_metric = mean_eval | mean_bias
        
        self.check_metrics_and_send_message(total_metric, THRESHOLDS)


