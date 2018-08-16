import numpy as np

from keras.callbacks import Callback
from sklearn.metrics import classification_report


class LogEpochStatistics(Callback):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def on_epoch_begin(self, epoch, logs=None):
        self.logger.info(f"Epoch {epoch} started")

    def on_epoch_end(self, epoch, logs=None):
        self.logger.info(f"Epoch {epoch} ended")
        self.logger.info(f"TrainAccuracy: {logs.get('acc')} \t TrainLoss: {logs.get('loss')} \t "
                         f"ValidAccuracy: {logs.get('val_acc')} \t ValidationLoss: {logs.get('val_loss')}")


class Metrics(Callback):
    def __init__(self, identifier, metrics_writer, write_per_rounds=10):
        super().__init__()
        self.identifier = identifier
        self.metrics_writer = metrics_writer
        self.write_per_rounds = write_per_rounds

    @staticmethod
    def _convert_classification_report_to_dict(report):
        report_data = dict()
        lines = report.split('\n')
        for line in lines[2:-3]:
            row = {}
            row_data = line.split('      ')
            row['class'] = row_data[0]
            row['precision'] = float(row_data[1])
            row['recall'] = float(row_data[2])
            row['f1_score'] = float(row_data[3])
            row['support'] = float(row_data[4])
            report_data.append(row)

        return report_data

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.write_per_rounds == 0:
            prediction = np.asarray(self.model.predict([self.validation_data[0]]))
            prediction = prediction.reshape(prediction.shape[0] * prediction.shape[1], prediction.shape[2])
            target = self.validation_data[2]
            target = target.reshape(target.shape[0] * target.shape[1], target.shape[2])

            report = classification_report(np.argmax(target, axis=-1), np.argmax(prediction, axis=-1))
            report_dict = Metrics._convert_classification_report_to_dict(report)
            self.metrics_writer.write(report_dict, identifer=self.identifier)
