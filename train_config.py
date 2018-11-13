class TrainConfig:
    def __init__(self):
        if 'optimizer' not in globals():
            self.optimizer = 'Adam'
        else:
            self.optimizer = optimizer

        if 'learning_rate' not in globals():
            self.learning_rate = '0.0001'
        else:
            self.learning_rate = learning_rate

        if 'epochs_count' not in globals():
            self.epochs_count = '50'
        else:
            self.epochs_count = epochs_count

        if 'batch_size' not in globals():
            self.batch_size = '32'
        else:
            self.batch_size = batch_size