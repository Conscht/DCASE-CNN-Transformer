from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

class DelayedEarlyStopping(EarlyStopping):
    def __init__(self, start_epoch, **kwargs):
        super().__init__(**kwargs)
        self.start_epoch = start_epoch

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.start_epoch:
            super().on_train_epoch_end(trainer, pl_module)