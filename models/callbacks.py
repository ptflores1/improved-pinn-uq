#import wandb
import time

class NeurodiffeqWNBCallback:
    def __init__(self):
        self.last_iter_time = None
        

    def __call__(self, solver):
        last_iter_time = None
        if self.last_iter_time is None:
            self.last_iter_time = time.time()
        else:
            last_iter_time = time.time() - self.last_iter_time
            self.last_iter_time = time.time()
        loss = solver.metrics_history['train_loss'][solver.local_epoch - 1]
        #wandb.log({ "loss": loss, "iter_time": last_iter_time }, step=solver.local_epoch - 1)

class BVICallback:
    def __init__(self):
        self.last_iter_time = None

    def __call__(self, loss, epoch):
        last_iter_time = None
        if self.last_iter_time is None:
            self.last_iter_time = time.time()
        else:
            last_iter_time = time.time() - self.last_iter_time
            self.last_iter_time = time.time()
        #wandb.log({ "loss": loss, "iter_time": last_iter_time }, step=epoch)

class HMCCallback:
    def __init__(self):
        self.last_iter_time = None

    def __call__(self, kernel, samples, stage, i):
        last_iter_time = None
        if self.last_iter_time is None:
            self.last_iter_time = time.time()
        else:
            last_iter_time = time.time() - self.last_iter_time
            self.last_iter_time = time.time()
        #wandb.log({ "iter_time": last_iter_time })