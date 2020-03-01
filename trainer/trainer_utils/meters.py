import numpy as np

from sklearn.metrics import accuracy_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ACCMeter(object):

    def __init__(self):
        self.target = np.ones(0)
        self.output = np.ones(0)

    def reset(self):
        self.target = np.ones(0)
        self.output = np.ones(0)

    def update(self, target, output):
        self.target = np.hstack([self.target, target])
        self.output = np.hstack([self.output, output.argmax(1)])

    def get_accuracy(self):
        acc = accuracy_score(self.target,
                             self.output)
        return acc

    def get_top_hard_examples(self, top_n=10):
        diff_arr = np.abs(self.target - self.output)
        hard_indexes = np.argsort(diff_arr)[::-1]
        hard_indexes = hard_indexes[:top_n]
        return hard_indexes, self.target[hard_indexes], self.output[hard_indexes]
