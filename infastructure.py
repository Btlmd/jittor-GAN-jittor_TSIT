import jittor as jt
import jittor.nn as nn


class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()

    def execute(self, *args, **kw):
        return self.forward(*args, **kw)
