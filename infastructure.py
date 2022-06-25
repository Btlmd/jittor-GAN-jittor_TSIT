import jittor as jt
import jittor.nn as nn
from IPython import embed


class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()

    def execute(self, *args, **kw):
        res = self.forward(*args, **kw)
        # embed()
        return res
