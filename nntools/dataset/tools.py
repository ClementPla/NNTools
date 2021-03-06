from nntools.utils.misc import partial_fill_kwargs


class Composition:
    def __init__(self, **config):
        self.config = config
        self.ops = []
        self.deactivated = []

    def add(self, *funcs):
        for f in funcs:
            self.ops.append(partial_fill_kwargs(f, self.config))
        return self

    def deactivate_op(self, index):
        if not isinstance(index, list):
            index = [index]
        self.deactivated += index

    def __call__(self, **kwargs):
        for i, op in enumerate(self.ops):
            if i in self.deactivated:
                continue
            kwargs = op(**kwargs)
        return kwargs

    def __lshift__(self, other):
        return self.add(other)

    def __str__(self):
        output = ''
        for i, o in enumerate(self.ops):
            output += '%i_' % i + str(o) + ' STATUS: ' + ('Active' if i not in self.deactivated else 'Inactive') + ' \n'
        return output
