import functools
import hashlib
import inspect


def nntools_wrapper(func):
    @functools.wraps(func)
    def wrapper(**kwargs):
        expected_parameters = inspect.signature(func).parameters.values()
        arguments = {}
        for p in expected_parameters:
            if p.name in kwargs:
                arguments[p.name] = kwargs.pop(p.name)
            elif p.default is not p.empty:
                arguments[p.name] = p.default
        output = func(**arguments)
        output.update(kwargs)
        return output

    return wrapper


class CacheBullet:
    def __call__(self, **kwargs):
        return kwargs


class Composition:
    def __init__(self):
        self.ops = []
        self._index_bullet = 0
        self._id = ""

    @property
    def id(self):
        # An explicitly assigned id always wins.
        if self._id:
            return self._id
        return self._compute_id()

    @id.setter
    def id(self, value):
        self._id = value

    def _compute_id(self):
        """Derive a stable identifier from the ops that influence cached data.

        Only the ops *before* the first CacheBullet affect the pre-cache
        (i.e. the data that gets written to disk/memory), so the id is built
        from those. This makes the disk cache invalidate correctly when the
        pre-cache pipeline changes, instead of silently reusing stale files.
        """
        parts = []
        for op in self.ops:
            f = op["f"]
            if isinstance(f, CacheBullet):
                break
            if not op["active"]:
                continue
            name = (
                getattr(f, "__qualname__", None)
                or getattr(f, "__name__", None)
                or f.__class__.__name__
            )
            parts.append(name)
        if not parts:
            return ""
        return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]

    def add(self, *funcs):
        for f in funcs:
            self.ops.append({"f": f, "active": True})
        return self

    def deactivate_op(self, index):
        if not isinstance(index, list):
            index = [index]
        for j, op in enumerate(self.ops):
            if j in index:
                op["active"] = False

    def reactivate_op(self, index):
        if not isinstance(index, list):
            index = [index]
        for j, op in enumerate(self.ops):
            if j in index:
                op["active"] = True

    def reactivate_all(self):
        for op in self.ops:
            op["active"] = True

    def __call__(self, **kwargs):
        batch_elements = kwargs
        for op in self.ops:
            if not op["active"]:
                continue
            if isinstance(op["f"], CacheBullet):
                continue
            batch_elements = op["f"](**batch_elements)
        return batch_elements

    def __lshift__(self, other):
        return self.add(other)

    def precache_call(self, **kwargs):
        if not self.has_bullet_cache:
            return kwargs
        else:
            for op in self.ops:
                if not op["active"]:
                    continue
                if isinstance(op["f"], CacheBullet):
                    break
                kwargs = op["f"](**kwargs)

            return kwargs

    def postcache_call(self, **kwargs):
        if not self.has_bullet_cache:
            return self(**kwargs)
        else:
            for i, op in enumerate(self.ops):
                if i <= self._index_bullet or not op["active"]:
                    continue
                else:
                    kwargs = op["f"](**kwargs)
            return kwargs

    @property
    def has_bullet_cache(self):
        for i, op in enumerate(self.ops):
            if isinstance(op["f"], CacheBullet):
                self._index_bullet = i
                return True
        self._index_bullet = 0
        return False

    def __str__(self):
        output = ""
        for i, o in enumerate(self.ops):
            output += (
                "%i_" % i
                + str(o["f"])
                + " STATUS: "
                + ("Active" if o["active"] else "Inactive")
                + " \n"
            )
        return output

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.ops)
