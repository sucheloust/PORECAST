"""Hyperparameter schedules (constant or linear ramp) and the CLI parser."""


# Hyperparameter schedules: constant, or linear ramp "start:end:epochs".
# Epoch is the training epoch (continues across --resume).
class ConstantSchedule:
    def __init__(self, value):
        self.value = float(value)

    def __call__(self, epoch):
        return self.value

    def __repr__(self):
        return f"const({self.value})"


class LinearSchedule:
    def __init__(self, start, end, epochs):
        self.start = float(start)
        self.end = float(end)
        self.epochs = int(epochs)

    def __call__(self, epoch):
        if self.epochs <= 0 or epoch >= self.epochs:
            return self.end
        if epoch <= 0:
            return self.start
        frac = epoch / self.epochs
        return self.start + (self.end - self.start) * frac

    def __repr__(self):
        return f"linear({self.start}->{self.end} over {self.epochs}ep)"


def parse_schedule(spec):
    """
    Parse a CLI schedule spec into a schedule object.
      "0.5"          -> ConstantSchedule(0.5)
      "0.3:0.5:40"   -> LinearSchedule(0.3, 0.5, 40)
    """
    spec = str(spec).strip()
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Schedule '{spec}' must be 'start:end:epochs' or a single "
                f"number.")
        start, end, epochs = parts
        return LinearSchedule(float(start), float(end), int(epochs))
    return ConstantSchedule(float(spec))
