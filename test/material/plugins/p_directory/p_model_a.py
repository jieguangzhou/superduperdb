from superduperdb import Model


class PModelA(Model):
    def predict(self, x) -> int:
        return x + 1
