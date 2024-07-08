from superduperdb import Model


class PModelB(Model):
    def predict(self, x) -> int:
        return x + 1
