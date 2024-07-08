from superduperdb import Model


class PModel(Model):
    def predict(self, x) -> int:
        return x + 1
