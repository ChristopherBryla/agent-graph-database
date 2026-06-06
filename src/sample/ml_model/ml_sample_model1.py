
from genesys.contract.data_contract import ContractField, DataContract


class LinearRegression:
    """Simple linear regression with 4 input features."""

    # ------------------------------------------------------------------
    # Data contract — declare what this model expects as input.
    # The component manager validates incoming data against this before
    # calling fit() or predict(). Add or change fields here as the model
    # evolves and the pipeline will enforce it automatically.
    # ------------------------------------------------------------------
    contract = DataContract(
        model_name="linear_regression",
        version="1.0.0",
        fields=[
            ContractField("x1", float, required=True, min_value=0.0, max_value=100.0, description="feature 1"),
            ContractField("x2", float, required=True, min_value=0.0, max_value=100.0, description="feature 2"),
            ContractField("x3", float, required=True, min_value=0.0, max_value=100.0, description="feature 3"),
            ContractField("x4", float, required=True, min_value=0.0, max_value=100.0, description="feature 4"),
        ],
    )

    output_contract = DataContract(
        model_name="linear_regression_output",
        version="1.0.0",
        fields=[
            ContractField("weights", list, required=True, description="learned weight vector"),
            ContractField("bias", float, required=True, description="learned bias term"),
            ContractField("epochs", int, required=True, min_value=1.0, description="training epochs run"),
            ContractField("learning_rate", float, required=True, min_value=0.0, description="learning rate used"),
        ],
    )

    def __init__(self) -> None:
        self.weights: list[float] = [0.0, 0.0, 0.0, 0.0]
        self.bias: float = 0.0

    def predict(self, x1: float, x2: float, x3: float, x4: float) -> float:
        inputs = [x1, x2, x3, x4]
        return sum(w * x for w, x in zip(self.weights, inputs)) + self.bias

    def fit(
        self,
        data: list[tuple[float, float, float, float]],
        targets: list[float],
        learning_rate: float = 0.01,
        epochs: int = 1000,
    ) -> None:
        n = len(data)
        for _ in range(epochs):
            for inputs, target in zip(data, targets):
                prediction = self.predict(*inputs)
                error = prediction - target
                self.weights = [
                    w - learning_rate * error * x
                    for w, x in zip(self.weights, inputs)
                ]
                self.bias -= learning_rate * error
