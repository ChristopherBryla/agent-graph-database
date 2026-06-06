import random


def generate_sample_data(
    n: int = 50,
    seed: int = 42,
) -> tuple[list[dict[str, float]], list[float]]:
    """Generate synthetic training data for the linear regression model.

    The target is roughly x1 + x2 + x3 + x4 with a small amount of noise,
    so after training the model should learn weights close to 1.0 each.
    Each row is returned as a named dict so it can be validated against the
    model's DataContract before training.
    """
    rng = random.Random(seed)

    data: list[dict[str, float]] = []
    targets: list[float] = []

    for _ in range(n):
        x1 = rng.uniform(0.0, 10.0)
        x2 = rng.uniform(0.0, 10.0)
        x3 = rng.uniform(0.0, 10.0)
        x4 = rng.uniform(0.0, 10.0)
        noise = rng.gauss(0.0, 0.5)                # small random noise
        target = x1 + x2 + x3 + x4 + noise

        data.append({"x1": x1, "x2": x2, "x3": x3, "x4": x4})
        targets.append(target)

    return data, targets

