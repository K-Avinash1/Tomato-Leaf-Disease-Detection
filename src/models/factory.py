from src.models.custom_cnn import TomatoCNN


def build_model(name: str, num_classes: int = 10, dropout: float = 0.3):
    if name == "custom_cnn":
        return TomatoCNN(num_classes, dropout)
    raise ValueError(f"Unknown model: {name}")


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())