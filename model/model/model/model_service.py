from model.model_adapter import train_model_sklearn


def build_model(data, target: str):
    """
    Orchestrates the training of the model.
    """
    model, le = train_model_sklearn(data, target)
    return model, le
