import torch


def model_predict(model, inputs):
    """
    Takes a pre-trained model and preprocessed inputs
    Returns the predicted class index
    The model should be already in evaluation mode
    Inputs should be pre-processed according to the model's requirements
    """

    with torch.no_grad():
        output = model(inputs)
        if hasattr(output, "logits"):
            logits = output.logits
        elif isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
    prediction = logits.argmax(dim=-1).item()
    return prediction


def get_target(true_label: int, n_classes: int) -> torch.Tensor:
    # Generate a random target label different from the true label
    target = torch.randint(0, n_classes, (1,))
    while target == true_label:
        target = torch.randint(0, n_classes, (1,))
    return target
