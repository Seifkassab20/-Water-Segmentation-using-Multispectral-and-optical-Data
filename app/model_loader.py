import torch
from models.unet import UNet
from models.deeplab import DeepLab

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_models_cache = {}

def load_model(model_name):

    if model_name in _models_cache:
        return _models_cache[model_name]

    if model_name == "unet":
        checkpoint = torch.load("models/unet.pth", map_location=DEVICE)

        model = UNet(in_channels=6, out_channels=2)

    elif model_name == "deeplab":
        checkpoint = torch.load("models/deeplab.pth", map_location=DEVICE)

        model = DeepLab(n_channels=6, n_classes=2)

    else:
        raise ValueError("Invalid model name")

    state_dict = checkpoint["model_state_dict"]
    
    if model_name == "deeplab":
        # If the saved checkpoint was from the inner model directly, prefix the keys 
        if not list(state_dict.keys())[0].startswith("model."):
            state_dict = {f"model.{k}": v for k, v in state_dict.items()}
            
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()

    _models_cache[model_name] = model
    return model