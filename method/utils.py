import torch

class BCESimilarityToConceptTarget:
    def __init__(self, features):
        self.features = features
    
    def __call__(self, model_output):
        bce = torch.nn.BCELoss()
        return -bce(0.5 * (model_output + 1), 0.5 * (self.features + 1))

class mask_BCESimilarityToConceptTarget:
    def __init__(self, features):
        self.features = features
    
    def __call__(self, model_output):
        bce = torch.nn.BCELoss()
        sign_model_output = model_output.detach().sign()
        model_output = torch.where((sign_model_output==self.features).float()>0, model_output, self.features)
        loss = -bce(0.5 * (model_output + 1), 0.5 * (self.features + 1))
        return loss
