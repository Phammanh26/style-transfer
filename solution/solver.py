import torch
import torch.optim as optim
from torchvision import models
from solution.feature import get_features
from solution.loss import gram_matrix, compute_content_loss, compute_style_loss
import config

def load_model():
    '''
    func: load model
    '''
    # get the "features" portion of VGG19 (we will not need the "classifier" portion)
    model = models.vgg19(pretrained=True).features
    # freeze all VGG parameters since we're only optimizing the target image
    for param in model.parameters():
        param.requires_grad_(False)

    # move the model to GPU, if available
    model.to(config.device)

    return model


class TransferSolver():
    def __init__(self):
        self.model = None
        self.setup()
    
    def setup(self):
        self.model = load_model()

    def solve(self, content: torch.Tensor, style: torch.Tensor, 
                content_weight = 1, style_weight =  1e6, 
                lr =  0.003, steps = 2000) -> torch.Tensor:
        '''
        func: a solution for transfer image
        - content: image content was converted to tensor
        - style: image style was converted to tensor
        - content_weight (option): rate weight of content image affect to loss value
        - style_weight (option): rate weight of style image affect to loss value
        - lr (option): learning rate param for optimize loss
        - steps (option)): num step for update target image

        output:
        - target: image was transfered from content image + style image
        '''
        # get content and style features only once before training
        content_features = get_features(content, self.model)
        style_features = get_features(style, self.model)

        # calculate the gram matrices for each layer of our style representation
        style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

        # create a third "target" image and prep it for change
        target = content.clone().requires_grad_(True).to(config.device)

        # iteration hyperparameters
        optimizer = optim.Adam([target], lr=lr)
        
        for ii in range(1, steps+1):
            # get the features from your target image
            target_features = get_features(target, self.model)
            
            # the content loss
            content_loss = compute_content_loss(target_features, content_features)
            # the style loss
            style_loss = compute_style_loss(target_features,style_grams)
            
            # calculate the *total* loss
            total_loss = content_weight * content_loss + style_weight * style_loss
            
            # update your target image
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        return target