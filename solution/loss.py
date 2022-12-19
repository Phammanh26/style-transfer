import torch

def gram_matrix(tensor):
    """ Calculate the Gram Matrix of a given tensor 
        Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
    """
    
    ## get the batch_size, depth, height, and width of the Tensor
    ## reshape it, so we're multiplying the features for each channel
    ## calculate the gram matrix
    b, d, h, w = tensor.size()

    tensor = tensor.view(d*b, h*w)
    gram = torch.mm(tensor, tensor.t())
    
    return gram 

def compute_style_loss(target_features: torch.Tensor, style_grams: torch.Tensor):

    style_loss = 0
    style_weights = {'conv1_1': 1.,
                    'conv2_1': 0.75,
                    'conv3_1': 0.2,
                    'conv4_1': 0.2,
                    'conv5_1': 0.2}
        
    # then add to it for each layer's gram matrix loss
    for layer in style_weights:
        # get the "target" style representation for the layer
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        _, d, h, w = target_feature.shape
        # get the "style" style representation
        style_gram = style_grams[layer]
        # the style loss for one layer, weighted appropriately
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
        
        # add to the style loss
        style_loss += layer_style_loss / (d * h * w)
    
    return style_loss

def compute_content_loss(target_features: torch.Tensor, content_features: torch.Tensor):
    content_loss = 0
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
    return content_loss