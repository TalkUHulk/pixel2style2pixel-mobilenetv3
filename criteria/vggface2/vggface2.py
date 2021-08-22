import torch
import torch.nn as nn
from criteria.vggface2 import resnet
from criteria.vggface2.utils import load_state_dict, vgg_preprocess


class VGGFaceLoss(nn.Module):
    def __init__(self, vgg_weight_file):
        super(VGGFaceLoss, self).__init__()
        self.net = self.get_network(vgg_weight_file).to("cuda")
        self.net.eval()
        self.net.fc.reset_parameters()
        for param in self.net.parameters():
            param.requires_grad = False

    def get_network(self, model_dir):
        model = resnet.resnet50()
        load_state_dict(model, model_dir)
        return model

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x_vgg = vgg_preprocess(x)
        y_vgg = vgg_preprocess(y)
        x_fea = self.net(x_vgg)
        y_fea = self.net(y_vgg)
        return torch.mean(torch.abs(x_fea - y_fea))