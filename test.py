from model.ow_model import OWmodel
from model.resnet_model import ResOWModel
from torchvision import transforms 
import numpy as np
from PIL import Image
import torch
from glob import glob


if __name__ == '__main__':
    image = Image.open('test1.png').convert('RGB')
    re = image.resize((256,256))
    names = glob('heroes/*')
    t = transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
    ])
    x_test = t(re).unsqueeze(0).cuda()

    model = ResOWModel() #OWmodel()
    model.load_state_dict(torch.load('checkpoints/res_model.pth'))
    model.cuda()
    model.eval()

    with torch.no_grad():
        result = model(x_test)
        idx = torch.argmax(result)

        print(idx)
        print(names[idx.data])
        print(result[0][idx.data])
        print(torch.topk(result.squeeze(0), 3).indices)





        

