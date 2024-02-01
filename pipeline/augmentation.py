import torchvision.transforms as T
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

class CustomTransform(object):
    def __init__(self, load_dim=286, target_dim=256):
        self.transform_train = T.Compose([
            T.Resize((load_dim, load_dim), antialias=True),
            T.RandomCrop((target_dim, target_dim)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2,
                          saturation=0.2, hue=0.1),
        ])
        self.transform = T.Resize((target_dim, target_dim), antialias=True)   
    def __call__(self, img, stage):
        if stage == "fit":
            img = self.transform_train(img)
        else:
            img = self.transform(img)
        return img * 2 - 1
    
# def show_img(img_tensor, nrow, title=""):
#     img_tensor = img_tensor.detach().cpu() * 0.5 + 0.5
#     img_grid = make_grid(img_tensor, nrow=nrow).permute(1, 2, 0)
#     plt.figure(figsize=(10, 7))
#     plt.imshow(img_grid)
#     plt.axis("off")
#     plt.title(title)
#     plt.show()
