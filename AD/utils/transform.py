import torchvision.transforms as transforms
from PIL import Image


def data_transform(file_name, device):
    transform = transforms.Compose([transforms.Resize(32),
                                    transforms.CenterCrop(32),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
                                    ])
    
    # 해당 PATH에서 이미지 열기
    img = Image.open("AD/save_img" + "/" + file_name).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)

    return img