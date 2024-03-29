import network
import torch
from torchsummary import summary
import cv2
import numpy as np

def show_seg_result(img, result, palette=None):
    if palette is None:
        palette = np.random.randint(
                0, 255, size=(3, 3))
    palette[0] = [0, 0, 0]
    palette[1] = [0, 255, 0]
    palette[2] = [255, 0, 0]
    palette = np.array(palette)
    assert palette.shape[0] == 3 # len(classes)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    
    color_area = np.zeros((result[0].shape[0], result[0].shape[1], 3), dtype=np.uint8)

    color_area[result[0] > 100] = [0, 255, 200] #DA
    color_area[result[1] > 100] = [255, 0, 0] #LL
    
    color_seg = color_area[..., ::-1]
    color_mask = np.mean(color_seg, 2)

    DA_mask = (result[0] > 100)
    LL_mask = (result[1] > 100)
    img[DA_mask] = img[DA_mask] * 0.6 + color_seg[DA_mask] * 0.4 # Lighter blend for DA
    img[LL_mask] = img[LL_mask] * 0.2 + color_seg[LL_mask] * 0.8  # Original blend for LL
    return 

def Run(model,img):
    img_rs=img.copy()
    img = cv2.resize(img, (640, 360))

    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img=torch.from_numpy(img)
    img = torch.unsqueeze(img, 0)  # add a batch dimension
    img=img.cuda().float() / 255.0
    img = img.cuda()
    with torch.no_grad():
        img_out = model(img)

    x0 = img_out[:, :2, :, :]  # Drivable area
    x1 = img_out[:, 1:, :, :]  # Lane line segmentation


    _,da_predict=torch.max(x0, 1)
    _,ll_predict=torch.max(x1, 1)
    
    '''
    x0: torch.Size([1, 2, 360, 640])
    da_predict: torch.Size([1, 360, 640])
    '''

    DA = da_predict.byte().cpu().data.numpy()[0]*255
    LL = ll_predict.byte().cpu().data.numpy()[0]*255
    DA = cv2.resize(DA, (1280, 720))
    LL = cv2.resize(LL, (1280, 720))

    show_seg_result(img_rs, (DA, LL))    
    return img_rs

path = '/home/ceec/tri/twinlite_github/image/b1d22449-117aa773.jpg'
model = network.modeling.__dict__['deeplabv3_resnet50'](num_classes=7, output_stride=8)
model = model.cuda()
model.eval()

img = cv2.imread(path)
output_test = model(torch.rand(6,3,500,500).cuda().float()/255.0)
print(output_test.shape)
output=Run(model, img)
cv2.imwrite('./test.jpg', output)
