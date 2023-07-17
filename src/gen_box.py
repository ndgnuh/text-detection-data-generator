import torch
from torchvision import ops
from PIL import ImageFont
import random
import numpy as np
import cv2


def gen_imgInteval(image_in: np.ndarray):
    
    if len(image_in.shape) == 2:
        image_process = np.asarray(image_in)
    elif image_in.shape[2] == 3:
        image_process = cv2.cvtColor(image_in, cv2.COLOR_RGB2BGR)
    elif image_in.shape[2] == 4:
        image_process = cv2.cvtColor(image_in, cv2.COLOR_RGBA2RGB)
    else:
        raise ValueError("wrong image format")

    src_h, src_w = image_process.shape[:2]
    
    # key_points = self._fast.detect(image_process, None)
    kp_image = cv2.Canny(image_process, 50, 150) // 255
    sum_arr = np.zeros((src_h, src_w), np.float32)
    image_integral = cv2.integral(kp_image, sum_arr, cv2.CV_32FC1)

    return image_integral

    
def box_gen(image_size,
            imageIntegral: np.ndarray,
            text : str,
            font: ImageFont,
            scale = (0.02, 0.5),
            font_range = (5, 50),
            max_intergral = 20):
    W, H = image_size
    h_range = (H*scale[0], H*scale[1])
    count_height_loop = 0
    count_intergral_loop = 0
    while True:
        if count_height_loop > 1500 or count_intergral_loop > 1500:
            return {
                    "box": None,
                    "text": text,
                    "fontsize": fontsize,
                    "font": font
                    } 
            
        # if count_intergral_loop > 10000:
        #     # raise ValueError("Can't find suitable integral") 

        fontsize = random.randint(*font_range)
        font_copy = font.font_variant(size=fontsize)
        x1, y1, x2, y2 = font_copy.getbbox(text, stroke_width=0)
        text_width, text_height = x2 - x1, y2 - x1

        # ROI
        x1 = random.uniform(0, W - text_width)
        y1 = random.uniform(0, H - text_height)
        x2 = min(x1 + text_width, W - 1)
        y2 = min(y1 + text_height, H - 1)
        if x1<0 or y1<0 or x2<0 or y2<0:
            continue
        integral = imageIntegral[int(y2), int(x2)] - imageIntegral[int(y2), int(x1)] - imageIntegral[int(y1), int(x2)] + imageIntegral[int(y1), int(x1)]

        if text_height < h_range[0] or text_height > h_range[1]:
            count_height_loop+=1
            continue
        elif integral > max_intergral:
            count_intergral_loop+=1
            continue
        else:
            break

    return {
        "box": [x1, y1, x2, y2],
        "text": text,
        "fontsize": fontsize,
        "font": font
    }


def text_gen():
    #Texts load
    texts_path = "./train_trg.txt"
    with open(texts_path,"r",encoding="utf-8") as f:
        text = random.choice(f.readlines()).strip('\n')
        # texts_data = [x.strip('\n') for x in f.readlines()]
    
    return text

def random_multi_boxes(image_size, 
                       imageIntegral: np.ndarray,
                       n: int, 
                       font_list: list, 
                       scale = (0.05, 0.5),
                       font_range = (10, 100),
                       max_intergral = 20):
    #Gen random multi rectangle box on image
    boxes_dict_list = [box_gen(image_size = image_size,
                                imageIntegral = imageIntegral,
                                text = text_gen(),
                                font = random.choice(font_list),
                                scale = scale,
                                font_range = font_range,
                                max_intergral = max_intergral) for _ in range(n)]

    boxes_dict_list = [x for x in boxes_dict_list if x['box'] is not None]
    if len(boxes_dict_list) < 1:
        return []
    rois = torch.tensor([x["box"] for x in boxes_dict_list])
    scores = torch.randn(rois.shape[0])
    keep = ops.nms(rois, scores, 0)
    # rois = rois[keep].type(torch.long)
    boxes_dict_list = [boxes_dict_list[i] for i in keep.tolist()] 
    return boxes_dict_list

def random_roi(
    image_size,
    scale=(0.05, 0.5),
    ratio=(0.5, 0.83),
):  
    #Gen random a rectangle box on image
    # SR
    scale = random.uniform(*scale)
    ratio = random.uniform(*ratio)

    # Template size
    W, H = image_size
    h = H * scale
    w = h * ratio

    # ROI
    x1 = random.uniform(0, W - w)
    y1 = random.uniform(0, H - h)
    x2 = min(x1 + w, W - 1)
    y2 = min(y1 + h, H - 1)
    return x1, y1, x2, y2


def random_multi_roi(size, n, scale, ratio):
    #Gen random multi rectangle box on image
    
    rois = torch.tensor([random_roi(size, scale, ratio) for _ in range(n)])
    scores = torch.randn(rois.shape[0])
    keep = ops.nms(rois, scores, 0)
    
    rois = rois[keep].type(torch.long)
    return rois.tolist()

