import numpy as np
import torch
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

from constants import *


def view_test_images(model_file_name, data_loader):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    pbar = tqdm(total=len(data_loader), leave=False, desc='Inference Test')

    model = torch.load(f'models/{model_file_name}')
    model.eval()

    preds_list = []
    targets_list = []

    # Deactivate the autograd engine
    with torch.no_grad():
        for images, targets in data_loader:
            images = torch.stack(images).to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            preds = model(images)

            preds_list += preds
            targets_list += targets

            pbar.update()

    pbar.close()

    print('hi')
    # loaded_model = get_model_instance_segmentation(num_classes=len(CLASS_DICT) + 1)
    # loaded_model.load_state_dict(torch.load("models/" + filename))
    # img, targets = test_dataset[idx]
    # img_name = ''.join(chr(i) for i in targets['name'])
    # print(img_name)
    #
    # label_boxes = targets['boxes'].detach().cpu().numpy()
    # # put the model in evaluation mode
    # loaded_model.eval()
    # with torch.no_grad():
    #     prediction = loaded_model([img])
    # image = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    # draw = ImageDraw.Draw(image)
    # # draw groundtruth
    # for elem in range(len(label_boxes)):
    #     label = test_dataset[idx][1]["labels"][elem].cpu().numpy()
    #     label = list(CLASS_DICT.keys())[
    #         list(CLASS_DICT.values()).index(label)]
    #     draw.rectangle([(label_boxes[elem][0], label_boxes[elem][1]),
    #                     (label_boxes[elem][2], label_boxes[elem][3])], outline="green", width=3)
    #     font = ImageFont.truetype(
    #         "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf", 50)
    #     draw.text((label_boxes[elem][0], label_boxes[elem][1]),
    #               text=label + " " + str(1), font=font, fill=(255, 255, 0, 0))
    # for element in range(len(prediction[0]["boxes"])):
    #     boxes = prediction[0]["boxes"][element].cpu().numpy()
    #     score = np.round(prediction[0]["scores"]
    #                      [element].cpu().numpy(), decimals=4)
    #     label = prediction[0]["labels"][element].cpu().numpy()
    #     label = list(CLASS_DICT.keys())[
    #         list(CLASS_DICT.values()).index(label)]
    #     if score > 0.5:
    #         draw.rectangle(
    #             [(boxes[0], boxes[1]), (boxes[2], boxes[3])], outline="red", width=5)
    #         font = ImageFont.truetype(
    #             "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf", 50)
    #         draw.text((boxes[0] - 1, boxes[3]), text=f"{label} {score}", font=font, fill=(255, 255, 255, 0))
    #
    # image = image.save("data/inference/" + img_name)
