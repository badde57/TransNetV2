import time
import torch
import numpy as np
from numpy import genfromtxt
from transnetv2_pytorch import TransNetV2
import imageio.v3 as iio
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
video_path = "./test.mp4"
file_name = video_path.split('/')[-1].split('.')[0]

model = TransNetV2(device=device)
state_dict = torch.load("transnetv2-pytorch-weights.pth")
model.load_state_dict(state_dict)

with torch.no_grad():
    start = time.time()
    video_frames, single_frame_predictions, all_frame_predictions = \
    model.predict_video(video_path)
    video_frames = video_frames.cpu().detach().numpy()
    single_frame_predictions = single_frame_predictions.cpu().detach().numpy()
    all_frame_predictions = all_frame_predictions.cpu().detach().numpy()
    scenes = model.predictions_to_scenes(single_frame_predictions)
    # model.visualize_predictions(video_frames, predictions=(single_frame_predictions, all_frame_predictions)).save("img1.png")
    print(time.time() - start,"detection finished!")
    start = time.time()
    scenes = scenes.reshape(-1) #???? fist image + last image per scene or ?????
    np.savetxt(f'{file_name}.csv', scenes, delimiter=',')
    if len(scenes) > 0:
        if not os.path.exists(f'{file_name}'):
            os.makedirs(f'{file_name}')
        for idx, frame in enumerate(iio.imiter(video_path)):
            if idx in scenes:
                iio.imwrite(f"{file_name}/frame{idx:03d}.jpg", frame) # Takes a long time!
    print(time.time() - start,'Image saved!')
    # model.predictions_to_scenes(single_frame_predictions)

# [TransNetV2] Extracting frames from ./test.mp4
# [TransNetV2] Processing video frames 24965/24965
# 33.428701400756836 detection finished!
# 73.72723340988159 Image saved!