import time
import torch
import numpy as np
from transnetv2_pytorch import TransNetV2
import os
import cv2
import torchvision

def save_scenes(scenes, video_path, out_path):
     if len(scenes) > 0:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        cap = cv2.VideoCapture(video_path)
        assert(cap.isOpened())
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for f_idx in scenes:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            _, frame = cap.read()
            cv2.imwrite(os.path.join(out_path, f'frame_{f_idx}.jpg'),frame)
        cap.release()

def get_scenes(model, video_path):
    video_frames, single_frame_predictions, all_frame_predictions = \
    model.predict_video(video_path)
    video_frames = video_frames.cpu().detach().numpy()
    single_frame_predictions = single_frame_predictions.cpu().detach().numpy()
    all_frame_predictions = all_frame_predictions.cpu().detach().numpy()
    scenes = model.predictions_to_scenes(single_frame_predictions)
    return scenes

def scene_detector(model, video_path, out_path):
    start = time.time()
    scenes = get_scenes(model, video_path)
    print(time.time()-start, 'get_scenes')

    np.savetxt(f'{file_name}.csv', scenes, delimiter=',')

    scenes = scenes.reshape(-1) #???? fist image + last image per scene or ?????
    start = time.time()
    save_scenes(scenes, video_path, out_path)
    print(time.time()-start, 'save_scenes')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

video_path_list = ['./test.mp4','./test.mp4']
model = TransNetV2(device=device)
state_dict = torch.load("transnetv2-pytorch-weights.pth")
model.load_state_dict(state_dict)


with torch.no_grad():
    i = 0
    for video_path in video_path_list:
        file_name = video_path.split('/')[-1].split('.')[0]
        scene_detector(model, video_path, f'./{file_name}{i}')
        i+=1
