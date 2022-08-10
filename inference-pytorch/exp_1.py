import time
import torch
import numpy as np
from transnetv2_pytorch import TransNetV2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

video_path = "/home/deng/data/video_info_extraction/code/utils/test.mp4"
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
    np.savetxt('test.csv', scenes, delimiter=',') 
    # model.visualize_predictions(video_frames, predictions=(single_frame_predictions, all_frame_predictions)).save("img1.png")
    
    print(time.time() - start,"single_frame_predictions",single_frame_predictions.shape)
    print('scenes', scenes.shape)
    # model.predictions_to_scenes(single_frame_predictions)