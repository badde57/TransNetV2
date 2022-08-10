import time
import torch
import torchvision
from pytorchvideo.data.encoded_video import EncodedVideo
from transnetv2_pytorch import TransNetV2
video_path = "/home/deng/data/video_info_extraction/code/utils/test.mp4"

video = EncodedVideo.from_path(video_path)
_EPS = 1e-9
# test_video.duration + self._EPS
clip_start_sec = 0.0 # secs 
clip_duration = 3.33 # secs 30FPS and we need 100 images each time
video_duration = video.duration + _EPS

no_clips = video_duration//clip_duration

input_video_frames = video.get_clip(start_sec=clip_start_sec, end_sec=clip_start_sec + clip_duration)['video'] #CTHW



trans = torchvision.transforms.Resize([27, 48])
model = TransNetV2()
state_dict = torch.load("transnetv2-pytorch-weights.pth")
model.load_state_dict(state_dict)
model.eval().cuda()

with torch.no_grad():
    # shape: batch dim x video frames x frame height x frame width x RGB (not BGR) channels
    start = time.time()
    input_video = input_video_frames
    input_video = trans(input_video)
    input_video = input_video.permute([1,2,3,0]).unsqueeze(0).to(torch.uint8)
    get_data_time = time.time()
    print('Get videp clip time: ',get_data_time - start)
    single_frame_pred, all_frame_pred = model(input_video.cuda())
    single_frame_pred = torch.sigmoid(single_frame_pred).cpu().numpy()
    all_frame_pred = torch.sigmoid(all_frame_pred["many_hot"]).cpu().numpy()
    print('Inference time: ', time.time() - get_data_time)
    print(single_frame_pred.shape)
    print(all_frame_pred.shape)
video.close()