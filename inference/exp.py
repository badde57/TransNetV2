from transnetv2 import TransNetV2

# location of learned weights is automatically inferred
# add argument model_dir="/path/to/transnetv2-weights/" to TransNetV2() if it fails
model = TransNetV2()
video_frames, single_frame_predictions, all_frame_predictions = \
    model.predict_video("/home/deng/data/video_info_extraction/code/utils/test.mp4")

# # or
# video_frames = ... # np.array, shape: [n_frames, 27, 48, 3], dtype: np.uint8, RGB (not BGR)
# single_frame_predictions, all_frame_predictions = \
#     model.predict_frames(video_frames)
list_of_scenes = model.predictions_to_scenes(predictions=single_frame_predictions)
print(list_of_scenes)
# pil_image = model.visualize_predictions(
#     video_frames, predictions=(single_frame_predictions, all_frame_predictions))