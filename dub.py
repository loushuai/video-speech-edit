import os
import cv2
import torch
import glob
import copy
import numpy as np
from transformers import WhisperModel
from tqdm import tqdm

# os.environ["PYTHONPATH"] = os.pathsep.join([
#     ".",
#     "MuseTalk",
# ])

initial_cwd = os.getcwd()
new_directory_path = "MuseTalk"
os.chdir(new_directory_path)

from MuseTalk.musetalk.utils.blending import get_image
from MuseTalk.musetalk.utils.face_parsing import FaceParsing
from MuseTalk.musetalk.utils.audio_processor import AudioProcessor
from MuseTalk.musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
from MuseTalk.musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder


def vid_to_frames(video_path, temp_dir="tmp"):
    """
    Extract frames from the input video and save them as images in the temporary directory.

    Args:
        video_path (str): Path to the input video file.
        tmp_dir (str): Path to the temporary directory to save extracted frames.
    """
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    input_basename = os.path.basename(video_path).split('.')[0]
    save_dir_full = os.path.join(temp_dir, input_basename)
    os.makedirs(save_dir_full, exist_ok=True)
    cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {save_dir_full}/%08d.png"
    print(cmd)
    os.system(cmd)
    input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))

    return input_img_list, save_dir_full


def inference(video_path, audio_path, output_path, tmp_dir="tmp", device='cuda'):
    """
    Perform inference on the given video and audio files to generate a dubbed video.

    Args:
        video_path (str): Path to the input video file.
        audio_path (str): Path to the input audio file.
        output_path (str): Path to save the output dubbed video file.
    """
    # Load model weights
    vae, unet, pe = load_all_model(
        unet_model_path="./models/musetalkV15/unet.pth", 
        vae_type="sd-vae",
        unet_config="./models/musetalkV15/musetalk.json",
        device=device
    )

    # Move models to specified device
    pe = pe.to(device)
    vae.vae = vae.vae.to(device)
    unet.model = unet.model.to(device)

    # Initialize audio processor and Whisper model
    audio_processor = AudioProcessor(feature_extractor_path="./models/whisper")
    weight_dtype = unet.model.dtype
    whisper = WhisperModel.from_pretrained("./models/whisper")
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)

    # Set result save paths
    output_basename = os.path.basename(output_path).split('.')[0]
    result_img_save_path = os.path.join(tmp_dir, output_basename)
    os.makedirs(result_img_save_path, exist_ok=True)

    # Extract frames from video
    input_img_list, save_dir_full = vid_to_frames(video_path, temp_dir=tmp_dir)

    fps = get_video_fps(video_path)

    # Extract audio features
    whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
    whisper_chunks = audio_processor.get_whisper_chunk(
        whisper_input_features, 
        device, 
        weight_dtype, 
        whisper, 
        librosa_length,
        fps=fps,
        audio_padding_length_left=2,
        audio_padding_length_right=2,
    )

    print("Extracting landmarks... time-consuming operation")
    coord_list, frame_list = get_landmark_and_bbox(input_img_list, 0)
    print("Landmarks extraction done.")
    print(f"Number of frames: {len(frame_list)}")  

    # Process each frame
    input_latent_list = []
    for bbox, frame in zip(coord_list, frame_list):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        y2 = y2 + 10
        y2 = min(y2, frame.shape[0])
        crop_frame = frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame, (256,256), interpolation=cv2.INTER_LANCZOS4)
        latents = vae.get_latents_for_unet(crop_frame)
        input_latent_list.append(latents)

    # Smooth first and last frames
    frame_list_cycle = frame_list + frame_list[::-1]
    coord_list_cycle = coord_list + coord_list[::-1]
    input_latent_list_cycle = input_latent_list + input_latent_list[::-1]

    # Batch inference
    print("Starting inference")
    video_num = len(whisper_chunks)
    batch_size = 8
    gen = datagen(
        whisper_chunks=whisper_chunks,
        vae_encode_latents=input_latent_list_cycle,
        batch_size=batch_size,
        delay_frame=0,
        device=device,
    )

    res_frame_list = []
    total = int(np.ceil(float(video_num) / batch_size))

    # Execute inference
    for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=total)):
        audio_feature_batch = pe(whisper_batch)
        latent_batch = latent_batch.to(dtype=unet.model.dtype)

        pred_latents = unet.model(latent_batch, torch.tensor([0], device=device), encoder_hidden_states=audio_feature_batch).sample
        recon = vae.decode_latents(pred_latents)
        for res_frame in recon:
            res_frame_list.append(res_frame)

    # Pad generated images to original video size
    print("Padding generated images to original video size")
    for i, res_frame in enumerate(tqdm(res_frame_list)):
        bbox = coord_list_cycle[i%(len(coord_list_cycle))]
        ori_frame = copy.deepcopy(frame_list_cycle[i%(len(frame_list_cycle))])
        x1, y1, x2, y2 = bbox
        y2 = y2 + 10
        y2 = min(y2, frame.shape[0])
        try:
            res_frame = cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))
        except Exception as e:
            print(f"Error resizing frame {i}: {e}")
            continue
        
        # Merge results with version-specific parameters
        fp = FaceParsing(
            left_cheek_width=90,
            right_cheek_width=90
        )
        combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], mode="jaw", fp=fp)
        cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", combine_frame)

    # Save prediction results
    temp_vid_path = f"{tmp_dir}/temp_{output_basename}.mp4"
    cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 {temp_vid_path}"
    print("Video generation command:", cmd_img2video)
    os.system(cmd_img2video)
    
    cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i {temp_vid_path} {output_path}"
    print("Audio combination command:", cmd_combine_audio) 
    os.system(cmd_combine_audio)

# test
if __name__ == "__main__":
    video_path = "../data/video/gdg_clip_no_audio.mp4"
    audio_path = "../data/audio/gdg.wav"
    output_path = "../results/output_dubbed_video.mp4"
    inference(video_path, audio_path, output_path, tmp_dir="../tmp", device='cuda')
