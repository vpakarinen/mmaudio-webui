from pathlib import Path

import gradio as gr
import torchaudio
import logging
import shutil
import torch
import uuid
import time
import sys
import os

from mmaudio.eval_utils import (all_model_cfg, generate, load_image, load_video, setup_eval_logging)
from mmaudio.model.utils.features_utils import FeaturesUtils
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import get_my_mmaudio

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mmaudio_webui.log"))
    ]
)
logger = logging.getLogger("MMAudioWebUI")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    logger.info(f"CUDA is available, using GPU: {torch.cuda.get_device_name(0)}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
    logger.info("MPS is available, using Apple Silicon GPU")
else:
    logger.warning('WARNING: CUDA/MPS are not available, running on CPU. This will be slow!')
dtype = torch.float32

logger.info("Setting up evaluation logging...")
setup_eval_logging()

logger.info(f"Loading model configuration: large_44k_v2")
model_config = all_model_cfg['large_44k_v2']
model_config.download_if_needed()

logger.info("Initializing MMAudio model... This may take a moment...")

def get_model():
    seq_cfg = model_config.seq_cfg
    net = get_my_mmaudio(model_config.model_name).to(device, dtype).eval()
    net.load_weights(torch.load(model_config.model_path, map_location=device, weights_only=True))
    
    feature_utils = FeaturesUtils(tod_vae_ckpt=model_config.vae_path,
                                  synchformer_ckpt=model_config.synchformer_ckpt,
                                  enable_conditions=True,
                                  mode=model_config.mode,
                                  bigvgan_vocoder_ckpt=model_config.bigvgan_16k_path)
    
    fm = FlowMatching(min_sigma=0.0, inference_mode='euler', num_steps=20)
    rng = torch.Generator(device='cpu')
    
    return net, feature_utils, seq_cfg, fm, rng

net, feature_utils, seq_cfg, fm, rng = get_model()
logger.info("Model initialized successfully!")

def print_tensor_info(tensor, name="Tensor"):
    if tensor is not None:
        print(f"{name} - Shape: {tensor.shape}, Type: {tensor.dtype}, Dims: {tensor.dim()}")
    else:
        print(f"{name} is None")

def generate_audio(video_file, prompt, negative_prompt="", seed=0, num_steps=20, cfg_strength=7.5, duration=8, video_start_time=0):
    """Generate audio for a video file or from text"""
    start_time = time.time()
    logger.info(f"Starting audio generation with parameters: seed={seed}, steps={num_steps}, cfg={cfg_strength}, duration={duration}s")
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    rng.manual_seed(seed)
    
    if video_file is None and not prompt:
        logger.warning("No video or text prompt provided")
        return None, "Please upload a video file or enter a text prompt."
    if video_file is not None and not prompt:
        logger.info("No prompt provided for video, using default: 'ambient sounds'")
        prompt = "ambient sounds"
    
    generation_id = str(uuid.uuid4())[:8]
    logger.info(f"Generation ID: {generation_id}")
    
    unique_id = str(uuid.uuid4())[:8]
    output_path = os.path.join(OUTPUT_DIR, f"output_{unique_id}")
    os.makedirs(output_path, exist_ok=True)
    logger.info(f"Output directory created: {output_path}")
    
    try:
        duration = float(duration)
        video_start_time = float(video_start_time)
        seed = int(seed)
        num_steps = int(num_steps)
        cfg_strength = float(cfg_strength)
    except ValueError:
        return None, "Invalid parameter values. Please check your inputs."

    # Ensure duration is at least 6 seconds for video inputs to have enough frames
    if video_file is not None and duration < 6:
        logger.warning(f"Requested duration ({duration}s) is too short for MMAudio. Setting to minimum 6 seconds.")
        duration = 6.0
    
    if duration <= 0:
        return None, "Duration must be greater than 0."
    
    try:
        with torch.inference_mode():
            custom_fm = FlowMatching(min_sigma=0.0, inference_mode='euler', num_steps=num_steps)
            gen = torch.Generator(device='cpu').manual_seed(seed)
            
            if video_file is not None:
                source_video_path = Path(video_file)
                audio_path = Path(output_path) / f"{source_video_path.stem}_audio.flac"
                
                logger.info(f"[{generation_id}] Loading video: {video_file}, starting at {video_start_time}s")
                video_info = load_video(video_file, duration, video_start_time)
                
                clip_frames = video_info.clip_frames.to(device=device, dtype=dtype).unsqueeze(0)
                sync_frames = video_info.sync_frames.to(device=device, dtype=dtype).unsqueeze(0)
                
                if device == 'cuda':
                    logger.info(f"[{generation_id}] Temporarily moving model to CPU for video-to-audio generation")
                    
                    net_cpu = net.to('cpu')
                    clip_frames_cpu = clip_frames.to('cpu')
                    sync_frames_cpu = sync_frames.to('cpu')
                    
                    waveform_float = generate(
                        clip_video=clip_frames_cpu,
                        sync_video=sync_frames_cpu,
                        text=[prompt] if prompt else None,
                        negative_text=[negative_prompt] if negative_prompt else None,
                        feature_utils=feature_utils,
                        net=net_cpu,
                        fm=custom_fm,
                        rng=gen,
                        cfg_strength=cfg_strength
                    )
                    
                    net_cpu.to(device)
                else:
                    waveform_float = generate(
                        clip_video=clip_frames,
                        sync_video=sync_frames,
                        text=[prompt] if prompt else None,
                        negative_text=[negative_prompt] if negative_prompt else None,
                        feature_utils=feature_utils,
                        net=net,
                        fm=custom_fm,
                        rng=gen,
                        cfg_strength=cfg_strength
                    )
                
                wave_int16 = (waveform_float * 32767.0).clamp(-32768.0, 32767.0).to(torch.int16).cpu()
                if wave_int16.dim() > 2:
                    wave_int16 = wave_int16.reshape(1, -1)
                torchaudio.save(str(audio_path), wave_int16, sample_rate=44100)
                
                logger.info(f"Audio generated and saved to {audio_path} in {time.time() - start_time:.2f}s")
                return audio_path, f"Audio generated successfully and saved to {audio_path}"
            else:
                if device == 'cuda':
                    logger.info(f"[{generation_id}] Temporarily moving model to CPU for text-to-audio generation")

                    net_cpu = net.to('cpu')
                    waveform_float = generate(
                        clip_video=None,
                        sync_video=None,
                        text=[prompt] if prompt else [],
                        negative_text=[negative_prompt] if negative_prompt else None,
                        feature_utils=feature_utils,
                        net=net_cpu,
                        fm=custom_fm,
                        rng=gen,
                        cfg_strength=cfg_strength
                    )
                    net_cpu.to(device)
                else:
                    waveform_float = generate(
                        clip_video=None,
                        sync_video=None,
                        text=[prompt] if prompt else [],
                        negative_text=[negative_prompt] if negative_prompt else None,
                        feature_utils=feature_utils,
                        net=net,
                        fm=custom_fm,
                        rng=gen,
                        cfg_strength=cfg_strength
                    )
                
                audio_path = Path(output_path) / f"text_to_audio_{unique_id}.flac"
                
                wave_int16 = (waveform_float * 32767.0).clamp(-32768.0, 32767.0).to(torch.int16).cpu()
                if wave_int16.dim() > 2:
                    wave_int16 = wave_int16.reshape(1, -1)
                torchaudio.save(str(audio_path), wave_int16, sample_rate=44100)
                
                logger.info(f"Audio generated and saved to {audio_path} in {time.time() - start_time:.2f}s")
                return audio_path, f"Audio generated successfully and saved to {audio_path}"
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error details: {error_details}")
        logger.error(f"Error generating audio: {str(e)}")
        return None, f"Error: {str(e)}"

def generate_image_to_audio(image_file, prompt, negative_prompt="", seed=0, num_steps=20, cfg_strength=7.5, duration=8):
    """Generate audio for an image (experimental)"""
    start_time = time.time()
    logger.info(f"Starting image-to-audio generation with parameters: seed={seed}, steps={num_steps}, cfg={cfg_strength}, duration={duration}s")
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    rng.manual_seed(seed)
    
    if image_file is None:
        logger.warning("No image file provided")
        return None, "Please upload an image file."
        
    if not prompt:
        logger.info("No prompt provided, using default prompt")
        prompt = "ambient sounds matching this image"
    
    unique_id = str(uuid.uuid4())[:8]
    output_path = os.path.join(OUTPUT_DIR, f"output_{unique_id}")
    os.makedirs(output_path, exist_ok=True)
    logger.info(f"Output directory created: {output_path}")
    
    generation_id = str(uuid.uuid4())[:8]
    logger.info(f"Generation ID: {generation_id}")
    
    try:
        duration = float(duration)
        seed = int(seed)
        num_steps = int(num_steps)
        cfg_strength = float(cfg_strength)
    except ValueError:
        return None, "Invalid numeric parameters. Please check your inputs."
    
    if duration <= 0:
        return None, "Duration must be greater than 0."
        
    # Ensure duration is at least 6 seconds for image inputs to have enough frames
    if duration < 6:
        logger.warning(f"Requested duration ({duration}s) is too short for MMAudio. Setting to minimum 6 seconds.")
        duration = 6.0
    
    try:
        with torch.inference_mode():
            custom_fm = FlowMatching(min_sigma=0.0, inference_mode='euler', num_steps=num_steps)
            gen = torch.Generator(device='cpu').manual_seed(seed)
            
            image_path = Path(image_file)
            audio_path = Path(output_path) / f"{image_path.stem}_audio.flac"
            
            image_info = load_image(image_path)
            
            clip_frames = image_info.clip_frames.to(device=device, dtype=dtype).unsqueeze(0)
            sync_frames = image_info.sync_frames.to(device=device, dtype=dtype).unsqueeze(0)
            
            if device == 'cuda':
                logger.info(f"[{generation_id}] Temporarily moving model to CPU for image-to-audio generation")

                net_cpu = net.to('cpu')
                clip_frames_cpu = clip_frames.to('cpu')
                sync_frames_cpu = sync_frames.to('cpu')
                
                waveform_float = generate(
                    clip_video=clip_frames_cpu,
                    sync_video=sync_frames_cpu,
                    text=[prompt] if prompt else [],
                    negative_text=[negative_prompt] if negative_prompt else None,
                    feature_utils=feature_utils,
                    net=net_cpu,
                    fm=custom_fm,
                    rng=gen,
                    cfg_strength=cfg_strength,
                    image_input=True,
                )
                net_cpu.to(device)
            else:
                waveform_float = generate(
                    clip_video=clip_frames,
                    sync_video=sync_frames,
                    text=[prompt] if prompt else [],
                    negative_text=[negative_prompt] if negative_prompt else None,
                    feature_utils=feature_utils,
                    net=net,
                    fm=custom_fm,
                    rng=gen,
                    cfg_strength=cfg_strength,
                    image_input=True,
                )
            
            wave_int16 = (waveform_float * 32767.0).clamp(-32768.0, 32767.0).to(torch.int16).cpu()
            if wave_int16.dim() > 2:
                wave_int16 = wave_int16.reshape(1, -1)
            torchaudio.save(str(audio_path), wave_int16, sample_rate=44100)
            
            logger.info(f"Audio generated and saved to {audio_path} in {time.time() - start_time:.2f}s")
            return audio_path, f"Audio generated successfully and saved to {audio_path}"
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error details: {error_details}")
        logger.error(f"Error generating audio: {str(e)}")
        return None, f"Error: {str(e)}"

def clear_outputs():
    """Clear the output directory"""
    try:
        if os.path.exists(OUTPUT_DIR):
            for file in os.listdir(OUTPUT_DIR):
                file_path = os.path.join(OUTPUT_DIR, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        return "Output directory cleared."
    except Exception as e:
        return f"Error clearing output directory: {str(e)}"

with gr.Blocks(title="MMAudio: Video-to-Audio & Text-to-Audio", theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate", spacing_size="sm", radius_size="sm", text_size="sm")) as demo:
    gr.Markdown("""
    <div style="text-align: center;">
    <h1>MMAudio Web UI</h1>
    
    <h2>Usage:</h2>
    1. <b>Video-to-Audio</b>: Upload a video and optionally provide a text prompt.<br>
    2. <b>Text-to-Audio</b>: Leave the video field empty and provide a text prompt.<br>
    3. <b>Image-to-Audio</b> (Experimental): Upload an image and provide a text prompt.<br><br>
    <p>Default duration is 8 seconds. For best results, keep it close to this value.</p>
    </div>
    """)

    with gr.Tab("Video-to-Audio & Text-to-Audio"):
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload Video (leave empty for text-to-audio)")
                prompt_input = gr.Textbox(label="Text Prompt", lines=2)
                negative_prompt_input = gr.Textbox(label="Negative Prompt (optional)", lines=2)
                with gr.Row():
                    seed_input = gr.Number(label="Random Seed", value=0)
                    num_steps_input = gr.Slider(label="Num Steps", minimum=10, maximum=50, value=20, step=1)
                    cfg_strength_input = gr.Slider(label="CFG Strength", minimum=1.0, maximum=15.0, value=7.5, step=0.5)
                generate_button = gr.Button("Generate Audio")
            
            with gr.Column():
                output_audio = gr.Audio(label="Output Audio")
                output_message = gr.Textbox(label="Status", lines=2)
                duration_input = gr.Number(label="Duration (seconds)", value=8, minimum=0.1)
                start_time_input = gr.Number(label="Video Start Time (seconds)", value=0, minimum=0)
                clear_button = gr.Button("Clear Output Directory")
        generate_button.click(
            generate_audio,
            inputs=[video_input, prompt_input, negative_prompt_input, seed_input, 
                   num_steps_input, cfg_strength_input, duration_input, start_time_input],
            outputs=[output_audio, output_message]
        )
        
        clear_button.click(
            clear_outputs,
            inputs=[],
            outputs=[output_message]
        )
    
    with gr.Tab("Image-to-Audio (Experimental)"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Upload Image", type="filepath")
                image_prompt_input = gr.Textbox(label="Text Prompt", lines=2)
                image_negative_prompt_input = gr.Textbox(label="Negative Prompt (optional)", lines=2)
                with gr.Row():
                    image_seed_input = gr.Number(label="Random Seed", value=0)
                    image_num_steps_input = gr.Slider(label="Num Steps", minimum=10, maximum=50, value=20, step=1)
                    image_cfg_strength_input = gr.Slider(label="CFG Strength", minimum=1.0, maximum=15.0, value=7.5, step=0.5)
                image_generate_button = gr.Button("Generate Audio")
            with gr.Column():
                image_output_audio = gr.Audio(label="Output Audio")
                image_output_message = gr.Textbox(label="Status", lines=2)
                image_duration_input = gr.Number(label="Duration (seconds)", value=8, minimum=0.1)
        
        image_generate_button.click(
            generate_image_to_audio,
            inputs=[image_input, image_prompt_input, image_negative_prompt_input, image_seed_input, 
                   image_num_steps_input, image_cfg_strength_input, image_duration_input],
            outputs=[image_output_audio, image_output_message]
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)
