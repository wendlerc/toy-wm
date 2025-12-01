import torch as t 
import numpy as np
import cv2

from ..inference import sample_video
from ..datasets.pong1m import fixed2frame


def annotate_frames(frames, annotations):
    # Assume shape: (batch, T, C, H, W)
    # frames should be uint8 [0, 255] (from fixed2frame)
    # Returns uint8 [0, 255] tensor to match pred2frame output
    frames_np = frames.cpu().numpy()  # Convert to numpy for cv2
    
    b, T, C, H, W = frames_np.shape
    extended_h = H + 8
    extended_w = W 
    extended_frames = np.ones((b, T, C, extended_h, extended_w), dtype=np.uint8) * 255  # White background

    for batch_idx in range(b):
        for t_idx in range(T):
            # Place old frame below the 7px top margin
            extended_frames[batch_idx, t_idx, :, 8:, :] = frames_np[batch_idx, t_idx]
            # Write action
            action_val = annotations[batch_idx, t_idx].item()
            # Convert CHW to HWC for cv2
            frame_for_label = np.moveaxis(extended_frames[batch_idx, t_idx], 0, 2).copy()
            # Choose color: black text
            color = (0, 0, 0)
            font_scale = 0.22
            thickness = 1
            org = (3, 6)
            txt = f'{action_val}'
            cv2.putText(
                frame_for_label, txt, org,
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, lineType=cv2.LINE_AA
            )
            # Move back to CHW
            extended_frames[batch_idx, t_idx] = np.moveaxis(frame_for_label, 2, 0)

    # Convert back to torch uint8 tensor [0, 255] to match pred2frame output
    extended_frames_torch = t.from_numpy(extended_frames)
    return extended_frames_torch

def basic_control(model, n_steps=6):
    actions = t.tensor(30*[1] + 60*[2] + 60*[3] + 30*[0], dtype=t.int32, device=model.device).unsqueeze(0)
    pred = sample_video(model, actions, n_steps=n_steps)
    frames = fixed2frame(pred)  
    return annotate_frames(frames, actions)
    