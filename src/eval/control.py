import torch as t
import numpy as np
import cv2

from ..inference import sample_video


def annotate_frames(frames, annotations, action_emb=None):
    # Assume shape: (batch, T, C, H, W)
    # frames should be uint8 [0, 255] (from pred2frame)
    # Returns uint8 [0, 255] tensor to match pred2frame output
    # action_emb: optional, if it has format_action() use it for labels
    frames_np = frames.cpu().numpy()  # Convert to numpy for cv2

    b, T, C, H, W = frames_np.shape
    T_ann = min(T, annotations.shape[1])  # align with annotations length
    base_font_scale = 0.22
    min_font_scale = 0.08
    thickness = 1
    top_margin = 10
    extended_h = H + top_margin
    extended_w = W
    extended_frames = np.ones((b, T, C, extended_h, extended_w), dtype=np.uint8) * 255  # White background

    format_fn = getattr(action_emb, "format_action", None) if action_emb is not None else None

    for batch_idx in range(b):
        for t_idx in range(T_ann):
            # Place old frame below the top label band.
            extended_frames[batch_idx, t_idx, :, top_margin:, :] = frames_np[batch_idx, t_idx]
            # Write action label
            ann = annotations[batch_idx, t_idx]
            if format_fn is not None:
                txt = format_fn(ann)
            elif ann.numel() == 1:
                txt = str(ann.item())
            else:
                txt = str(ann.cpu().tolist())[:20]  # truncate long vectors
            # Convert CHW to HWC for cv2
            frame_for_label = np.moveaxis(extended_frames[batch_idx, t_idx], 0, 2).copy()
            color = (0, 0, 0)
            font_scale = base_font_scale
            max_text_width = W - 10
            (text_width, text_height), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            if text_width > max_text_width:
                font_scale = max(min_font_scale, base_font_scale * max_text_width / text_width)
            org = (3, max(7, top_margin - 3))
            cv2.putText(
                frame_for_label, txt, org,
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, lineType=cv2.LINE_AA
            )
            # Move back to CHW
            extended_frames[batch_idx, t_idx] = np.moveaxis(frame_for_label, 2, 0)

    # Convert back to torch uint8 tensor [0, 255] to match pred2frame output
    extended_frames_torch = t.from_numpy(extended_frames)
    return extended_frames_torch

def basic_control(model, pred2frame, n_steps=6):
    actions = model.action_emb.basic_control_actions
    pred = sample_video(model, actions, n_steps=n_steps)
    pred = pred.to(model.device)
    frames = pred2frame(pred)
    return annotate_frames(frames, actions, action_emb=model.action_emb)
