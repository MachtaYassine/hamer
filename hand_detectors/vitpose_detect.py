"""ViTPose Wholebody hand detector — batched FP16 path.

Bypasses mmpose, runs backbone + head directly with FP16 autocast.
Full-frame crops assumed (no person bbox needed).
"""

import cv2
import numpy as np
import torch

from .base import HandDetection, HandDetector


class ViTPoseHandDetector(HandDetector):

    def __init__(
        self,
        cpm,
        device: str = "cuda",
        batch_size: int = 16,
        min_keypoint_confidence: float = 0.5,
        min_valid_keypoints: int = 3,
    ):
        self.device = device
        self.batch_size = batch_size
        self.min_conf = min_keypoint_confidence
        self.min_valid = min_valid_keypoints

        model = cpm.model
        model.eval()
        self._backbone = model.backbone
        self._head = model.keypoint_head
        self._img_size = model.cfg.data_cfg['image_size']  # [192, 256]
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def detect_hands(self, img_bgr: np.ndarray) -> list[HandDetection]:
        raise NotImplementedError("Use detect_hands_video for batched processing")

    def detect_hands_video(self, video_path: str):
        """Detect hands in all frames of a video.

        Returns:
            dict[int, list[HandDetection]]: frame_idx -> detections.
        """
        W, H = self._img_size
        BS = self.batch_size

        cap = cv2.VideoCapture(video_path)
        all_crops = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            resized = cv2.resize(frame[:, :, ::-1], (W, H))
            normalized = (resized / 255.0 - self._mean) / self._std
            all_crops.append(torch.from_numpy(normalized.transpose(2, 0, 1)).float())
        cap.release()

        n_frames = len(all_crops)
        if n_frames == 0:
            return {}

        # Batched FP16 forward
        all_max_vals = []
        all_max_idx = []
        Hh, Hw = None, None
        for i in range(0, n_frames, BS):
            batch = torch.stack(all_crops[i:i + BS]).to(self.device)
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
                out = self._backbone(batch)
                if isinstance(out, (list, tuple)):
                    out = out[-1]
                hm = self._head(out).float().cpu()
            _, n_kps, Hh, Hw = hm.shape
            flat = hm.view(hm.shape[0], n_kps, -1)
            vals, idx = flat.max(dim=2)
            all_max_vals.append(vals)
            all_max_idx.append(idx)
            del batch, out, hm

        del all_crops

        max_vals = torch.cat(all_max_vals, dim=0)
        max_idx = torch.cat(all_max_idx, dim=0)

        # Flat index to (x, y) in original image coords
        cap = cv2.VideoCapture(video_path)
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        scale_x = orig_w / W
        scale_y = orig_h / H
        kps_x = (max_idx % Hw).float() * (W / Hw) * scale_x
        kps_y = (max_idx // Hw).float() * (H / Hh) * scale_y
        confidences = max_vals.numpy()

        results = {}
        for fi in range(n_frames):
            kps_all = np.stack([
                kps_x[fi].numpy(),
                kps_y[fi].numpy(),
                confidences[fi],
            ], axis=1)  # (133, 3)

            detections = self._extract_hands(kps_all)
            if detections:
                results[fi] = detections

        return results

    def _extract_hands(self, kps_all):
        detections = []
        for hand_kps, is_right in [(kps_all[-42:-21], False), (kps_all[-21:], True)]:
            valid = hand_kps[:, 2] > self.min_conf
            if int(valid.sum()) <= self.min_valid:
                continue
            kps_valid = hand_kps[valid]
            bbox = np.array([
                kps_valid[:, 0].min(), kps_valid[:, 1].min(),
                kps_valid[:, 0].max(), kps_valid[:, 1].max(),
            ], dtype=np.float32)
            detections.append(HandDetection(
                bbox=bbox, is_right=is_right,
                confidence=float(kps_valid[:, 2].mean()),
                keypoints=hand_kps.astype(np.float32),
            ))
        return detections

    def close(self):
        self._backbone = None
        self._head = None
