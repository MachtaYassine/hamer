from pathlib import Path
import torch
import argparse
import os
import io
import cv2
import numpy as np

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


class _ListDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        return self.items[idx]


def auto_find_batch_size(model, sample_item, device, target_util=0.85):
    """Two-probe GPU memory estimation, via the one shared heuristic."""
    import gc
    from hmr4d.utils.auto_batch import auto_batch_size

    def probe(n):
        loader = torch.utils.data.DataLoader(_ListDataset([sample_item] * n), batch_size=n)
        return model(recursive_to(next(iter(loader)), device))

    gc.collect()
    return auto_batch_size(
        probe, label="HaMeR", cap=256, device=device,
        target_util=target_util, small_gpu_target_util=0.65,
    )


def main():
    parser = argparse.ArgumentParser(description='HaMeR hand mesh recovery')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--img_folder', type=str, default='images')
    parser.add_argument('--out_folder', type=str, default='out_demo')
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--rescale_factor', type=float, default=2.0)
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'])
    parser.add_argument('--auto_batch_size', action='store_true', default=True)
    parser.add_argument('--no_auto_batch_size', action='store_false', dest='auto_batch_size')
    parser.add_argument('--focal_length', type=float, default=0,
                        help='Override focal length for cam_t_full (0=use default HaMeR focal)')
    parser.add_argument('--frame_stride', type=int, default=1,
                        help='Source-frame stride used when extracting img_folder. Frames are '
                             'renumbered 1..N by ffmpeg but correspond to source frames '
                             '0,stride,2*stride,... — without this, hands land on the wrong frames.')
    parser.add_argument('--video', type=str, default='',
                        help='Read frames from video directly (skip frame extraction to disk)')

    args = parser.parse_args()

    # Buffered torch.load: one bulk read then deserialize from RAM.
    _torch_load_orig = torch.load
    def _torch_load_buffered(f, *a, **kw):
        kw.setdefault("weights_only", False)
        if isinstance(f, (str, Path)):
            with open(f, "rb") as fh:
                return _torch_load_orig(io.BytesIO(fh.read()), *a, **kw)
        return _torch_load_orig(f, *a, **kw)
    torch.load = _torch_load_buffered

    download_models(CACHE_DIR_HAMER)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # On small GPUs (<12GB), defer HaMeR loading until after ViTPose finishes
    small_gpu = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 12e9
    model = None
    model_cfg = None
    if not small_gpu:
        model, model_cfg = load_hamer(args.checkpoint)
        model = model.to(device)
        model.eval()
    else:
        # Only load config (not the model) to save GPU memory for ViTPose
        from pathlib import Path as _P
        from hamer.configs import get_config
        _cfg_path = str(_P(args.checkpoint).parent.parent / 'model_config.yaml')
        model_cfg = get_config(_cfg_path, update_cachedir=True)
        if model_cfg.MODEL.BACKBONE.TYPE == 'vit' and 'BBOX_SHAPE' not in model_cfg.MODEL:
            model_cfg.defrost()
            model_cfg.MODEL.BBOX_SHAPE = [192, 256]
            model_cfg.freeze()

    # ViTPose batched FP16 hand detector
    from vitpose_model import ViTPoseModel
    from hand_detectors import create_hand_detector
    cpm = ViTPoseModel(str(device))
    hand_detector = create_hand_detector(cpm=cpm, device=str(device))

    os.makedirs(args.out_folder, exist_ok=True)

    # Get frames
    video_cap = None
    if args.video:
        video_cap = cv2.VideoCapture(args.video)
        n_total = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        img_paths = [Path(f"{i:06d}.jpg") for i in range(1, n_total + 1)]
        print(f"  [Video] {n_total} frames from {args.video}")
    else:
        img_paths = sorted([img for end in args.file_type for img in Path(args.img_folder).glob(end)])
        n_total = len(img_paths)

    # ── Phase 1: Batched FP16 ViTPose hand detection ─────────────────────
    import time as _time
    print(f"Phase 1: ViTPose FP16 batched on {n_total} frames...")

    if args.video:
        t0 = _time.time()
        video_dets = hand_detector.detect_hands_video(args.video, stride=args.frame_stride)
        t_detect = _time.time() - t0
        n_seen = (n_total + args.frame_stride - 1) // args.frame_stride
        print(f"  {n_seen}/{n_total} frames (stride={args.frame_stride}) in {t_detect:.1f}s "
              f"({n_seen/t_detect:.0f} fps)")
    else:
        video_dets = {}
        for frame_idx in tqdm(range(n_total), desc="Detecting"):
            img_cv2 = cv2.imread(str(img_paths[frame_idx]))
            dets = hand_detector.detect_hands(img_cv2)
            if dets:
                video_dets[frame_idx] = dets

    hand_detector.close()
    del cpm
    torch.cuda.empty_cache()

    # Create ViTDetDataset crops from detections
    detections = []
    all_items = []
    item_to_det = []

    if video_cap is not None:
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for frame_idx in tqdm(range(n_total), desc="Cropping"):
        hand_dets = video_dets.get(frame_idx, [])
        if video_cap is not None:
            # grab() advances the decoder cheaply; only retrieve() (decode->BGR) the
            # frames that actually have detections. With stride>1 most are skipped.
            if not video_cap.grab():
                break
            if not hand_dets:
                continue
            ret, img_cv2 = video_cap.retrieve()
            if not ret:
                break
        else:
            if not hand_dets:
                continue
            img_cv2 = cv2.imread(str(img_paths[frame_idx]))

        boxes = np.stack([d.bbox for d in hand_dets])
        right = np.array([int(d.is_right) for d in hand_dets])
        kp_counts = [int((d.keypoints[:, 2] > 0.5).sum()) if d.keypoints is not None else 0 for d in hand_dets]
        kp_mean_confs = [float(d.confidence) for d in hand_dets]

        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        det_idx = len(detections)
        detections.append({
            'img_path': img_paths[frame_idx], 'n_crops': len(dataset),
            'kp_counts': kp_counts, 'kp_mean_confs': kp_mean_confs,
        })
        for i in range(len(dataset)):
            all_items.append(dataset[i])
            item_to_det.append(det_idx)

    if video_cap is not None:
        video_cap.release()

    n_hands = len(all_items)
    print(f"  Found {n_hands} hands in {len(detections)}/{n_total} frames")

    if n_hands == 0:
        print("No hands detected!")
        return

    # ── Phase 2: Batched HaMeR inference ────────────────────────────────
    if model is None:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        model, _ = load_hamer(args.checkpoint)
        model = model.to(device).eval()
        print(f"  [Small GPU] Loaded HaMeR after freeing ViTPose")

    if args.auto_batch_size and device.type == 'cuda':
        batch_size = auto_find_batch_size(model, all_items[0], device)
    else:
        batch_size = args.batch_size

    print(f"Phase 2: HaMeR inference ({n_hands} crops, batch_size={batch_size})...")

    loader = torch.utils.data.DataLoader(
        _ListDataset(all_items), batch_size=batch_size, shuffle=False, num_workers=0
    )

    all_hand_params = {
        "frame_idx": [], "vertices": [], "cam_t_full": [],
        "hand_pose": [], "global_orient": [], "betas": [],
        "is_right": [], "scaled_focal_length": [],
        "kp_count": [], "kp_mean_conf": [],
    }

    for batch in tqdm(loader, desc="HaMeR"):
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)

        bs = batch['img'].shape[0]
        multiplier = (2*batch['right']-1)
        pred_cam = out['pred_cam']
        pred_cam[:,1] = multiplier*pred_cam[:,1]
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()

        if args.focal_length > 0:
            per_sample_fl = torch.full((bs,), args.focal_length, device=pred_cam.device)
        else:
            per_sample_fl = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max(dim=1)[0]
        from hamer.utils.renderer import cam_crop_to_full
        pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, per_sample_fl).detach().cpu().numpy()

        mano_params = out['pred_mano_params']

        for n in range(bs):
            det_idx = item_to_det[len(all_hand_params["frame_idx"])]
            det = detections[det_idx]
            person_id = int(batch['personid'][n])
            img_fn = os.path.splitext(os.path.basename(det['img_path']))[0]

            kp_count = det['kp_counts'][person_id] if person_id < len(det['kp_counts']) else 0
            kp_conf = det['kp_mean_confs'][person_id] if person_id < len(det['kp_mean_confs']) else 0.0

            all_hand_params["frame_idx"].append(int(img_fn) - 1)
            all_hand_params["vertices"].append(out['pred_vertices'][n].detach().cpu().numpy())
            all_hand_params["cam_t_full"].append(pred_cam_t_full[n])
            all_hand_params["hand_pose"].append(mano_params['hand_pose'][n].detach().cpu().numpy())
            all_hand_params["global_orient"].append(mano_params['global_orient'][n].detach().cpu().numpy())
            all_hand_params["betas"].append(mano_params['betas'][n].detach().cpu().numpy())
            all_hand_params["is_right"].append(batch['right'][n].cpu().numpy())
            all_hand_params["scaled_focal_length"].append(float(per_sample_fl[n]))
            all_hand_params["kp_count"].append(kp_count)
            all_hand_params["kp_mean_conf"].append(kp_conf)

    # Save consolidated output
    if len(all_hand_params["frame_idx"]) > 0:
        consolidated = {
            "frame_idx": torch.tensor(all_hand_params["frame_idx"], dtype=torch.long),
            "vertices": torch.tensor(np.stack(all_hand_params["vertices"]), dtype=torch.float32),
            "cam_t_full": torch.tensor(np.stack(all_hand_params["cam_t_full"]), dtype=torch.float32),
            "hand_pose": torch.tensor(np.stack(all_hand_params["hand_pose"]), dtype=torch.float32),
            "global_orient": torch.tensor(np.stack(all_hand_params["global_orient"]), dtype=torch.float32),
            "betas": torch.tensor(np.stack(all_hand_params["betas"]), dtype=torch.float32),
            "is_right": torch.tensor(np.array(all_hand_params["is_right"], dtype=bool)),
            "scaled_focal_length": torch.tensor(np.array(all_hand_params["scaled_focal_length"], dtype=np.float32)),
            "kp_count": torch.tensor(np.array(all_hand_params["kp_count"], dtype=np.int64)),
            "kp_mean_conf": torch.tensor(np.array(all_hand_params["kp_mean_conf"], dtype=np.float32)),
        }
        pt_path = os.path.join(args.out_folder, "hamer_hands.pt")
        torch.save(consolidated, pt_path)
        print(f"  Saved {len(all_hand_params['frame_idx'])} hand detections -> {pt_path}")
    else:
        print("No hands detected!")

if __name__ == '__main__':
    main()
