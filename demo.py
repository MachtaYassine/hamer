from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
from collections import defaultdict

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full
import detectron2.data.transforms as T

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

from vitpose_model import ViTPoseModel

import json
from typing import Dict, Optional

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


class _ListDataset(torch.utils.data.Dataset):
    """Wrap a list of dicts as a Dataset for DataLoader batching."""
    def __init__(self, items):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        return self.items[idx]


def auto_find_batch_size(model, sample_item, device, target_util=0.85):
    """Probe GPU memory with a small test batch to find optimal batch size."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    baseline = torch.cuda.memory_allocated(device)

    # Test with a small batch to measure per-sample cost
    test_bs = 2
    loader = torch.utils.data.DataLoader(
        _ListDataset([sample_item] * test_bs), batch_size=test_bs
    )
    batch = next(iter(loader))
    batch = recursive_to(batch, device)
    with torch.no_grad():
        _ = model(batch)

    peak = torch.cuda.max_memory_allocated(device)
    per_sample = (peak - baseline) / test_bs

    del batch, _
    torch.cuda.empty_cache()

    total = torch.cuda.get_device_properties(device).total_memory
    available = total * target_util - baseline
    optimal = int(available / max(per_sample, 1))
    optimal = max(1, min(optimal, 512))

    print(f"  [Auto BS] GPU: {total/1e9:.1f}GB total, model: {baseline/1e9:.1f}GB, "
          f"per_sample: {per_sample/1e6:.0f}MB -> batch_size={optimal}")
    return optimal


def _preprocess_for_detector(detector, img_cv2):
    """Preprocess a single image for the detectron2 detector."""
    original_image = img_cv2
    if detector.input_format == "RGB":
        original_image = original_image[:, :, ::-1]
    height, width = original_image.shape[:2]
    image = detector.aug(T.AugInput(original_image)).apply_image(original_image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    return {"image": image, "height": height, "width": width}


def _auto_det_batch_size(detector, sample_img_cv2, fp16=True, target_util=0.85):
    """Probe GPU to find optimal batch size for body detector."""
    device = next(detector.model.parameters()).device
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)

    test_bs = 2
    inputs = [_preprocess_for_detector(detector, sample_img_cv2) for _ in range(test_bs)]
    with torch.no_grad():
        if fp16 and torch.cuda.is_available():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                _ = detector.model(inputs)
        else:
            _ = detector.model(inputs)
    peak = torch.cuda.max_memory_allocated(device)
    per_sample = (peak - baseline) / test_bs
    del inputs, _
    torch.cuda.empty_cache()

    total = torch.cuda.get_device_properties(device).total_memory
    available = total * target_util - baseline
    optimal = max(1, min(128, int(available / max(per_sample, 1))))
    print(f"  [Auto BS] ViTDet: {total/1e9:.1f}GB GPU, {per_sample/1e6:.0f}MB/sample -> det_batch_size={optimal}")
    return optimal


def _batch_body_detect(detector, images_cv2, det_batch_size=4, fp16=True):
    """Run body detection on multiple images in batches.

    Returns list of prediction dicts, one per image.
    """
    all_preds = []
    for i in range(0, len(images_cv2), det_batch_size):
        batch_imgs = images_cv2[i:i+det_batch_size]
        inputs = [_preprocess_for_detector(detector, img) for img in batch_imgs]

        with torch.no_grad():
            if fp16 and torch.cuda.is_available():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    preds = detector.model(inputs)
            else:
                preds = detector.model(inputs)
        all_preds.extend(preds)
    return all_preds


def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=48, help='Batch size for HaMeR inference')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--body_detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Using regnety improves runtime and reduces memory')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')
    parser.add_argument('--save_params', dest='save_params', action='store_true', default=False, help='If set, save MANO params per frame as NPZ')
    parser.add_argument('--no_render', action='store_true', default=False, help='Skip rendering (much faster, only save params/meshes)')
    parser.add_argument('--auto_batch_size', action='store_true', default=True, help='Auto-detect optimal batch size from GPU VRAM')
    parser.add_argument('--no_auto_batch_size', action='store_false', dest='auto_batch_size', help='Disable auto batch size')
    parser.add_argument('--det_batch_size', type=int, default=8, help='Batch size for body detector (ViTDet/RegNetY)')
    parser.add_argument('--fp16', action='store_true', default=True, help='Use FP16 for detection (faster)')
    parser.add_argument('--no_fp16', action='store_false', dest='fp16', help='Disable FP16')
    parser.add_argument('--gvhmr_bboxes', type=str, default='', help='Path to GVHMR bbx.pt file — skip ViTDet, use GVHMR person bboxes')
    parser.add_argument('--focal_length', type=float, default=0,
                        help='Override focal length for cam_t_full (0=use default HaMeR focal)')

    args = parser.parse_args()

    # Download and load checkpoints
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load detector (skip if using GVHMR bboxes)
    detector = None
    if not args.gvhmr_bboxes:
        from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
        if args.body_detector == 'vitdet':
            from detectron2.config import LazyConfig
            import hamer
            cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
            detectron2_cfg = LazyConfig.load(str(cfg_path))
            detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
            for i in range(3):
                detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
            detector = DefaultPredictor_Lazy(detectron2_cfg)
        elif args.body_detector == 'regnety':
            from detectron2 import model_zoo
            from detectron2.config import get_cfg
            detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
            detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
            detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
            detector       = DefaultPredictor_Lazy(detectron2_cfg)
    else:
        print(f"  [Skip] Body detector not loaded (using GVHMR bboxes)")

    # keypoint detector
    cpm = ViTPoseModel(device)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    # Get all demo images (sorted for deterministic video-frame order)
    img_paths = sorted([img for end in args.file_type for img in Path(args.img_folder).glob(end)])

    # ── Phase 1: Detection (batched body detector + per-image keypoints) ─
    # Load GVHMR bboxes if provided (skip ViTDet)
    gvhmr_bboxes = None
    if args.gvhmr_bboxes:
        import torch as _torch
        bbx_data = _torch.load(args.gvhmr_bboxes, map_location='cpu', weights_only=False)
        gvhmr_bboxes = bbx_data['bbx_xyxy'].numpy()  # (L, 4)
        print(f"Phase 1: Using GVHMR bboxes ({len(gvhmr_bboxes)} frames) — skipping ViTDet")
    else:
        print(f"Phase 1: Detecting hands in {len(img_paths)} images "
              f"(det_batch={args.det_batch_size}, fp16={args.fp16})...")

    detections = []   # per-image: {img_path, n_crops}
    all_items = []    # flat list of preprocessed crop dicts
    item_to_det = []  # maps each crop index -> detection index

    n_total = len(img_paths)
    det_bs = args.det_batch_size

    # Auto-detect optimal detection batch size
    if gvhmr_bboxes is None and detector is not None and torch.cuda.is_available() and n_total > 0:
        probe_img = cv2.imread(str(img_paths[0]))
        det_bs = _auto_det_batch_size(detector, probe_img, fp16=args.fp16)
        del probe_img

    for chunk_start in tqdm(range(0, n_total, det_bs), desc="Detecting",
                            total=(n_total + det_bs - 1) // det_bs):
        chunk_paths = img_paths[chunk_start:chunk_start+det_bs]
        chunk_images = [cv2.imread(str(p)) for p in chunk_paths]

        if gvhmr_bboxes is not None:
            # Use GVHMR bboxes directly — one person bbox per frame
            chunk_det_bboxes = []
            for i, p in enumerate(chunk_paths):
                frame_idx = chunk_start + i
                if frame_idx < len(gvhmr_bboxes):
                    bbox = gvhmr_bboxes[frame_idx]
                    chunk_det_bboxes.append(np.array([[bbox[0], bbox[1], bbox[2], bbox[3], 1.0]]))
                else:
                    chunk_det_bboxes.append(np.zeros((0, 5)))
        else:
            # Batched body detection
            chunk_det_outs = _batch_body_detect(
                detector, chunk_images,
                det_batch_size=det_bs, fp16=args.fp16,
            )
            chunk_det_bboxes = []
            for det_out in chunk_det_outs:
                det_instances = det_out['instances']
                valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
                pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
                pred_scores=det_instances.scores[valid_idx].cpu().numpy()
                chunk_det_bboxes.append(np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1))

        # Per-image: keypoint detection + hand bbox extraction + crop creation
        for img_cv2, img_path, det_bboxes_scores in zip(chunk_images, chunk_paths, chunk_det_bboxes):
            img = img_cv2.copy()[:, :, ::-1]

            if len(det_bboxes_scores) == 0:
                continue

            # Detect human keypoints for each person
            vitposes_out = cpm.predict_pose(
                img,
                [det_bboxes_scores],
            )

            bboxes = []
            is_right = []

            # Use hands based on hand keypoint detections
            for vitposes in vitposes_out:
                left_hand_keyp = vitposes['keypoints'][-42:-21]
                right_hand_keyp = vitposes['keypoints'][-21:]

                # Rejecting not confident detections
                keyp = left_hand_keyp
                valid = keyp[:,2] > 0.5
                if sum(valid) > 3:
                    bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                    bboxes.append(bbox)
                    is_right.append(0)
                keyp = right_hand_keyp
                valid = keyp[:,2] > 0.5
                if sum(valid) > 3:
                    bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                    bboxes.append(bbox)
                    is_right.append(1)

            if len(bboxes) == 0:
                continue

            boxes = np.stack(bboxes)
            right = np.stack(is_right)

            # Create crop dataset and extract all items
            dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
            det_idx = len(detections)
            detections.append({'img_path': img_path, 'n_crops': len(dataset)})

            for i in range(len(dataset)):
                all_items.append(dataset[i])
                item_to_det.append(det_idx)

    n_hands = len(all_items)
    n_images_with_hands = len(detections)
    print(f"  Found {n_hands} hands in {n_images_with_hands}/{len(img_paths)} images")

    if n_hands == 0:
        print("No hands detected!")
        return

    # ── Phase 2: Batched HaMeR inference ────────────────────────────────
    if args.auto_batch_size and device.type == 'cuda':
        batch_size = auto_find_batch_size(model, all_items[0], device)
    else:
        batch_size = args.batch_size

    print(f"Phase 2: HaMeR inference ({n_hands} crops, batch_size={batch_size})...")

    loader = torch.utils.data.DataLoader(
        _ListDataset(all_items), batch_size=batch_size, shuffle=False, num_workers=0
    )

    all_results = []  # per-crop results

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

        # Per-sample focal length (correct for mixed-resolution batches)
        if args.focal_length > 0:
            per_sample_fl = torch.full((bs,), args.focal_length, device=pred_cam.device)
        else:
            per_sample_fl = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max(dim=1)[0]
        pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, per_sample_fl).detach().cpu().numpy()

        for n in range(bs):
            verts = out['pred_vertices'][n].detach().cpu().numpy()
            is_right_val = batch['right'][n].cpu().numpy()

            result = {
                'pred_vertices': verts,
                'pred_cam_t': out['pred_cam_t'][n].detach().cpu().numpy(),
                'cam_t_full': pred_cam_t_full[n],
                'is_right': is_right_val,
                'personid': int(batch['personid'][n]),
                'img_size': img_size[n].cpu().numpy(),
                'scaled_focal_length': float(per_sample_fl[n]),
            }

            # Only keep crop tensor if we need it for rendering
            if not args.no_render:
                result['img_crop'] = batch['img'][n].cpu()

            if args.save_params:
                mano_params = out['pred_mano_params']
                result['hand_pose'] = mano_params['hand_pose'][n].detach().cpu().numpy()
                result['global_orient'] = mano_params['global_orient'][n].detach().cpu().numpy()
                result['betas'] = mano_params['betas'][n].detach().cpu().numpy()
                result['pred_keypoints_3d'] = out['pred_keypoints_3d'][n].detach().cpu().numpy()

            all_results.append(result)

    # ── Phase 3: Save params / meshes / render ──────────────────────────
    print(f"Phase 3: Saving results...")

    # Group results by detection (image) index
    det_results = defaultdict(list)
    for res_idx, det_idx in enumerate(item_to_det):
        det_results[det_idx].append(all_results[res_idx])

    for det_idx, det in enumerate(tqdm(detections, desc="Saving")):
        img_path = det['img_path']
        img_fn = os.path.splitext(os.path.basename(img_path))[0]
        results = det_results[det_idx]

        all_verts = []
        all_cam_t = []
        all_right_list = []

        for r in results:
            person_id = r['personid']
            verts = r['pred_vertices']
            is_right_val = r['is_right']
            cam_t = r['cam_t_full']

            # Per-crop rendering
            if not args.no_render:
                white_img = (torch.ones_like(r['img_crop']) - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
                input_patch = r['img_crop'] * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                input_patch = input_patch.permute(1,2,0).numpy()

                regression_img = renderer(verts,
                                        r['pred_cam_t'],
                                        r['img_crop'],
                                        mesh_base_color=LIGHT_BLUE,
                                        scene_bg_color=(1, 1, 1),
                                        )

                if args.side_view:
                    side_img = renderer(verts,
                                            r['pred_cam_t'],
                                            white_img,
                                            mesh_base_color=LIGHT_BLUE,
                                            scene_bg_color=(1, 1, 1),
                                            side_view=True)
                    final_img = np.concatenate([input_patch, regression_img, side_img], axis=1)
                else:
                    final_img = np.concatenate([input_patch, regression_img], axis=1)

                cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_{person_id}.png'), 255*final_img[:, :, ::-1])

            # Collect for full-frame rendering
            verts_ff = verts.copy()
            verts_ff[:,0] = (2*is_right_val-1)*verts_ff[:,0]
            all_verts.append(verts_ff)
            all_cam_t.append(cam_t)
            all_right_list.append(is_right_val)

            # Save meshes to disk
            if args.save_mesh:
                camera_translation = cam_t.copy()
                tmesh = renderer.vertices_to_trimesh(verts_ff, camera_translation, LIGHT_BLUE, is_right=is_right_val)
                tmesh.export(os.path.join(args.out_folder, f'{img_fn}_{person_id}.obj'))

            # Save MANO params
            if args.save_params:
                params_dir = os.path.join(args.out_folder, 'mano_params')
                os.makedirs(params_dir, exist_ok=True)
                np.savez(
                    os.path.join(params_dir, f'{img_fn}_{person_id}.npz'),
                    hand_pose=r['hand_pose'],
                    global_orient=r['global_orient'],
                    betas=r['betas'],
                    vertices=verts,
                    keypoints_3d=r['pred_keypoints_3d'],
                    cam_t_full=cam_t,
                    is_right=is_right_val,
                    img_fn=img_fn,
                    scaled_focal_length=r['scaled_focal_length'],
                )

        # Render front view (full frame with all hands overlaid)
        if not args.no_render and args.full_frame and len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=results[0]['scaled_focal_length'],
            )
            img_size_tensor = torch.tensor(results[0]['img_size'])
            cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size_tensor, is_right=all_right_list, **misc_args)

            # Overlay image
            img_cv2 = cv2.imread(str(img_path))
            input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

            cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_all.jpg'), 255*input_img_overlay[:, :, ::-1])

    print("Done!")

if __name__ == '__main__':
    main()
