import torch
import json
import cv2
import argparse
import numpy as np
import os
import os.path as osp
import time

import sys
sys.path.append("../..")

from PIL import Image
from tqdm import tqdm
from scipy.optimize import least_squares


def estimate_pose_essential(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    # From https://github.com/zju3dv/LoFTR/blob/master/src/utils/metrics.py#L72
    if len(kpts0) < 5:
        return None

    # normalize keypoints
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # normalize ransac threshold
    ransac_thr = thresh / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])

    # compute pose with cv2
    E, mask = cv2.findEssentialMat(kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.USAC_MAGSAC)
    
    if E is None:
        return None

    # recover pose from E
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n

    return ret


def estimate_pose_fundamental(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 7:
        return None

    # Find fundamental matrix using RANSAC
    try:
        F, mask = cv2.findFundamentalMat(kpts0, kpts1, ransacReprojThreshold=thresh, confidence=conf, method=cv2.USAC_MAGSAC)
    except:
        return None

    if F is None:
        return None
    
    # Convert fundamental matrix to essential matrix using intrinsics
    E = K1.T @ F @ K0
    
    # Recover pose from essential matrix
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n

    return ret


def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0):
    # Angle error between 2 vectors
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err_angle = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err_angle = np.minimum(t_err_angle, 180 - t_err_angle)  # handle E ambiguity
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err_angle = 0

    # Metric translation error
    t_err_metric = np.linalg.norm(t - t_gt)

    # Angle error between 2 rotation matrices
    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # handle numerical errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err_angle, t_err_metric, R_err


def backproject_to_3D(uv_points, depth, K):
    # Convert 2D points to homogeneous for and clip points outside image
    uv_homogeneous = np.hstack((uv_points, np.ones((uv_points.shape[0], 1)))).round().astype(int)
    uv_homogeneous = np.clip(uv_homogeneous, a_min=(0, 0, 1), a_max=(1599, 899, 1))

    selected_depths = depth[uv_homogeneous[:, 1], uv_homogeneous[:, 0]]

    # Scale by depth
    points_3D = selected_depths[:, None] * (np.linalg.inv(K) @ (uv_homogeneous.T)).T
    return points_3D


def scale_cost_function(scale, R, t, pts3D_1, pts3D_2, delta=1.0):
    # Apply transformation
    pts3D_1 = (R @ pts3D_1.T).T + scale * t
    res = np.linalg.norm(pts3D_1 - pts3D_2, axis=1)

    # Huber loss with 1m threshold
    cost = np.where(res <= delta, 0.5 * res ** 2, delta * (res - 0.5 * delta))
    return cost


def get_scale(func, scale_ini, R, t, pts3D_1, pts3D_2):
    res = least_squares(func, scale_ini, args=(R, t, pts3D_1, pts3D_2), bounds=(0, np.inf))
    scale = res.x[0]
    return scale


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="xfeat", help="""Method to evaluate in (xfeat, 
                                                                                              xfeat_star, 
                                                                                              xfeat_lighterglue, 
                                                                                              loftr,
                                                                                              sp+lightglue,
                                                                                              disk+lightglue,
                                                                                              sift+lightglue,
                                                                                              aliked+lightglue,
                                                                                              roma,
                                                                                              dedode,
                                                                                              eloftr,
                                                                                              aspanformer,
                                                                                              mast3r,
                                                                                              dust3r,
                                                                                              orb,
                                                                                              rootsift)""")
    parser.add_argument("--estimate_pose", type=str, default="essential", help="""Method to estimate pose in (essential, fundamental)""")
    args = parser.parse_args()

    method = args.method
    estimate_pose = args.estimate_pose
    print(method)
    os.makedirs('results', exist_ok=True)

    # Init model, data and nuscenes
    if "xfeat" in method:
        sys.path.append("../../detector_based/accelerated_features")
        from modules.xfeat import XFeat

        xfeat = XFeat()

    elif method == "loftr":
        sys.path.append("../../detector_free/LoFTR")
        from src.loftr import LoFTR, default_cfg

        loftr = LoFTR(config=default_cfg)
        loftr.load_state_dict(torch.load("../../detector_free/LoFTR/weights/outdoor_ds.ckpt")['state_dict'])
        loftr = loftr.eval().cuda()

    elif method == "sp+lightglue":
        sys.path.append("../../detector_based/LightGlue")
        from lightglue import LightGlue, SuperPoint

        extractor = SuperPoint(max_num_keypoints=None).eval().cuda()  # load the extractor
        matcher = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().cuda()  # load the matcher

    elif method == "disk+lightglue":
        sys.path.append("../../detector_based/LightGlue")
        from lightglue import LightGlue, DISK

        extractor = DISK(max_num_keypoints=None).eval().cuda()  # load the extractor
        matcher = LightGlue(features='disk', depth_confidence=-1, width_confidence=-1).eval().cuda()  # load the matcher

    elif method == "sift+lightglue":
        sys.path.append("../../detector_based/LightGlue")
        from lightglue import LightGlue, SIFT

        extractor = SIFT().eval().cuda()  # load the extractor
        matcher = LightGlue(features='sift', depth_confidence=-1, width_confidence=-1).eval().cuda()  # load the matcher

    elif method == "aliked+lightglue":
        sys.path.append("../../detector_based/LightGlue")
        from lightglue import LightGlue, ALIKED

        extractor = ALIKED().eval().cuda()  # load the extractor
        matcher = LightGlue(features='aliked', depth_confidence=-1, width_confidence=-1).eval().cuda()  # load the matcher

    elif method == "roma":
        sys.path.append("../../detector_free/RoMa")
        from romatch import roma_outdoor

        roma_model = roma_outdoor(device="cuda")

    elif method == "dedode":
        sys.path.append("../../detector_based/DeDoDe")
        from DeDoDe import dedode_detector_L, dedode_descriptor_G
        from DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher

        # You can either provide weights manually, or not provide any. If none
        # are provided we automatically download them. None: We now use v2 detector weights by default.
        detector = dedode_detector_L(weights = None)
        descriptor = dedode_descriptor_G(weights = None, 
                                         dinov2_weights = None) # You can manually load dinov2 weights, or we'll pull from facebook

        matcher = DualSoftMaxMatcher()
    
    elif method == "eloftr":
        sys.path.append("../../detector_free/EfficientLoFTR")
        from copy import deepcopy
        from src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter

        _default_cfg = deepcopy(full_default_cfg)
        print(_default_cfg)
        matcher = LoFTR(config=_default_cfg)

        matcher.load_state_dict(torch.load("../../detector_free/EfficientLoFTR/weights/eloftr_outdoor.ckpt")['state_dict'])
        matcher = reparameter(matcher) # no reparameterization will lead to low performance
        matcher = matcher.eval().cuda()

    elif method == "aspanformer":
        sys.path.append("../../detector_free/ml-aspanformer")
        from src.ASpanFormer.aspanformer import ASpanFormer 
        from src.config.default import get_cfg_defaults
        from src.utils.misc import lower_config
        import demo.demo_utils as demo_utils

        config = get_cfg_defaults()
        config.merge_from_file("../../detector_free/ml-aspanformer/configs/aspan/outdoor/aspan_test.py")
        _config = lower_config(config)

        matcher = ASpanFormer(config=_config['aspan'])
        state_dict = torch.load("../../detector_free/ml-aspanformer/weights/outdoor.ckpt", map_location='cpu')['state_dict']
        matcher.load_state_dict(state_dict, strict=False)
        matcher.cuda()
        matcher.eval()

    elif method == "mast3r":
        sys.path.append("../../detector_free/mast3r")
        import mast3r.utils.path_to_dust3r
        from dust3r.inference import inference
        from dust3r.utils.image import load_images
        from mast3r.fast_nn import fast_reciprocal_NNs
        from mast3r.model import AsymmetricMASt3R

        model = AsymmetricMASt3R.from_pretrained("../../detector_free/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth").cuda()
        model = model.eval()

    elif method == "dust3r":
        sys.path.append("../../detector_free/dust3r")
        from dust3r.inference import inference
        from dust3r.utils.image import load_images
        from dust3r.image_pairs import make_pairs
        from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
        from dust3r.utils.geometry import find_reciprocal_matches,xy_grid
        from dust3r.model import AsymmetricCroCo3DStereo

        model = AsymmetricCroCo3DStereo.from_pretrained("../../detector_free/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth").cuda()
        model = model.eval()

    elif method == "orb":
        orb = cv2.ORB_create()
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    elif method == "rootsift":
        sift = cv2.SIFT_create()
        bf = cv2.BFMatcher()

    else:
        raise ValueError("Invalid method")

    data = json.load(open("../final_pairs.json"))

    # Get all scenes to get intrinsics to be able to recover pose
    all_scenes = list(set([scene for box in data for scene in data[box]]))
    
    results = {}

    for scene in tqdm(all_scenes):
        # We iterate over scenes whether than boxes to avoid loading all images from every scene at once
        for box in tqdm(data, leave=False):
            if data[box].get(scene) is None:
                continue

            if results.get(box) is None:
                results[box] = {}

            pairs = [eval(el) for el in list(data[box][scene].keys())]
            paths = [[osp.join("../..", "nuscenes", "sweeps", el[0].split("__")[1].split("__")[0], el[0]), 
                      osp.join("../..", "nuscenes", "sweeps", el[1].split("__")[1].split("__")[0], el[1])] for el in pairs]

            for i, pair in enumerate(paths):
                # Get gt pose
                gt_pose = np.array(data[box][scene][str(pairs[i])]["rel_pose"])

                # Get matches
                if method == "xfeat":
                    im1 = cv2.imread(pair[0])
                    im2 = cv2.imread(pair[1])
                    start = time.time()
                    mkpts1, mkpts2 = xfeat.match_xfeat(im1, im2, top_k=4096)
                    elapsed = time.time() - start

                elif method == "xfeat_star":
                    im1 = cv2.imread(pair[0])
                    im2 = cv2.imread(pair[1])
                    start = time.time()
                    mkpts1, mkpts2 = xfeat.match_xfeat_star(im1, im2, top_k=8000)
                    elapsed = time.time() - start

                elif method == "xfeat_lighterglue":
                    # Inference with batch = 1
                    im1 = cv2.imread(pair[0])
                    im2 = cv2.imread(pair[1])
                    start = time.time()
                    output1 = xfeat.detectAndCompute(im1, top_k=4096)[0]
                    output2 = xfeat.detectAndCompute(im2, top_k=4096)[0]
                    elapsed = time.time() - start

                    #Update with image resolution (required)
                    output1.update({'image_size': (im1.shape[1], im1.shape[0])})
                    output2.update({'image_size': (im2.shape[1], im2.shape[0])})

                    start = time.time()
                    mkpts1, mkpts2 = xfeat.match_lighterglue(output1, output2)
                    elapsed += time.time() - start

                elif method == "loftr":
                    img1_raw = cv2.imread(pair[0], cv2.IMREAD_GRAYSCALE)
                    img2_raw = cv2.imread(pair[1], cv2.IMREAD_GRAYSCALE)
                    
                    # Get original dimensions
                    H1, W1 = img1_raw.shape
                    H2, W2 = img2_raw.shape
                    
                    # Resize to be divisible by 8
                    img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1]//8*8, img1_raw.shape[0]//8*8))
                    img2_raw = cv2.resize(img2_raw, (img2_raw.shape[1]//8*8, img2_raw.shape[0]//8*8))

                    img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
                    img2 = torch.from_numpy(img2_raw)[None][None].cuda() / 255.
                    batch = {'image0': img1, 'image1': img2}

                    # Inference with LoFTR and get prediction
                    with torch.no_grad():
                        start = time.time()
                        loftr(batch)
                        elapsed = time.time() - start
                        mkpts1 = batch['mkpts0_f'].cpu().numpy()
                        mkpts2 = batch['mkpts1_f'].cpu().numpy()

                    # Scale matches back to original resolution
                    scale_w1 = W1 / img1_raw.shape[1]
                    scale_h1 = H1 / img1_raw.shape[0]
                    scale_w2 = W2 / img2_raw.shape[1]
                    scale_h2 = H2 / img2_raw.shape[0]

                    mkpts1[:, 0] *= scale_w1
                    mkpts1[:, 1] *= scale_h1
                    mkpts2[:, 0] *= scale_w2
                    mkpts2[:, 1] *= scale_h2

                elif "+lightglue" in method:
                    from lightglue.utils import load_image
                    from lightglue import match_pair

                    img1 = load_image(pair[0]).cuda()
                    img2 = load_image(pair[1]).cuda()

                    # Extract keypoints and descriptors
                    try:  # Problem for SIFT features
                        start = time.time()
                        feats0, feats1, matches01 = match_pair(extractor, matcher, img1, img2)
                        elapsed = time.time() - start

                        # Match keypoints
                        matches = matches01['matches']
                        points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
                        points1 = feats1['keypoints'][matches[..., 1]]

                        mkpts1, mkpts2 = points0.cpu().numpy(), points1.cpu().numpy()

                    except Exception as _: 
                        mkpts1, mkpts2 = np.array([]), np.array([])

                elif method == "roma":
                    start = time.time()
                    warp, certainty = roma_model.match(pair[0], pair[1], device="cuda")
                    elapsed = time.time() - start
                    # Sample matches for estimation
                    matches, certainty = roma_model.sample(warp, certainty)
                    # Convert to pixel coordinates (RoMa produces matches in [-1,1]x[-1,1])
                    mkpts1, mkpts2 = roma_model.to_pixel_coordinates(matches, 900, 1600, 900, 1600)
                    mkpts1, mkpts2 = mkpts1.cpu().numpy(), mkpts2.cpu().numpy()

                elif method == "dedode":
                    im_A = Image.open(pair[0])
                    im_B = Image.open(pair[1])
                    W_A, H_A = im_A.size
                    W_B, H_B = im_B.size

                    start = time.time()
                    detections_A = detector.detect_from_path(pair[0], num_keypoints=10_000)
                    elapsed = time.time() - start
                    keypoints_A, P_A = detections_A["keypoints"], detections_A["confidence"]

                    start = time.time()
                    detections_B = detector.detect_from_path(pair[1], num_keypoints=10_000)
                    elapsed += time.time() - start
                    keypoints_B, P_B = detections_B["keypoints"], detections_B["confidence"]

                    start = time.time()
                    description_A = descriptor.describe_keypoints_from_path(pair[0], keypoints_A)["descriptions"]
                    description_B = descriptor.describe_keypoints_from_path(pair[1], keypoints_B)["descriptions"]

                    matches_A, matches_B, batch_ids = matcher.match(keypoints_A, description_A,
                        keypoints_B, description_B,
                        P_A = P_A, P_B = P_B,
                        normalize = True, inv_temp=20, threshold=0.1)  #Increasing threshold -> fewer matches, fewer outliers
                    elapsed += time.time() - start

                    mkpts1, mkpts2 = matcher.to_pixel_coords(matches_A, matches_B, H_A, W_A, H_B, W_B)
                    mkpts1, mkpts2 = mkpts1.cpu().numpy(), mkpts2.cpu().numpy()

                elif method == "eloftr":
                    img1_raw = cv2.imread(pair[0], cv2.IMREAD_GRAYSCALE)
                    img2_raw = cv2.imread(pair[1], cv2.IMREAD_GRAYSCALE)
                    
                    # Get original dimensions
                    H1, W1 = img1_raw.shape
                    H2, W2 = img2_raw.shape
                    
                    # Resize to be divisible by 32
                    img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1]//32*32, img1_raw.shape[0]//32*32))
                    img2_raw = cv2.resize(img2_raw, (img2_raw.shape[1]//32*32, img2_raw.shape[0]//32*32))

                    img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
                    img2 = torch.from_numpy(img2_raw)[None][None].cuda() / 255.
                    batch = {'image0': img1, 'image1': img2}

                    # Inference with eLoFTR and get prediction
                    with torch.no_grad():
                        start = time.time()
                        matcher(batch)
                        elapsed = time.time() - start
                        mkpts1 = batch['mkpts0_f'].cpu().numpy()
                        mkpts2 = batch['mkpts1_f'].cpu().numpy()

                    # Scale matches back to original resolution
                    scale_w1 = W1 / img1_raw.shape[1]
                    scale_h1 = H1 / img1_raw.shape[0]
                    scale_w2 = W2 / img2_raw.shape[1]
                    scale_h2 = H2 / img2_raw.shape[0]
                    
                    mkpts1[:, 0] *= scale_w1
                    mkpts1[:, 1] *= scale_h1
                    mkpts2[:, 0] *= scale_w2
                    mkpts2[:, 1] *= scale_h2

                elif method == "aspanformer":
                    img0, img1 = cv2.imread(pair[0]), cv2.imread(pair[1])
                    img0_g, img1_g = cv2.imread(pair[0], 0), cv2.imread(pair[1], 0)
                    
                    # Get original dimensions
                    H1, W1 = img0.shape[:2]
                    H2, W2 = img1.shape[:2]

                    img0, img1 = demo_utils.resize(img0, 1024), demo_utils.resize(img1, 1024)
                    img0_g, img1_g = demo_utils.resize(img0_g, 1024), demo_utils.resize(img1_g, 1024)
                    H1_g, W1_g = img0_g.shape[:2]
                    H2_g, W2_g = img1_g.shape[:2]

                    batch={'image0': torch.from_numpy(img0_g/255.)[None,None].cuda().float(),
                           'image1': torch.from_numpy(img1_g/255.)[None,None].cuda().float()} 

                    with torch.no_grad():
                        start = time.time() 
                        matcher(batch, online_resize=True)
                        elapsed = time.time() - start
                        mkpts1, mkpts2 = batch['mkpts0_f'].cpu().numpy(), batch['mkpts1_f'].cpu().numpy()

                    # Scale matches back to original resolution
                    scale_w1 = W1 / W1_g
                    scale_h1 = H1 / H1_g
                    scale_w2 = W2 / W2_g
                    scale_h2 = H2 / H2_g

                    mkpts1[:, 0] *= scale_w1
                    mkpts1[:, 1] *= scale_h1
                    mkpts2[:, 0] *= scale_w2
                    mkpts2[:, 1] *= scale_h2

                elif method == "mast3r":
                    # Run inference
                    images = load_images(pair, size=512, verbose=False)
                    start = time.time()
                    output = inference([tuple(images)], model, "cuda", batch_size=1, verbose=False)
                    elapsed = time.time() - start

                    # raw predictions
                    view1, pred1 = output['view1'], output['pred1']
                    view2, pred2 = output['view2'], output['pred2']
                    
                    desc1 = pred1['desc'].squeeze(0).detach()
                    desc2 = pred2['desc'].squeeze(0).detach()
                    
                    # find 2D-2D matches between the two images
                    start = time.time()
                    matches_im1, matches_im2 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8, device="cuda", dist='dot', block_size=2**13)
                    elapsed += time.time() - start
                    
                    # ignore small border around the edge
                    H1, W1 = view1['true_shape'][0]
                    valid_matches_im1 = (matches_im1[:, 0] >= 3) & \
                                        (matches_im1[:, 0] < int(W1) - 3) & \
                                        (matches_im1[:, 1] >= 3) & \
                                        (matches_im1[:, 1] < int(H1) - 3)
                    
                    H2, W2 = view2['true_shape'][0]
                    valid_matches_im2 = (matches_im2[:, 0] >= 3) & \
                                        (matches_im2[:, 0] < int(W2) - 3) & \
                                        (matches_im2[:, 1] >= 3) & \
                                        (matches_im2[:, 1] < int(H2) - 3)
                    
                    valid_matches = valid_matches_im1 & valid_matches_im2
                    
                    # matches are Nx2 image coordinates
                    mkpts1 = matches_im1[valid_matches].astype(float)
                    mkpts2 = matches_im2[valid_matches].astype(float)

                    # Scale matches back to original resolution (1600x900) to get depths
                    scale_w1 = 1600. / W1.item()
                    scale_h1 = 900. / H1.item()
                    scale_w2 = 1600. / W2.item()
                    scale_h2 = 900. / H2.item()

                    mkpts1[:, 0] *= scale_w1  # Scale x coordinates
                    mkpts1[:, 1] *= scale_h1  # Scale y coordinates
                    mkpts2[:, 0] *= scale_w2
                    mkpts2[:, 1] *= scale_h2

                elif method == "dust3r":
                    # Make images and pred
                    images = load_images(pair, size=512, verbose=False)
                    images = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
                    start = time.time()
                    pred = inference(images, model, "cuda", batch_size=1, verbose=False)

                    # Get final pose
                    align = global_aligner(pred, device="cuda", mode=GlobalAlignerMode.PairViewer, verbose=False)
                    elapsed = time.time() - start
                    imgs = align.imgs
                    pts3d = align.get_pts3d()
                    confidence_masks = align.get_masks()

                    try:
                        # Get matches to extract scale
                        pts2d_list, pts3d_list = [], []
                        for j in range(2):
                            conf_j = confidence_masks[j].cpu().numpy()
                            pts2d_list.append(xy_grid(*imgs[j].shape[:2][::-1])[conf_j])
                            pts3d_list.append(pts3d[j].detach().cpu().numpy()[conf_j])

                        reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(*pts3d_list)
                        matches_im2 = pts2d_list[1][reciprocal_in_P2]
                        matches_im1 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]

                        # Scale matches back to original resolution (1600x900)
                        scale_w = 1600. / imgs[0].shape[1]
                        scale_h = 900. / imgs[0].shape[0]

                        mkpts1 = matches_im1.copy().astype(float)
                        mkpts2 = matches_im2.copy().astype(float)
                        
                        mkpts1[:, 0] *= scale_w
                        mkpts1[:, 1] *= scale_h
                        mkpts2[:, 0] *= scale_w
                        mkpts2[:, 1] *= scale_h

                    except Exception as _:
                        mkpts1, mkpts2 = np.array([]), np.array([])

                elif method == "orb":
                    img1 = cv2.imread(pair[0], cv2.IMREAD_GRAYSCALE)
                    img2 = cv2.imread(pair[1], cv2.IMREAD_GRAYSCALE)

                    start = time.time()
                    kp1, des1 = orb.detectAndCompute(img1, None)
                    kp2, des2 = orb.detectAndCompute(img2, None)
                    matches = bf.match(des1, des2)
                    elapsed = time.time() - start

                    mkpts1 = np.array([kp1[m.queryIdx].pt for m in matches])
                    mkpts2 = np.array([kp2[m.trainIdx].pt for m in matches])

                elif method == "rootsift":
                    img1 = cv2.imread(pair[0], cv2.IMREAD_GRAYSCALE)
                    img2 = cv2.imread(pair[1], cv2.IMREAD_GRAYSCALE)

                    start = time.time()
                    kp1, des1 = sift.detectAndCompute(img1, None)
                    kp2, des2 = sift.detectAndCompute(img2, None)
                    matches = bf.knnMatch(des1, des2, k=2)
                    elapsed = time.time() - start

                    # Apply ratio test
                    good = []
                    for m,n in matches:
                        if m.distance < 0.75*n.distance:
                            good.append(m)

                    mkpts1 = np.array([kp1[m.queryIdx].pt for m in good])
                    mkpts2 = np.array([kp2[m.trainIdx].pt for m in good])

                # Estimate pose
                K1 = np.array(data[box][scene][str(pairs[i])]["K1"])
                K2 = np.array(data[box][scene][str(pairs[i])]["K2"])
                if estimate_pose == "essential":
                    ret = estimate_pose_essential(mkpts1, mkpts2, K1, K2, 0.5)

                elif estimate_pose == "fundamental":
                    ret = estimate_pose_fundamental(mkpts1, mkpts2, K1, K2, 0.5)

                if ret is None:
                    results[box][str(pairs[i])] = {"R_est": None,
                                                   "t_est": None,
                                                   "t_err_angle": np.inf, 
                                                   "t_err_metric": np.inf,
                                                   "R_err": np.inf,
                                                   "time": round(elapsed, 5)}
                
                else:
                    R_est, t_est, _ = ret

                    # Normalize t_est
                    t_est = t_est / np.linalg.norm(t_est)

                    # Get scale factor using unidepths by minimizing distance to 3D points after applying transformation
                    depth1 = np.load(f"../unidepths/{osp.basename(pair[0]).replace('.jpg', '.npy')}")
                    depth2 = np.load(f"../unidepths/{osp.basename(pair[1]).replace('.jpg', '.npy')}")

                    # Get 3D points
                    pts3D_1 = backproject_to_3D(mkpts1, depth1, K1)
                    pts3D_2 = backproject_to_3D(mkpts2, depth2, K2)

                    # Get scale factor
                    scale = get_scale(scale_cost_function, 1, R_est, t_est, pts3D_1, pts3D_2)
                    t_est *= scale
                
                    # Compute error
                    t_err_angle, t_err_metric, R_err = relative_pose_error(gt_pose, R_est, t_est)

                    results[box][str(pairs[i])] = {"R_est": R_est.tolist(),
                                                   "t_est": t_est.tolist(),
                                                   "t_err_angle": round(t_err_angle, 5),
                                                   "t_err_metric": round(t_err_metric, 5), 
                                                   "R_err": round(R_err, 5),
                                                   "time": round(elapsed, 5)}

    # Save results
    json.dump(results, open(f"results/results_{method}_{estimate_pose}.json", "w"), indent=2)