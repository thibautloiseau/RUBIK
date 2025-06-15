import os
import os.path as osp
import torch
import json
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image


def get_unidepth():
    # Get model
    model = torch.hub.load("lpiccinelli-eth/UniDepth", 
                           "UniDepth", 
                           version="v2", 
                           backbone="vitl14", 
                           pretrained=True, 
                           trust_repo=True, 
                           force_reload=True)

    # Move to CUDA, if any
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nuscenes_path", type=str, default="./nuscenes", help="path to nuscenes dataset folder")
    args = parser.parse_args()

    nuscenes_path = args.nuscenes_path

    # Get all images we want to get depth for
    rubik = json.load(open("rubik.json", "r"))
    
    # Get UniDepth model and create output folder
    unidepth = get_unidepth()
    os.makedirs("unidepths", exist_ok=True)

    for box in tqdm(rubik):
        for scene in tqdm(rubik[box], leave=False):
            for pair in tqdm(rubik[box][scene], leave=False):
                # Get information about the pair
                image1, image2 = eval(pair)[0], eval(pair)[1]
                cam1 = image1.split("__")[1].split("__")[0]
                cam2 = image2.split("__")[1].split("__")[0]
                image1_path = osp.join(nuscenes_path, "sweeps", cam1, image1)
                image2_path = osp.join(nuscenes_path, "sweeps", cam2, image2)
                K1, K2 = np.array(rubik[box][scene][pair]["K1"]), np.array(rubik[box][scene][pair]["K2"])

                # The same image might be in several pairs
                # Image
                if not osp.exists(osp.join("unidepths", image1.replace(".jpg", ".npy"))):
                    rgb1 = torch.from_numpy(np.array(Image.open(image1_path))).permute(2, 0, 1) # C, H, W
                    intrinsics1 = torch.from_numpy(K1).float()

                    predictions1 = unidepth.infer(rgb1, intrinsics1)
                    depth1 = predictions1["depth"].squeeze().cpu().numpy().astype(np.float32)

                    np.save(osp.join("unidepths", image1.replace(".jpg", ".npy")), depth1)
                
                # Image 2
                if not osp.exists(osp.join("unidepths", image2.replace(".jpg", ".npy"))):
                    rgb2 = torch.from_numpy(np.array(Image.open(image2_path))).permute(2, 0, 1) # C, H, W
                    intrinsics2 = torch.from_numpy(K2).float()

                    predictions2 = unidepth.infer(rgb2, intrinsics2)
                    depth2 = predictions2["depth"].squeeze().cpu().numpy().astype(np.float32)

                    np.save(osp.join("unidepths", image2.replace(".jpg", ".npy")), depth2)

    return


if __name__ == "__main__":
    main()