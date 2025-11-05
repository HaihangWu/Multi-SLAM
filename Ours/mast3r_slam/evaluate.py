import pathlib
from typing import Optional
import cv2
import numpy as np
import torch
from mast3r_slam.dataloader import Intrinsics
from mast3r_slam.frame import SharedKeyframes
from mast3r_slam.lietorch_utils import as_SE3
from mast3r_slam.config import config
from mast3r_slam.geometry import constrain_points_to_ray
from plyfile import PlyData, PlyElement
import lietorch


def prepare_savedir(save_directory, dataset):
    save_dir = pathlib.Path("logs")
    if save_directory != "default":
        save_dir = save_dir / save_directory
    save_dir.mkdir(exist_ok=True, parents=True)
    seq_name = dataset.dataset_path.stem
    return save_dir, seq_name


def save_traj(
    logdir,
    logfile,
    timestamps,
    frames: SharedKeyframes,
    intrinsics: Optional[Intrinsics] = None,
):
    # log
    logdir = pathlib.Path(logdir)
    logdir.mkdir(exist_ok=True, parents=True)
    logfile = logdir / logfile
    with open(logfile, "w") as f:
        # for keyframe_id in frames.keyframe_ids:
        for i in range(len(frames)):
            keyframe = frames[i]
            t = timestamps[keyframe.frame_id]
            # frame_id = int(frames.dataset_idx[i].item()) #NEW
            # t = timestamps[frame_id] #NEW
            # T_sim3 = lietorch.Sim3(frames.T_WC[i]) #NEW
            if intrinsics is None:
                T_WC = as_SE3(keyframe.T_WC)
                # T_WC = as_SE3(T_sim3) #NEW
                print("save results 1:")
            else:
                T_WC = intrinsics.refine_pose_with_calibration(keyframe) #?
            x, y, z, qx, qy, qz, qw = T_WC.data.numpy().reshape(-1)
            print("save results final:",x, y, z, qx, qy, qz, qw)
            f.write(f"{t} {x} {y} {z} {qx} {qy} {qz} {qw}\n")



def save_reconstruction(savedir, filename, keyframes, c_conf_threshold):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    pointclouds = []
    colors = []
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        # if config["use_calib"]:
        #     X_canon = constrain_points_to_ray(
        #         keyframe.img_shape.flatten()[:2], keyframe.X_canon[None], keyframe.K
        #     )
        #     keyframe.X_canon = X_canon.squeeze(0)

        if config["use_calib"]: #NEW
            X_canon = constrain_points_to_ray( #NEW
                keyframes.img_shape[i].flatten()[:2], #NEW
                keyframes.X[i][None], #NEW
                keyframes.K, #NEW
            ).squeeze(0) #NEW
        else:
            X_canon = keyframes.X[i] #NEW

        #pW = keyframe.T_WC.act(keyframe.X_canon).cpu().numpy().reshape(-1, 3)
        T_sim3 = lietorch.Sim3(keyframes.T_WC[i])#NEw
        pW = T_sim3.act(keyframes.X[i]).cpu().numpy().reshape(-1, 3)#NEW
        #color = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)
        color = (keyframes.uimg[i].cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)#NEW
        valid = (
            keyframe.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)
            > c_conf_threshold
        )
        pointclouds.append(pW[valid])
        colors.append(color[valid])
    pointclouds = np.concatenate(pointclouds, axis=0)
    colors = np.concatenate(colors, axis=0)

    save_ply(savedir / filename, pointclouds, colors)


def save_keyframes(savedir, timestamps, keyframes: SharedKeyframes):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    for i in range(len(keyframes)):
        # keyframe = keyframes[i]
        # t = timestamps[keyframe.frame_id]
        # filename = savedir / f"{t}.png"
        # cv2.imwrite(
        #     str(filename),
        #     cv2.cvtColor(
        #         (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
        #     ),
        # )
        frame_id = int(keyframes.dataset_idx[i].item()) #NEW
        t = timestamps[frame_id] #NEW
        uimg = keyframes.uimg[i].cpu().numpy() #NEW
        cv2.imwrite(str(savedir / f"{t}.png"), cv2.cvtColor((uimg * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)) #NEW


def save_ply(filename, points, colors):
    colors = colors.astype(np.uint8)
    # Combine XYZ and RGB into a structured array
    pcd = np.empty(
        len(points),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    pcd["x"], pcd["y"], pcd["z"] = points.T
    pcd["red"], pcd["green"], pcd["blue"] = colors.T
    vertex_element = PlyElement.describe(pcd, "vertex")
    ply_data = PlyData([vertex_element], text=False)
    ply_data.write(filename)
