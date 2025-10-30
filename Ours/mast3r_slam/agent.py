import lietorch
from .frame import Mode, SharedKeyframes, SharedStates, create_frame
from .mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
)
from . import evaluate as eval
from .config import config, set_global_config
from .global_opt import FactorGraph
import torch.multiprocessing as mp
from .visualization import WindowMsg, run_visualization
import torch
import sys
import pathlib
import yaml
from .dataloader import Intrinsics, load_dataset
from .tracker import FrameTracker
import tqdm
import time
import cv2
import datetime
import os

class Agent:
    def __init__(self, agent_id, args, dataset, states, keyframes, frontend_procs,
                 backend_procs, manager, device):
        self.agent_id = agent_id
        self.args=args
        self.device = device
        self.save_frames = False
        self.datetime_now = str(datetime.datetime.now()).replace(" ", "_")
        self.save_directory = os.path.join(self.args.save_as, self.args.datasets[self.agent_id])
        self.dataset = dataset
        print("agent",config)
        self.dataset.subsample(config["dataset"]["subsample"])
        h, w = self.dataset.get_img_shape()[0]

        if args.calib:
            with open(args.calib, "r") as f:
                intrinsics = yaml.load(f, Loader=yaml.SafeLoader)
            config["use_calib"] = True
            self.dataset.use_calibration = True
            self.dataset.camera_intrinsics = Intrinsics.from_calib(
                self.dataset.img_size,
                intrinsics["width"],
                intrinsics["height"],
                intrinsics["calibration"],
            )

        states[agent_id] = SharedStates(manager, h, w)
        keyframes[agent_id] = SharedKeyframes(manager, h, w)
        self.states = states
        self.keyframes = keyframes

        self.model = load_mast3r(device=device)
        has_calib = self.dataset.has_calib()
        use_calib = config["use_calib"]

        if use_calib and not has_calib:
            print("[Warning] No calibration provided for this dataset!")
            sys.exit(0)
        K = None
        if use_calib:
            K = torch.from_numpy(self.dataset.camera_intrinsics.K_frame).to(
                device, dtype=torch.float32
            )
            self.keyframes[agent_id].set_intrinsics(K)

        if self.dataset.save_results:
            save_dir, seq_name = eval.prepare_savedir(self.save_directory, self.dataset)
            traj_file = save_dir / f"{seq_name}.txt"
            recon_file = save_dir / f"{seq_name}.ply"
            if traj_file.exists():
                traj_file.unlink()
            if recon_file.exists():
                recon_file.unlink()

        self.tracker = FrameTracker(self.model, self.keyframes[agent_id], device)
        self.last_msg = WindowMsg()
        frontend_procs.append(mp.Process(target=self.run_frontend,args=(config, )))
        backend_procs.append(mp.Process(target=self.run_backend,args=(config, K)))

        # self.initialize_agent()
        # self.model = model.to(device)
    #
    # def initialize_agent(self,args):
    #     # Each agent’s main process

    def run_frontend(self,cfg):
        set_global_config(cfg)
        print(f"Agent {self.agent_id} is tracking...")
        i = 0
        fps_timer = time.time()

        frames = []
        while True:
            mode = self.states[self.agent_id].get_mode()
            # msg = try_get_msg(viz2main)
            # self.last_msg = msg if msg is not None else last_msg
            # if self.last_msg.is_terminated:
            #     states.set_mode(Mode.TERMINATED)
            #     break
            #
            # if self.last_msg.is_paused and not self.last_msg.next:
            #     states.pause()
            #     time.sleep(0.01)
            #     continue
            #
            # if not last_msg.is_paused:
            #     states.unpause()

            if i == len(self.dataset):
                self.states[self.agent_id].set_mode(Mode.TERMINATED)
                break

            timestamp, img = self.dataset[i]
            if self.save_frames:
                frames.append(img)

            # get frames last camera pose
            T_WC = (
                lietorch.Sim3.Identity(1, device=self.device)
                if i == 0
                else self.states[self.agent_id].get_frame().T_WC
            )
            frame = create_frame(i, img, T_WC, img_size=self.dataset.img_size, device=self.device)

            if mode == Mode.INIT:
                # Initialize via mono inference, and encoded features neeed for database
                X_init, C_init = mast3r_inference_mono(self.model, frame)
                frame.update_pointmap(X_init, C_init)
                self.keyframes[self.agent_id].append(frame)
                self.states[self.agent_id].queue_global_optimization(len(self.keyframes[self.agent_id]) - 1)
                self.states[self.agent_id].set_mode(Mode.TRACKING)
                self.states[self.agent_id].set_frame(frame)
                i += 1
                continue

            if mode == Mode.TRACKING:
                add_new_kf, match_info, try_reloc = self.tracker.track(frame)
                if try_reloc:
                    self.states[self.agent_id].set_mode(Mode.RELOC)
                self.states[self.agent_id].set_frame(frame)

            elif mode == Mode.RELOC:
                X, C = mast3r_inference_mono(self.model, frame)
                frame.update_pointmap(X, C)
                self.states[self.agent_id].set_frame(frame)
                self.states[self.agent_id].queue_reloc()
                # In single threaded mode, make sure relocalization happen for every frame
                while config["single_thread"]:
                    with self.states[self.agent_id].lock:
                        if self.states[self.agent_id].reloc_sem.value == 0:
                            break
                    time.sleep(0.01)

            else:
                raise Exception("Invalid mode")

            if add_new_kf:
                self.keyframes[self.agent_id].append(frame)
                self.states[self.agent_id].queue_global_optimization(len(self.keyframes[self.agent_id]) - 1)
                # In single threaded mode, wait for the backend to finish
                while config["single_thread"]:
                    with self.states[self.agent_id].lock:
                        if len(self.states[self.agent_id].global_optimizer_tasks) == 0:
                            break
                    time.sleep(0.01)
            # log time
            if i % 30 == 0:
                FPS = i / (time.time() - fps_timer)
                print(f"FPS: {FPS}")
            i += 1

        if self.dataset.save_results:
            save_dir, seq_name = eval.prepare_savedir(self.save_directory, self.dataset)
            eval.save_traj(save_dir, f"{seq_name}.txt", self.dataset.timestamps, self.keyframes[self.agent_id])
            eval.save_reconstruction(
                save_dir,
                f"{seq_name}.ply",
                self.keyframes[self.agent_id],
                self.last_msg.C_conf_threshold,
            )
            eval.save_keyframes(
                save_dir / "keyframes" / seq_name, self.dataset.timestamps, self.keyframes[self.agent_id]
            )
        if self.save_frames:
            savedir = pathlib.Path(f"logs/frames/{self.datetime_now}")
            savedir.mkdir(exist_ok=True, parents=True)
            for i, frame in tqdm.tqdm(enumerate(frames), total=len(frames)):
                frame = (frame * 255).clip(0, 255)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"{savedir}/{i}.png", frame)

    def run_backend(self, cfg, K):
        set_global_config(cfg)

        device = self.keyframes[self.agent_id].device
        factor_graph = FactorGraph(self.model, self.keyframes[self.agent_id], K, device)
        retrieval_database = load_retriever(self.model)

        mode = self.states[self.agent_id].get_mode()
        while mode is not Mode.TERMINATED:
            mode = self.states[self.agent_id].get_mode()
            if mode == Mode.INIT or self.states[self.agent_id].is_paused():
                time.sleep(0.01)
                continue
            if mode == Mode.RELOC:
                frame = self.states[self.agent_id].get_frame()
                success = self.relocalization(frame, factor_graph, retrieval_database)
                if success:
                    self.states[self.agent_id].set_mode(Mode.TRACKING)
                self.states[self.agent_id].dequeue_reloc()
                continue
            idx = -1
            with self.states[self.agent_id].lock:
                if len(self.states[self.agent_id].global_optimizer_tasks) > 0:
                    idx = self.states[self.agent_id].global_optimizer_tasks[0]
            if idx == -1:
                time.sleep(0.01)
                continue

            # Graph Construction
            kf_idx = []
            # k to previous consecutive keyframes
            n_consec = 1
            for j in range(min(n_consec, idx)):
                kf_idx.append(idx - 1 - j)
            frame = self.keyframes[self.agent_id][idx]
            retrieval_inds = retrieval_database.update(
                frame,
                add_after_query=True,
                k=config["retrieval"]["k"],
                min_thresh=config["retrieval"]["min_thresh"],
            )
            kf_idx += retrieval_inds

            lc_inds = set(retrieval_inds)
            lc_inds.discard(idx - 1)
            if len(lc_inds) > 0:
                print("Database retrieval", idx, ": ", lc_inds)

            kf_idx = set(kf_idx)  # Remove duplicates by using set
            kf_idx.discard(idx)  # Remove current kf idx if included
            kf_idx = list(kf_idx)  # convert to list
            frame_idx = [idx] * len(kf_idx)
            if kf_idx:
                factor_graph.add_factors(
                    kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
                )

            with self.states[self.agent_id].lock:
                self.states[self.agent_id].edges_ii[:] = factor_graph.ii.cpu().tolist()
                self.states[self.agent_id].edges_jj[:] = factor_graph.jj.cpu().tolist()

            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()

            with self.states[self.agent_id].lock:
                if len(self.states[self.agent_id].global_optimizer_tasks) > 0:
                    idx = self.states[self.agent_id].global_optimizer_tasks.pop(0)

    def relocalization(self, frame, factor_graph, retrieval_database):
        # we are adding and then removing from the keyframe, so we need to be careful.
        # The lock slows viz down but safer this way...
        with self.keyframes[self.agent_id].lock:
            kf_idx = []
            retrieval_inds = retrieval_database.update(
                frame,
                add_after_query=False,
                k=config["retrieval"]["k"],
                min_thresh=config["retrieval"]["min_thresh"],
            )
            kf_idx += retrieval_inds
            successful_loop_closure = False
            if kf_idx:
                self.keyframes[self.agent_id].append(frame)
                n_kf = len(self.keyframes[self.agent_id])
                kf_idx = list(kf_idx)  # convert to list
                frame_idx = [n_kf - 1] * len(kf_idx)
                print("RELOCALIZING against kf ", n_kf - 1, " and ", kf_idx)
                if factor_graph.add_factors(
                        frame_idx,
                        kf_idx,
                        config["reloc"]["min_match_frac"],
                        is_reloc=config["reloc"]["strict"],
                ):
                    retrieval_database.update(
                        frame,
                        add_after_query=True,
                        k=config["retrieval"]["k"],
                        min_thresh=config["retrieval"]["min_thresh"],
                    )
                    print("Success! Relocalized")
                    successful_loop_closure = True
                    self.keyframes[self.agent_id].T_WC[n_kf - 1] = self.keyframes[self.agent_id].T_WC[kf_idx[0]].clone()
                else:
                    self.keyframes[self.agent_id].pop_last()
                    print("Failed to relocalize")

            if successful_loop_closure:
                if config["use_calib"]:
                    factor_graph.solve_GN_calib()
                else:
                    factor_graph.solve_GN_rays()
            return successful_loop_closure







    # class Agent(object):
    #     def __init__(self, agent_id: int, run_info: dict, config: dict) -> None:
    #         self.device = "cuda"
    #         config = deepcopy(config)
    #         self.run_info = run_info
    #
    #         self.output_path = Path(config["data"]["output_path"]) / f"agent_{self.agent_id}"
    #         self.scene_name = config["data"]["scene_name"]
    #         self.dataset_name = config["dataset_name"]
    #         agent_input_path = sorted(Path(config["data"]["input_path"]).glob("*"))[self.agent_id]
    #         config["data"]["input_path"] = str(agent_input_path)
    #         self.dataset = get_dataset(config["dataset_name"])({**config["data"], **config["cam"]})






