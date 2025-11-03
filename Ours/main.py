import argparse
import torch
from mast3r_slam import evaluate as eval
from mast3r_slam.agent import Agent
from mast3r_slam.config import load_config, config, set_global_config
from mast3r_slam.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
)

from mast3r_slam.dataloader import Intrinsics, load_dataset
import torch.multiprocessing as mp
from mast3r_slam.global_opt import FactorGraph
import pathlib
import tqdm
import cv2
import numpy as np
from mast3r_slam.frame import SharedKeyframes


class MultiAgentSystem:
    def __init__(self):
        self.agents = []
        self.frontend_procs = []
        self.backend_procs = []
        self.states = {}  # Store shared states for each agent
        self.keyframes = {}  # Store shared keyframes for each agent
        self.model = load_mast3r(device="cpu")
        self.model.share_memory()

    def initialize_agents(self,args, manager):
        # Initialize pipes and agents
        load_config(args.config)
        num_agents=len(args.datasets)
        for agent_id in range(num_agents):
            dataset =args.datasets[agent_id]
            parts = dataset.split('_', 1)  # split at first '_'
            scene = parts[0]  # first part
            agent = parts[1]  # everything after first '_'

            # Reconstruct the new folder structure: room0/agent_0/
            full_dataset_path = f"{args.base_dataset_path}{scene}/{agent}/results/"
            dataset = load_dataset(full_dataset_path)
            print(full_dataset_path)

            # Create agent instance
            agent = Agent(args, agent_id, dataset, self.model, self.states, self.keyframes,self.frontend_procs,
            self.backend_procs, manager,device=f"cuda:{agent_id}")
            self.agents.append(agent)

    def start_agents(self):
        # Start agent processes

        # Start all processes
        processes=self.frontend_procs + self.backend_procs
        for p in processes:
            p.start()

        # Wait for all processes to complete
        for p in processes:
            p.join()

    def global_graph_opt(self,manager):
        print("\n=== Starting Global Graph Optimization ===")

        device = "cuda:0"
        torch.cuda.set_device(0)
        self.model = self.model.to(device)
        retrieval_database = load_retriever(self.model, device=device)

        # Step 1: Collect all keyframes
        h, w = self.agents[0].dataset.get_img_shape()[0]
        print("h,w",h,w)
        global_kfs = SharedKeyframes(manager, h, w, device=device)
        global_factor_graph = FactorGraph(self.model, global_kfs, device=device)

        agent_offsets = {}
        offset = 0

        for agent_id, kfs in self.keyframes.items():
            n_kf = len(kfs)
            for i in range(n_kf):
                kf = kfs[i]

                # Move all relevant tensors to the target device
                kf.img = kf.img.to(device, non_blocking=True)
                kf.img_shape = kf.img_shape.to(device, non_blocking=True)
                kf.img_true_shape = kf.img_true_shape.to(device, non_blocking=True)
                kf.T_WC = kf.T_WC.to(device, non_blocking=True)
                kf.X_canon = kf.X_canon.to(device, non_blocking=True)
                kf.C = kf.C.to(device, non_blocking=True)
                kf.feat = kf.feat.to(device, non_blocking=True)
                kf.pos = kf.pos.to(device, non_blocking=True)
                if hasattr(kf, "K") and kf.K is not None:
                    kf.K = kf.K.to(device, non_blocking=True)

                global_kfs.append(kf)
                # Graph Construction
                kf_idx = []
                # k to previous consecutive keyframes
                n_consec = 1
                for j in range(min(n_consec, i)):
                    kf_idx.append(i - 1 - j)
                frame_idx = [i] * len(kf_idx)
                if kf_idx:
                    global_factor_graph.add_factors(
                        kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
                    )

            agent_offsets[agent_id] = (offset, offset + n_kf)
            offset += n_kf

        total_num_keyframes=len(global_kfs)
        for i in range(total_num_keyframes):
            retrieval_database.update(global_kfs[i], add_after_query=True, k=config["retrieval"]["k"])

        print(f"Collected {total_num_keyframes} keyframes from {len(self.keyframes)} agents")


        for agent_id, (start, end) in agent_offsets.items():
            for i in range(start, end):
                print("global graph TWC 1",global_factor_graph.frames.T_WC[i])
        print("global graph edge 1",global_factor_graph.idx_ii2jj)

        # Step 2: Cross-agent loop detection
        for id_a, (start_a, end_a) in agent_offsets.items():
            for id_b, (start_b, end_b) in agent_offsets.items():
                if id_a >= id_b:
                    continue
                for i in range(start_a, end_a):
                    frame_i = global_kfs[i]
                    topk = retrieval_database.update(
                        frame_i,
                        add_after_query=False,
                        k=config["retrieval"]["k"],
                        min_thresh=config["retrieval"]["min_thresh"]
                    )
                    topk = [idx for idx in topk if start_b <= idx < end_b]
                    if topk:
                        print("topk",i,topk)
                        frame_idx = [i] * len(topk)
                        global_factor_graph.add_factors(frame_idx, topk, config["local_opt"]["min_match_frac"])

        # Step 3: Optimize
        if config["use_calib"]:
            global_factor_graph.solve_GN_calib()
        else:
            global_factor_graph.solve_GN_rays()
        print("Global optimization completed.")

        for agent_id, (start, end) in agent_offsets.items():
            for i in range(start, end):
                print("global graph TWC 2",global_factor_graph.frames.T_WC[i])
        print("global graph edge 2",global_factor_graph.idx_ii2jj)

        # Step 4: Update poses in each agent's keyframes
        for agent_id, (start, end) in agent_offsets.items():
            device_tmp=self.keyframes[agent_id].device
            for i in range(start, end):
                if global_kfs.device != device_tmp:
                    T_WC_tmp = global_kfs.T_WC[i].to(device_tmp)
                else:
                    T_WC_tmp = global_kfs.T_WC[i]
                print("keyframes",self.keyframes[agent_id].T_WC[i - start],"tmp",T_WC_tmp)
                self.keyframes[agent_id].update_T_WCs(T_WC_tmp, i - start)

            # Step 5: Save results
            if self.agents[agent_id].dataset.save_results:
                save_dir, seq_name = eval.prepare_savedir(self.agents[agent_id].save_directory, self.agents[agent_id].dataset)
                eval.save_traj(save_dir, f"{seq_name}.txt", self.agents[agent_id].dataset.timestamps, self.keyframes[agent_id])
                eval.save_reconstruction(
                    save_dir,
                    f"{seq_name}.ply",
                    self.keyframes[agent_id],
                    self.agents[agent_id].last_msg.C_conf_threshold,
                )
                eval.save_keyframes(
                    save_dir / "keyframes" / seq_name, self.agents[agent_id].dataset.timestamps, self.keyframes[agent_id]
                )

            if self.agents[agent_id].save_frames:
                savedir = pathlib.Path(f"logs/frames/{self.agents[agent_id].datetime_now}")
                savedir.mkdir(exist_ok=True, parents=True)
                for i, frame in tqdm.tqdm(enumerate(self.agents[agent_id].frames), total=len(self.agents[agent_id].frames)):
                    frame_img = (frame * 255).clip(0, 255).astype(np.uint8)
                    frame_img = cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f"{savedir}/{i}.png", frame_img)

        print("=== Global Graph Optimization & Saving Finished ===\n")



if __name__ == "__main__":
    # Configuration and model setup
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dataset_path", default="datasets/tum/rgbd_dataset_freiburg1_desk")
    parser.add_argument(
        "--datasets",
        nargs="+",  # accept one or more values
        required=True,
        help="List of dataset names or paths"
    )
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--save-as", default="default")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--calib", default="")
    # parser.add_argument("--agents",type=int, default=1, help="number of agents")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    mp.set_start_method("spawn")
    manager = mp.Manager()

    # model = load_mast3r(device="cpu")
    # model.share_memory()

    # Instantiate the multi-agent system
    multi_agent_system = MultiAgentSystem()

    # Initialize and start agents
    multi_agent_system.initialize_agents(args, manager)
    multi_agent_system.start_agents()

    # Global Graph Optimization
    multi_agent_system.global_graph_opt(manager)

    # T_WC_tmp = global_kfs[i].T_WC.to(device_tmp)
    # self.keyframes[agent_id].update_T_WCs(T_WC_tmp,i-start)
    # .keyframes[agent_id][i - start].T_WC = global_kfs[i].T_WC.clone()
