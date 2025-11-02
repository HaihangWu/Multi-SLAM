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

    def global_graph_opt(self):
        print("\n=== Starting Global Graph Optimization ===")

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        retrieval_database = load_retriever(self.model, device=device)
        global_factor_graph = FactorGraph(self.model, None, device=device)

        # Step 1: Collect all keyframes
        all_keyframes = []
        agent_offsets = {}
        offset = 0
        for agent_id, kfs in self.keyframes.items():
            n_kf = len(kfs)
            all_keyframes.extend(kfs)
            agent_offsets[agent_id] = (offset, offset + n_kf)
            offset += n_kf
        global_factor_graph.frames = all_keyframes
        print(f"Collected {len(all_keyframes)} keyframes from {len(self.keyframes)} agents")

        # Step 2: Cross-agent loop detection
        for id_a, (start_a, end_a) in agent_offsets.items():
            for id_b, (start_b, end_b) in agent_offsets.items():
                if id_a >= id_b:
                    continue
                for i in range(start_a, end_a):
                    frame_i = all_keyframes[i]
                    topk = retrieval_database.update(
                        frame_i,
                        add_after_query=False,
                        k=config["retrieval"]["k"],
                        min_thresh=config["retrieval"]["min_thresh"]
                    )
                    topk = [idx for idx in topk if start_b <= idx < end_b]
                    if topk:
                        frame_idx = [i] * len(topk)
                        global_factor_graph.add_factors(frame_idx, topk, config["local_opt"]["min_match_frac"])

        # Step 3: Optimize
        if config["use_calib"]:
            global_factor_graph.solve_GN_calib()
        else:
            global_factor_graph.solve_GN_rays()
        print("Global optimization completed.")

        # Step 4: Update poses in each agent's keyframes
        for agent_id, (start, end) in agent_offsets.items():
            for i, kf in enumerate(self.keyframes[agent_id]):
                kf.pose = global_factor_graph.frames[start + i].pose.clone()

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

    # def global_graph_opt(self):
    #     set_global_config(config)
    #
    #     torch.cuda.set_device(0)
    #     self.model = self.model.to("cuda:0")
    #     num_agents=len(args.datasets)
    #     for agent_id in range(num_agents):
    #
    #     K = None
    #     if use_calib:
    #         K = torch.from_numpy(self.dataset.camera_intrinsics.K_frame).to(
    #             device, dtype=torch.float32
    #         )
    #         self.keyframes[agent_id].set_intrinsics(K)
    #
    #     factor_graph = FactorGraph(self.model, self.keyframes[self.agent_id], K, self.device)
    #     retrieval_database = load_retriever(self.model,device=self.device)
    #
    #     # Graph Construction
    #     kf_idx = []
    #     # k to previous consecutive keyframes
    #     n_consec = 1
    #     for j in range(min(n_consec, idx)):
    #         kf_idx.append(idx - 1 - j)
    #     frame = self.keyframes[self.agent_id][idx]
    #     retrieval_inds = retrieval_database.update(
    #         frame,
    #         add_after_query=True,
    #         k=config["retrieval"]["k"],
    #         min_thresh=config["retrieval"]["min_thresh"],
    #     )
    #     kf_idx += retrieval_inds
    #
    #     lc_inds = set(retrieval_inds)
    #     lc_inds.discard(idx - 1)
    #     if len(lc_inds) > 0:
    #         print("Database retrieval", idx, ": ", lc_inds)
    #
    #     kf_idx = set(kf_idx)  # Remove duplicates by using set
    #     kf_idx.discard(idx)  # Remove current kf idx if included
    #     kf_idx = list(kf_idx)  # convert to list
    #     frame_idx = [idx] * len(kf_idx)
    #     if kf_idx:
    #         factor_graph.add_factors(
    #             kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
    #         )
    #
    #     with self.states[self.agent_id].lock:
    #         self.states[self.agent_id].edges_ii[:] = factor_graph.ii.cpu().tolist()
    #         self.states[self.agent_id].edges_jj[:] = factor_graph.jj.cpu().tolist()
    #
    #     if config["use_calib"]:
    #         factor_graph.solve_GN_calib()
    #     else:
    #         factor_graph.solve_GN_rays()
    #
    #
    #     if self.dataset.save_results:
    #         save_dir, seq_name = eval.prepare_savedir(self.save_directory, self.dataset)
    #         eval.save_traj(save_dir, f"{seq_name}.txt", self.dataset.timestamps, self.keyframes[self.agent_id])
    #         eval.save_reconstruction(
    #             save_dir,
    #             f"{seq_name}.ply",
    #             self.keyframes[self.agent_id],
    #             self.last_msg.C_conf_threshold,
    #         )
    #         eval.save_keyframes(
    #             save_dir / "keyframes" / seq_name, self.dataset.timestamps, self.keyframes[self.agent_id]
    #         )
    #     if self.save_frames:
    #         savedir = pathlib.Path(f"logs/frames/{self.datetime_now}")
    #         savedir.mkdir(exist_ok=True, parents=True)
    #         for i, frame in tqdm.tqdm(enumerate(frames), total=len(frames)):
    #             frame = (frame * 255).clip(0, 255)
    #             frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #             cv2.imwrite(f"{savedir}/{i}.png", frame)


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
    multi_agent_system.global_graph_opt()
