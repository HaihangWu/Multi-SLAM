import argparse
import torch
from mast3r_slam.agent import Agent
from mast3r_slam.config import load_config, config, set_global_config


from mast3r_slam.dataloader import Intrinsics, load_dataset
import torch.multiprocessing as mp


class MultiAgentSystem:
    def __init__(self):
        self.agents = []
        self.frontend_procs = []
        self.backend_procs = []
        self.states = {}  # Store shared states for each agent
        self.keyframes = {}  # Store shared keyframes for each agent

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
            agent = Agent(agent_id, args, dataset, self.states, self.keyframes,self.frontend_procs,
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
