import argparse
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import numpy as np

import datasets
from positional_embeddings import PositionalEmbedding

from policy_gradients import DiffusionActor, PGAgent


class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))


class MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int = 128,
        hidden_layers: int = 3,
        emb_size: int = 128,
        time_emb: str = "sinusoidal",
        input_emb: str = "sinusoidal",
    ):
        super().__init__()

        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0)

        concat_size = (
            len(self.time_mlp.layer)
            + len(self.input_mlp1.layer)
            + len(self.input_mlp2.layer)
        )
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, 2))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        x1_emb = self.input_mlp1(x[:, 0])
        x2_emb = self.input_mlp2(x[:, 1])
        t_emb = self.time_mlp(t)
        x = torch.cat((x1_emb, x2_emb, t_emb), dim=-1)
        x = self.joint_mlp(x)
        return x


class NoiseScheduler:
    def __init__(
        self,
        num_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
    ):
        self.num_timesteps = num_timesteps
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32
            )
        elif beta_schedule == "quadratic":
            self.betas = (
                torch.linspace(
                    beta_start**0.5,
                    beta_end**0.5,
                    num_timesteps,
                    dtype=torch.float32,
                )
                ** 2
            )

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # required for self.add_noise
        self.sqrt_alphas_cumprod = self.alphas_cumprod**0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        # required for reconstruct_x0
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(1 / self.alphas_cumprod - 1)

        # required for q_posterior
        self.posterior_mean_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def reconstruct_x0(self, x_t, t, noise):
        s1 = self.sqrt_inv_alphas_cumprod[t]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        s1 = self.posterior_mean_coef1[t]
        s2 = self.posterior_mean_coef2[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):
        if t == 0:
            return 0

        variance = (
            self.betas[t]
            * (1.0 - self.alphas_cumprod_prev[t])
            / (1.0 - self.alphas_cumprod[t])
        )
        variance = variance.clip(1e-20)
        return variance

    def step(self, model_output, timestep, sample):
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps):
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]

        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)

        return s1 * x_start + s2 * x_noise

    def __len__(self):
        return self.num_timesteps


def sample_trajectories(model, trajectory_count=1000, trajectory_lengths=50, device='cuda'):
    noise_scheduler = NoiseScheduler(num_timesteps=trajectory_lengths)
    points = torch.randn(trajectory_count, 2)
    timesteps = list(range(trajectory_lengths))[::-1]

    # SAMPLING
    points_at_timestep_n = []  # [[P1, P2, P3], [P1, P2, P3]]
    actions_at_timestep_n = []  # [[A1, A2, A3], [A1, A2, A3]]
    print("creating trajectories...")
    for t in tqdm(timesteps):
        t = torch.from_numpy(np.repeat(t, trajectory_count)).long()[:, None]
        s = torch.concat((points, t), dim=1)
        with torch.no_grad():
            residual = model(s.to(device)).sample()
        points = noise_scheduler.step(residual.cpu(), t[0].cpu(), points)
        points_at_timestep_n.append(s.cpu().numpy())
        actions_at_timestep_n.append(residual.cpu().numpy())

    trajectories = []
    for i in range(trajectory_count):
        obs, next_obs, actions, rews = [], [], [], []
        
        for k in range(len(timesteps)-1):
            state = points_at_timestep_n[k][i]
            next_state = points_at_timestep_n[k+1][i]
            action = actions_at_timestep_n[k][i]

            obs.append(state)
            next_obs.append(next_state)
            actions.append(action)
            rews.append(
                get_reward(next_state, end_timestep=0)
            )

        trajectories.append(
            {
                "observation": np.array(obs, dtype=np.float32),
                "next_observation": np.array(next_obs, dtype=np.float32),
                "action": np.array(actions, dtype=np.float32),
                "reward": np.array(rews, dtype=np.float32),
            }
        )

    return trajectories


def goodness(point, good_points, bad_points):
    import math

    closest_good_distance = float("inf")
    for good_point in good_points:
        distance = math.sqrt(
            (point[0] - good_point[0]) ** 2 + (point[1] - good_point[1]) ** 2
        )
        if distance < closest_good_distance:
            closest_good_distance = distance

    closest_bad_distance = float("inf")
    for bad_point in bad_points:
        distance = math.sqrt(
            (point[0] - bad_point[0]) ** 2 + (point[1] - bad_point[1]) ** 2
        )
        if distance < closest_bad_distance:
            closest_bad_distance = distance

    return closest_bad_distance - closest_good_distance


def eyes_dataset(n=800):
    rng = np.random.default_rng(42)

    x, y = [], []

    # Eyes: adding points for two small circles
    for eye_x in [-0.5, 0.5]:  # x-coordinates for left and right eyes
        eye_y = 0.5  # y-coordinate (same for both eyes)
        eye_radius = 0.2
        t = (
            2 * np.pi * rng.uniform(0, 1, n // 20)
        )  # divide by 20 to have fewer points for eyes
        eye_points_x = eye_x + eye_radius * np.cos(t)
        eye_points_y = eye_y + eye_radius * np.sin(t)
        x = np.concatenate([x, eye_points_x])
        y = np.concatenate([y, eye_points_y])

    X = np.stack((x, y), axis=1)
    X *= 3

    return X.astype(np.float32)


def mouth_dataset(n=800):
    rng = np.random.default_rng(42)

    # Generate circle points
    x = np.round(rng.uniform(-0.5, 0.5, n) / 2, 1) * 2
    y = np.round(rng.uniform(-0.5, 0.5, n) / 2, 1) * 2
    norm = np.sqrt(x**2 + y**2) + 1e-10
    x /= norm
    y /= norm

    # Add noise
    theta = 2 * np.pi * rng.uniform(0, 1, n)
    r = rng.uniform(0, 0.03, n)
    x += r * np.cos(theta)
    y += r * np.sin(theta)

    # Mouth: adding points for a semi-circle
    mouth_radius = 0.5
    t = np.pi * rng.uniform(
        0, 1, n // 10
    )  # divide by 10 to have fewer points for mouth
    mouth_points_x = mouth_radius * np.cos(t)
    mouth_points_y = -(
        0.5 + mouth_radius * np.sin(t) - mouth_radius
    )  # adjust y to position the mouth correctly
    x = mouth_points_x
    y = mouth_points_y

    X = np.stack((x, y), axis=1)
    X *= 3

    return X.astype(np.float32)


def get_reward(state, end_timestep):
    # compute distance to prefered points and disliked points, return average
    position, timestep = state[:2], state[2]    

    good_points = mouth_dataset(n=50)
    bad_points = eyes_dataset(n=50)

    # if timestep.item() == end_timestep:
    return goodness(position, good_points, bad_points)

    # return 0
    # return 1 / (position[1] + 1e-6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default="base")
    parser.add_argument(
        "--dataset",
        type=str,
        default="dino",
        choices=["circle", "dino", "line", "moons"],
    )
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument(
        "--beta_schedule", type=str, default="linear", choices=["linear", "quadratic"]
    )
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument(
        "--time_embedding",
        type=str,
        default="sinusoidal",
        choices=["sinusoidal", "learnable", "linear", "zero"],
    )
    parser.add_argument(
        "--input_embedding",
        type=str,
        default="sinusoidal",
        choices=["sinusoidal", "learnable", "linear", "identity"],
    )
    parser.add_argument("--save_images_step", type=int, default=5)

    parser.add_argument("--n_iter", "-n", type=int, default=200)
    parser.add_argument("--use_reward_to_go", "-rtg", action="store_true")
    parser.add_argument("--use_baseline", action="store_true")
    parser.add_argument("--baseline_learning_rate", "-blr", type=float, default=5e-3)
    parser.add_argument("--baseline_gradient_steps", "-bgs", type=int, default=5)
    parser.add_argument("--gae_lambda", type=float, default=None)
    parser.add_argument("--normalize_advantages", "-na", action="store_true")
    parser.add_argument(
        "--batch_size", "-b", type=int, default=1000
    )  # steps collected per train iteration
    parser.add_argument(
        "--eval_batch_size", "-eb", type=int, default=400
    )  # steps collected per eval iteration
    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--n_layers", "-l", type=int, default=2)
    parser.add_argument("--layer_size", "-s", type=int, default=64)
    parser.add_argument(
        "--ep_len", type=int
    )  # students shouldn't change this away from env's default
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--video_log_freq", type=int, default=-1)
    parser.add_argument("--scalar_log_freq", type=int, default=1)
    parser.add_argument("--action_noise_std", type=float, default=0)
    parser.add_argument("--device", type=str, default='cpu', help='')
    config = parser.parse_args()
    device = config.device
    # dataset = datasets.get_dataset(config.dataset)
    # dataloader = DataLoader(
    #    dataset, batch_size=config.train_batch_size, shuffle=True, drop_last=True)

    # model = MLP(
    #     hidden_size=config.hidden_size,
    #     hidden_layers=config.hidden_layers,
    #     emb_size=config.embedding_size,
    #     time_emb=config.time_embedding,
    #     input_emb=config.input_embedding,
    # )

    model = DiffusionActor(
        hidden_size=128,
        hidden_layers=3,
        emb_size=128,
        learning_rate=1e-4,
        time_emb="sinusoidal",
        input_emb="sinusoidal",
    )
    model = model.to(device)
    if config.load_model:
        model.load_state_dict(torch.load(config.load_model))

    ob_dim, ac_dim = 3, 2
    agent = PGAgent(
        pretrained=model,
        # learning_rate=config.learning_rate,
        # use_baseline=config.use_baseline,
        # use_reward_to_go=config.use_reward_to_go,
        # baseline_learning_rate=config.baseline_learning_rate,
        # baseline_gradient_steps=config.baseline_gradient_steps,
        # gae_lambda=config.gae_lambda,
        # normalize_advantages=config.normalize_advantages,
    )

    noise_scheduler = NoiseScheduler(
        num_timesteps=config.num_timesteps, beta_schedule=config.beta_schedule
    )

    """
    for epoch in n_epochs:
        1. Sample many trajectories, based on the policy
            forward()
        2. Update agent based on sampled trajectories
            pg_agent.update()

    
    wrapper for actor:
    class actor():
        def forward()
            mean = model(sample, t)
            return torch.gaussian(mean, variance)
    
    """
    global_step = 0
    frames = []
    losses = []
    print("Finetunning model...")
    
    outdir = f"exps/{config.experiment_name}"
    os.makedirs(outdir, exist_ok=True)
    
    for epoch in range(config.num_epochs):
        # model.train()
        # progress_bar = tqdm(total=len(dataloader))
        # progress_bar.set_description(f"Epoch {epoch}")
        agent.actor.eval()
        trajs = sample_trajectories(agent.actor, trajectory_count=200, trajectory_lengths=config.num_timesteps, device=device)

        trajs_dict = {k: torch.Tensor([traj[k] for traj in trajs]).to(device) for k in trajs[0]}
        agent.actor.train()
        loss = agent.update(
            trajs_dict["observation"],
            trajs_dict["action"],
            trajs_dict["reward"],
        )
        print(f"{epoch} / {config.num_epochs} loss: {loss}")

        # agent.update(trajs_dict)

        # logs = {"loss": loss.detach().item(), "step": global_step}
        # losses.append(loss.detach().item())

        if epoch % config.save_images_step == 0 or epoch == config.num_epochs - 1:
            xmin, xmax = -6, 6
            ymin, ymax = -6, 6
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            for t in trajs[:100]:
                # plt.plot(t["observation"][:, 0], t["observation"][:, 1])
                plt.scatter(t["observation"][:, 0][-1], t["observation"][:, 1][-1])
            plt.savefig(f"{outdir}/epoch={epoch}.png")
            plt.close()

            agent.actor.eval()
            sample = torch.randn(config.eval_batch_size, 2).to(device)
            timesteps = list(range(len(noise_scheduler)))[::-1]
            for i, t in enumerate(tqdm(timesteps)):
                t = torch.from_numpy(np.repeat(t, config.eval_batch_size)).long().to(device)
                with torch.no_grad():
                    residual = agent.actor(torch.cat((sample, t.unsqueeze(-1)), dim=1)).sample()
                sample = noise_scheduler.step(residual, t[0], sample)
            frames.append(sample.numpy())

    print("Saving model...")
    torch.save(model.state_dict(), f"{outdir}/model.pth")

    print("Saving images...")
    imgdir = f"{outdir}/images"
    os.makedirs(imgdir, exist_ok=True)
    frames = np.stack(frames)
    xmin, xmax = -6, 6
    ymin, ymax = -6, 6
    for i, frame in enumerate(frames):
        plt.figure(figsize=(10, 10))
        plt.scatter(frame[:, 0], frame[:, 1])
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.savefig(f"{imgdir}/{i:04}.png")
        plt.close()

    print("Saving loss as numpy array...")
    np.save(f"{outdir}/loss.npy", np.array(losses))

    print("Saving frames...")
    np.save(f"{outdir}/frames.npy", frames)
