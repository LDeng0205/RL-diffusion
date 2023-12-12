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
from ddpm import *
from positional_embeddings import PositionalEmbedding

from reward import get_reward

from policy_gradients import DiffusionActor, PGAgent

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default="base")
    parser.add_argument("--dataset", type=str, default="dino", choices=["circle", "dino", "line", "moons"],)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "quadratic"])
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--time_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "zero"],)
    parser.add_argument("--input_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "identity"],)
    parser.add_argument("--save_images_step", type=int, default=5)
    parser.add_argument("--use_baseline", action='store_true', default=False, help='')

    parser.add_argument("--batch_size", "-b", type=int, default=1000)  # steps collected per train iteration
    parser.add_argument("--eval_batch_size", "-eb", type=int, default=400)  # steps collected per eval iteration
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default='cpu', help='')
    config = parser.parse_args()
    device = config.device

    model = DiffusionActor(
        hidden_size=128,
        hidden_layers=3,
        emb_size=128,
        learning_rate=1e-3,
        time_emb="sinusoidal",
        input_emb="sinusoidal",
    )

    model = model.to(device)
    if config.load_model:
        model.load_state_dict(torch.load(config.load_model))

    ob_dim, ac_dim = 3, 2
    agent = PGAgent(
        pretrained=model,
        use_baseline=config.use_baseline
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

        agent.actor.eval()
        trajs = sample_trajectories(agent.actor, trajectory_count=200, trajectory_lengths=config.num_timesteps, device=device)

        trajs_dict = {k: torch.Tensor([traj[k] for traj in trajs]).to(device) for k in trajs[0]}
        agent.actor.train()
        loss = agent.update(
            trajs_dict["observation"],
            trajs_dict["action"],
            trajs_dict["reward"],
        )
        print(f"{epoch} / {config.num_epochs} Actor loss: {loss['Actor Loss']}")
        print(f"{epoch} / {config.num_epochs} Critic loss: {loss['Critic Loss']}")

        if epoch % config.save_images_step == 0 or epoch == config.num_epochs - 1:
            # xmin, xmax = -6, 6
            # ymin, ymax = -6, 6
            # plt.xlim(xmin, xmax)
            # plt.ylim(ymin, ymax)
            # for t in trajs[:100]:
            #     # plt.plot(t["observation"][:, 0], t["observation"][:, 1])
            #     plt.scatter(t["observation"][:, 0][-1], t["observation"][:, 1][-1])
            # plt.savefig(f"{outdir}/epoch={epoch}.png")
            # plt.close()

            agent.actor.eval()
            sample = torch.randn(config.eval_batch_size, 2).to(device)
            timesteps = list(range(len(noise_scheduler)))[::-1]
            for i, t in enumerate(tqdm(timesteps)):
                t = torch.from_numpy(np.repeat(t, config.eval_batch_size)).long().to(device)
                with torch.no_grad():
                    residual = agent.actor(torch.cat((sample, t.unsqueeze(-1)), dim=1)).sample()
                sample = noise_scheduler.step(residual, t[0], sample)
            frames.append(sample.numpy())
            # sample is of shape (eval_batch_szie, 2)
            # call reward function on all samples and sum

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

"""

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
"""