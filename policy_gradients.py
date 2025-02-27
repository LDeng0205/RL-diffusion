from typing import Optional, Sequence
import numpy as np
import torch
import copy

from torch import nn

from ddpm import MLP


class DiffusionActor(MLP):
    def __init__(
        self,
        variance: float = 0.05,
        learning_rate: float = 0.001,
        hidden_size: int = 128,
        hidden_layers: int = 3,
        emb_size: int = 128,
        time_emb: str = "sinusoidal",
        input_emb: str = "sinusoidal",
    ):
        super().__init__(hidden_size, hidden_layers, emb_size, time_emb, input_emb)

        self.variance = variance
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            learning_rate,
        )

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.95)

    def forward(self, obs):
        x, t = obs[..., :2], obs[..., 2]
        mean = super().forward(x, t)
        return torch.distributions.MultivariateNormal(
            mean,
            scale_tril=self.variance
            * torch.eye(mean.shape[1]).repeat(mean.shape[0], 1, 1).to(mean.device),
        )

    def update(self, obs, actions, advantages):
        self.optimizer.zero_grad()
        log_p = self.forward(obs).log_prob(actions)
        loss = -torch.sum(torch.mul(log_p, advantages))
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()
    
class Critic(MLP):
    def __init__(
        self,
        learning_rate: float = 0.001,
        hidden_size: int = 128,
        hidden_layers: int = 3,
        emb_size: int = 128,
        time_emb: str = "sinusoidal",
        input_emb: str = "sinusoidal",
    ):
        super().__init__(hidden_size, hidden_layers, emb_size, time_emb, input_emb)

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            learning_rate,
        )
        
        self.mse = nn.MSELoss()

    def forward(self, obs):
        x, t = obs[..., :2], obs[..., 2]
        out = torch.sum(super().forward(x, t), dim=1)
        return out

    def update(self, obs, q_values):
        # q_values = (q_values - q_values.mean())/(q_values.std())

        self.optimizer.zero_grad()
        pred = self.forward(obs)
        loss = self.mse(q_values, pred)
        loss.backward()
        self.optimizer.step()

        return loss.item()

class PGAgent(nn.Module):
    def __init__(
        self,
        pretrained,
        gamma=0.999,
        use_baseline=False,
        use_reward_to_go=False,
        baseline_learning_rate=0.01,
        baseline_gradient_steps=20,
        gae_lambda=None,
        advantage_normalization=True,
        device='cpu',
    ):
        super().__init__()

        # create the actor (policy) network
        self.actor = pretrained
        
        # create the critic (baseline) network, if needed
        if use_baseline:
            self.critic = Critic(learning_rate=baseline_learning_rate).to(device)
            self.baseline_gradient_steps = baseline_gradient_steps
        else:
            self.critic = None

        # other agent parameters
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.advantage_normalization = advantage_normalization

    def _calculate_q_vals(self, rewards):
        discounted_rtg = []
        for traj_rewards in rewards:
            traj_discounted_rtg = []
            for t in range(len(traj_rewards)):
                traj_discounted_rtg.append(
                    sum(
                        [
                            self.gamma ** (i - t) * traj_rewards[i].item()
                            for i in range(t, len(traj_rewards))
                        ]
                    )
                )
            discounted_rtg.append(traj_discounted_rtg)

        return np.asarray(discounted_rtg)

    def _estimate_advantage(self, obs, rewards, q_values):
        # If no baseline (value function), just return q_values
        # TODO: implement reward
        if self.critic is None:
            advantages = copy.deepcopy(q_values)
        else:
            self.critic.eval()
            with torch.no_grad():
                values = self.critic(obs.reshape(-1, obs.size(-1)))
                advantages = q_values - values.reshape(q_values.shape).cpu().numpy()
        
        if self.advantage_normalization:
            advantages = (advantages - advantages.mean())/(advantages.std())

        return advantages

    def update(self, obs, actions, rewards):
        # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values = self._calculate_q_vals(rewards)

        # step 2: calculate advantages from Q values
        advantages: np.ndarray = self._estimate_advantage(obs, rewards, q_values)
        advantages = torch.tensor(advantages).to(rewards.device)
        # step 3: use all datapoints (s_t, a_t, adv_t) to update the PG actor/policy
        obs = obs.reshape(-1, obs.size(-1))
        actions = actions.reshape(-1, actions.size(-1))
        advantages = advantages.flatten()
        self.actor.train()
        actor_loss = self.actor.update(obs, actions, advantages)

        # step 4: if needed, use all datapoints (s_t, a_t, q_t) to update the PG critic/baseline
        baseline_epoch_loss = None
        if self.critic is not None:
            self.critic.train()
            baseline_epoch_loss = 0.0
            for _ in range(self.baseline_gradient_steps):
                baseline_epoch_loss += self.critic.update(obs, torch.tensor(q_values).to(rewards.device).flatten().float())  # TODO: log the update information
            self.critic.eval()
            baseline_epoch_loss /= self.baseline_gradient_steps
        return {
            "Actor Loss": actor_loss,
            "Critic Loss": baseline_epoch_loss,
        }


"""

eval_batch_size = 1000
num_timesteps = 50
plot_step = 5
noise_scheduler = ddpm.NoiseScheduler(num_timesteps=num_timesteps)
sample = torch.randn(eval_batch_size, 2)
timesteps = list(range(num_timesteps))[::-1]
samples = []
steps = []
for i, t in enumerate(tqdm(timesteps)):
    t = torch.from_numpy(np.repeat(t, eval_batch_size)).long()
    with torch.no_grad():
        residual = model(sample, t)
    sample = noise_scheduler.step(residual, t[0], sample)
    if (i + 1) % plot_step == 0:
        samples.append(sample.numpy())
        steps.append(i + 1)
    

class Env:
  # as used now
  observation_space = None
  action_space = None

  # instead we can just:
  ob_dim = (3) # position x,y
  ac_dim = (2) # noise vector x,y

  def __init__(self):
    self.pos = None
    self.timestepsLeft = None
    pass
  
  def reset(self):
    self.pos = randomPos()
    self.timestepsLeft = 5
    return self.pos

  def step(self, action):
    self.pos = self.pos + action

    self.timestepsLeft -= 1
    done = self.timestepsLeft == 0

    rew = 0
    if done:
      rew = -distanceToGoal(self.pos)

    return (self.pos[0], self.pos[1], 5 - self.timestepsLeft), rew, done, _

env = Env()

"""
