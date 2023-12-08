



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
