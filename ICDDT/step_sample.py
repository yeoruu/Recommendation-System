import numpy as np
from abc import ABC, abstractmethod
import torch as th
import torch.distributed as dist
from scipy.stats import beta


class ScheduleSampler(ABC):
    @abstractmethod
    def weights(self):
        """
        """

    def sample(self, batch_size, device):
        
        w = self.weights() 
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights






class LossAwareSampler(ScheduleSampler):
    def update_with_local_losses(self, local_ts, local_losses):
        
        batch_sizes = [
            th.tensor([0], dtype=th.int32, device=local_ts.device)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(
            batch_sizes,
            th.tensor([len(local_ts)], dtype=th.int32, device=local_ts.device),
        )

        # Pad all_gather batches to be the maximum batch size.
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        timestep_batches = [th.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [th.zeros(max_bs).to(local_losses) for bs in batch_sizes]
        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)
        timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        Update the reweighting using losses from a model.

        Sub-classes should override this method to update the reweighting
        using losses from the model.

        This method directly updates the reweighting without synchronizing
        between workers. It is called by update_with_local_losses from all
        ranks with identical arguments. Thus, it should have deterministic
        behavior to maintain state across workers.

        :param ts: a list of int timesteps.
        :param losses: a list of float losses, one per timestep.
        """


class LossSecondMomentResampler(LossAwareSampler):
    def __init__(self, num_timesteps, history_per_term=10, uniform_prob=0.001):
        self.num_timesteps = num_timesteps
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [self.num_timesteps, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([self.num_timesteps], dtype=np.int)

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()


class FixSampler(ScheduleSampler):
    def __init__(self, num_timesteps):
        self.num_timesteps = num_timesteps
        ###############################################################
        ### You can custome your own sampling weight of steps here. ###
        ###############################################################
        self._weights = np.concatenate([np.ones([num_timesteps//2]), np.zeros([num_timesteps//2]) + 0.5])

    def weights(self):
        return self._weights

class UniformSampler(ScheduleSampler):
    def __init__(self, num_timesteps):
        self.num_timesteps = num_timesteps
        self._weights = np.ones([self.num_timesteps])
        
    def weights(self):
        return self._weights

class BehaAwareSampler(ScheduleSampler):
    ''' beta distribution with alpha and beta
        beta = 1 
        '''
    def __init__(self, num_timesteps, alpha=1.0, beta=1.0):
        self.num_timesteps = num_timesteps
        self.alpha = alpha 
        self.beta = beta 
        
        # self._weights = np.ones([self.num_timesteps])
        
    def weights(self):
        x_steps = np.linspace(1/self.num_timesteps, 1, self.num_timesteps) # t/T
        weights = beta.pdf(x_steps, self.alpha, self.beta)
        return weights


def create_named_schedule_sampler(name, num_timesteps, alpha, beta):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.
    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """ 
    if name == "uniform":
        return UniformSampler(num_timesteps)
    elif name == "lossaware":
        return LossSecondMomentResampler(num_timesteps)  ## default setting 
    elif name == "fixstep":
        return FixSampler(num_timesteps)
    elif name == "behaaware":
        return BehaAwareSampler(num_timesteps, alpha, beta)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")
