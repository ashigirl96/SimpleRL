# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

from ..network import *
from ..component import *
from .BaseAgent import *
import torchvision


class TD3Agent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()  # type: TwinDelayDeterministicActorCriticNet
        self.target_network = config.network_fn()  # type: TwinDelayDeterministicActorCriticNet
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn()
        self.total_steps = 0
        self.state = None
        self.min_action = min(self.task.action_space.low)
        self.max_action = max(self.task.action_space.high)

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                               param * self.config.target_network_mix)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        action = self.network(state)
        self.config.state_normalizer.unset_read_only()
        return to_np(action)

    def step(self):
        config = self.config
        if self.state is None:
            self.random_process.reset_states()
            self.state = self.task.reset()
            self.state = config.state_normalizer(self.state)

        if self.total_steps < config.warm_up:
            action = [self.task.action_space.sample()]
        else:
            action = self.network(self.state)
            action = to_np(action)
            action += self.random_process.sample()
        action = np.clip(action, self.task.action_space.low, self.task.action_space.high)
        next_state, reward, done, info = self.task.step(action)
        next_state = self.config.state_normalizer(next_state)
        self.record_online_return(info)
        reward = self.config.reward_normalizer(reward)

        experiences = list(zip(self.state, action, reward, next_state, done))
        self.replay.feed_batch(experiences)
        if done[0]:
            self.random_process.reset_states()
        self.state = next_state
        self.total_steps += 1

        if self.replay.size() >= config.warm_up:
            experiences = self.replay.sample()
            states, actions_, rewards, next_states, terminals = experiences
            states = tensor(states)
            actions = tensor(actions_)
            rewards = tensor(rewards).unsqueeze(-1)
            next_states = tensor(next_states)
            mask = tensor(1 - terminals).unsqueeze(-1)

            phi_next = self.target_network.feature(next_states)
            # For target policy smoothing, the added noise is clipped to the range of possible actions
            action_noise = tensor(actions_).normal_(0, config.policy_noise)
            action_noise.clamp_(-config.noise_clip, config.noise_clip)
            a_next = self.target_network.actor(phi_next).add(action_noise)
            a_next.clamp_(self.min_action, self.max_action)
            a_next.detach_()
            q_next1 = self.target_network.critic1(phi_next, a_next)
            q_next2 = self.target_network.critic2(phi_next, a_next)
            q_next = torch.min(q_next1, q_next2)
            q_next = config.discount * mask * q_next
            q_next.add_(rewards)
            q_next = q_next.detach()
            phi = self.network.feature(states)
            q1 = self.network.critic1(phi, actions)
            q2 = self.network.critic2(phi, actions)
            critic_loss1 = (q1 - q_next).pow(2).mul(0.5).sum(-1).mean()
            critic_loss2 = (q2 - q_next).pow(2).mul(0.5).sum(-1).mean()
            critic_loss = critic_loss1 + critic_loss2

            self.network.zero_grad()
            critic_loss.backward()
            self.network.critic_opt.step()

            # Delayed policy updates
            if self.total_steps % config.policy_update_frequency == 0:
                phi = self.network.feature(states)
                action = self.network.actor(phi)
                policy_loss = -self.network.critic1(phi.detach(), action).mean()

                self.network.zero_grad()
                policy_loss.backward()
                self.network.actor_opt.step()

                self.soft_update(self.target_network, self.network)
