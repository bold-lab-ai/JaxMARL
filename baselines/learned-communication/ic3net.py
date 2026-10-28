"""
CommNet, IC3Net and their IC and IRIC controls, ported from the IC3Net reference
implementation (https://github.com/IC3Net/IC3Net).

As in the reference, every update collects whole episodes and trains with
REINFORCE and a learned value baseline:

- IC and IRIC: independent agents that don't communicate, with a tanh RNN as in
  the reference's commands (the paper describes an LSTM for every method).
- CommNet: agents add the mean of the other agents' hidden states to their
  input before each LSTM step.
- IC3Net: CommNet where each agent also chooses at every step whether to
  communicate at the next one.

IC and CommNet train every agent on the team's mean return, IRIC and IC3Net on
its own (MEAN_RATIO). All agents share parameters, so they must have the same
observation and action spaces.
"""

import os
from typing import Callable, NamedTuple

import distrax
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

import jaxmarl
import wandb

# BASELINE: (communicates, gates communication, default MEAN_RATIO)
BASELINES = {
    "ic": (False, False, 1.0),
    "iric": (False, False, 0.0),
    "commnet": (True, False, 1.0),
    "ic3net": (True, True, 0.0),
}


class CommPolicy(nn.Module):
    """One step of all agents' policies, with the structure of the reference's `RNN`
    and `CommNetMLP`."""

    action_dim: int
    hidden_size: int
    communicate: bool
    gated: bool
    comm_passes: int

    @nn.compact
    def __call__(self, carry, obs, alive, talk):
        # carry: (h, c), each (..., agents, hidden); obs: (..., agents, obs_dim);
        # alive, talk: (..., agents), 1.0 for agents that are alive or talking.
        h, c = carry
        x = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="encoder",
        )(obs)
        if self.communicate:
            # Agents that are dead or silent neither send nor receive messages.
            gate = (alive * talk if self.gated else alive)[..., None]
            num_alive = alive.sum(-1, keepdims=True)[..., None]
            denom = jnp.where(num_alive > 1, num_alive - 1, 1.0)
            lstm = nn.OptimizedLSTMCell(self.hidden_size, name="lstm")
            for i in range(self.comm_passes):
                sent = gate * h
                received = gate * (sent.sum(-2, keepdims=True) - sent) / denom
                message = nn.Dense(
                    self.hidden_size,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                    name=f"comm_{i}",
                )(received)
                (c, h), _ = lstm((c, h), x + message)
        else:
            h = nn.tanh(
                x
                + nn.Dense(
                    self.hidden_size,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                    name="recurrent",
                )(h)
            )

        logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(h)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(h)
        talk_logits = None
        if self.gated:
            talk_logits = nn.Dense(
                2, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="talk"
            )(h)
        return (h, c), logits, talk_logits, value.squeeze(-1)


class Transition(NamedTuple):
    obs: jnp.ndarray  # (envs, agents, obs_dim)
    alive: jnp.ndarray  # agents alive when acting, (envs, agents)
    talk: jnp.ndarray  # agents talking this step, (envs, agents)
    action: jnp.ndarray
    talk_action: jnp.ndarray
    reward: jnp.ndarray
    alive_after: jnp.ndarray  # agents alive after the step
    completed: jnp.ndarray  # agents whose episode ended during the step
    valid: jnp.ndarray  # step belongs to the first episode, (envs,)
    done: jnp.ndarray
    success: jnp.ndarray


def compute_returns(reward, done, completed, gamma, mean_ratio):
    """Each agent's return, mixing the team's mean return and its own.

    reward and completed are (steps, envs, agents) and done (steps, envs). Returns
    end with the episode, or at the last step; an agent's own return also ends
    when its episode does (a car leaving Traffic Junction).
    """
    num_steps = reward.shape[0]
    last = done | (jnp.arange(num_steps) == num_steps - 1)[:, None]
    keep = (~last)[..., None]
    keep_own = keep & ~completed

    def _step(carry, xs):
        team, own = carry
        reward, keep, keep_own = xs
        team = reward + gamma * team * keep
        own = reward + gamma * own * keep_own
        mixed = mean_ratio * team.mean(-1, keepdims=True) + (1 - mean_ratio) * own
        return (team, own), mixed

    zeros = jnp.zeros(reward.shape[1:])
    _, returns = jax.lax.scan(
        _step, (zeros, zeros), (reward, keep, keep_own), reverse=True
    )
    return returns


def make_optimizer(lr):
    """PyTorch's RMSprop, with the reference's alpha and eps."""
    return optax.rmsprop(lr, decay=0.97, eps=1e-6, eps_in_sqrt=False)


class Learner(NamedTuple):
    network: CommPolicy
    init_params: Callable  # rng -> params
    rollout: Callable  # (params, rng, curriculum level) -> Transition
    returns: Callable  # Transition -> returns
    loss: Callable  # (params, Transition, returns) -> (loss, aux)
    curriculum_level: Callable  # update -> curriculum level
    curriculum_values: np.ndarray
    info_keys: tuple
    gated: bool
    always_talk: bool


def make_learner(config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    communicate, gated, default_mean_ratio = BASELINES[config["BASELINE"]]
    mean_ratio = config.get("MEAN_RATIO")
    if mean_ratio is None:
        mean_ratio = default_mean_ratio
    # The reference makes IC3Net agents always talk in Traffic Junction.
    always_talk = gated and config.get("COMM_ACTION_ONE", False)

    agents = env.agents
    num_agents = env.num_agents
    num_envs, num_steps = config["NUM_ENVS"], config["NUM_STEPS"]
    obs_dim = int(np.prod(env.observation_space(agents[0]).shape))
    action_dim = env.action_space(agents[0]).n
    for a in agents:
        assert int(np.prod(env.observation_space(a).shape)) == obs_dim
        assert env.action_space(a).n == action_dim
    num_updates = int(config["TOTAL_TIMESTEPS"] // num_steps // num_envs)
    config["NUM_UPDATES"] = num_updates

    # CURRICULUM raises an environment argument from START to END in steps of
    # STEP, between the fractions BEGIN and FINISH of training, as the reference
    # raises add_rate in Traffic Junction. Each value gets its own environment.
    curriculum = config.get("CURRICULUM")
    if curriculum:
        num_levels = round(
            (curriculum["END"] - curriculum["START"]) / curriculum["STEP"]
        )
        values = curriculum["START"] + curriculum["STEP"] * np.arange(num_levels + 1)
        envs = [
            jaxmarl.make(
                config["ENV_NAME"],
                **{**config["ENV_KWARGS"], curriculum["KEY"]: float(v)},
            )
            for v in values
        ]
    else:
        values, envs = np.zeros(1), [env]
    env_steps = [jax.vmap(e.step) for e in envs]

    def curriculum_level(update):
        if not curriculum:
            return 0
        progress = (update / num_updates - curriculum["BEGIN"]) / (
            curriculum["FINISH"] - curriculum["BEGIN"]
        )
        return jnp.floor(jnp.clip(progress, 0, 1) * (len(envs) - 1)).astype(int)

    network = CommPolicy(
        action_dim=action_dim,
        hidden_size=config["HIDDEN_SIZE"],
        communicate=communicate,
        gated=gated,
        comm_passes=config["COMM_PASSES"],
    )

    def init_carry(batch_shape):
        shape = (*batch_shape, num_agents, config["HIDDEN_SIZE"])
        return jnp.zeros(shape), jnp.zeros(shape)

    def init_params(rng):
        return network.init(
            rng,
            init_carry((1,)),
            jnp.zeros((1, num_agents, obs_dim)),
            jnp.ones((1, num_agents)),
            jnp.ones((1, num_agents)),
        )

    def batchify(x):
        return jnp.stack([x[a].reshape(num_envs, -1) for a in agents], axis=1)

    # Which per-agent statistics the environment reports in its info.
    def _info(key):
        _, state = env.reset(key)
        actions = {a: env.action_space(a).sample(key) for a in agents}
        return env.step(key, state, actions)[-1]

    info_keys = tuple(jax.eval_shape(_info, jax.random.PRNGKey(0)).keys())

    def rollout(params, rng, level):
        """Run every environment from reset for NUM_STEPS steps."""
        rng, _rng = jax.random.split(rng)
        obs, env_state = jax.vmap(env.reset)(jax.random.split(_rng, num_envs))

        def _env_step(carry, _):
            hidden, env_state, obs, alive, talk, valid, rng = carry
            obs_batch = batchify(obs)
            hidden, logits, talk_logits, _ = network.apply(
                params, hidden, obs_batch, alive, talk
            )
            rng, key_act, key_talk, key_step = jax.random.split(rng, 4)
            action = distrax.Categorical(logits=logits).sample(seed=key_act)
            talk_action = jnp.zeros_like(action)
            if gated:
                talk_action = distrax.Categorical(logits=talk_logits).sample(
                    seed=key_talk
                )
            obs, env_state, reward, done, info = jax.lax.switch(
                level,
                env_steps,
                jax.random.split(key_step, num_envs),
                env_state,
                {a: action[:, i] for i, a in enumerate(agents)},
            )
            ones = jnp.ones((num_envs, num_agents))
            none = jnp.zeros((num_envs, num_agents), dtype=bool)
            alive_after = info["alive"] if "alive" in info_keys else ones
            alive_after = alive_after.astype(jnp.float32)
            transition = Transition(
                obs=obs_batch,
                alive=alive,
                talk=talk,
                action=action,
                talk_action=talk_action,
                reward=jnp.stack([reward[a] for a in agents], axis=1),
                alive_after=alive_after,
                completed=info["completed"] if "completed" in info_keys else none,
                valid=valid,
                done=done["__all__"],
                success=info["success"][:, 0] if "success" in info_keys else valid,
            )
            talk = ones if always_talk else talk_action.astype(jnp.float32)
            valid = valid & ~done["__all__"]
            carry = (hidden, env_state, obs, alive_after, talk, valid, rng)
            return carry, transition

        carry = (
            init_carry((num_envs,)),
            env_state,
            obs,
            # As in the reference, every agent counts as alive and none talks
            # at the first step.
            jnp.ones((num_envs, num_agents)),
            jnp.zeros((num_envs, num_agents)),
            jnp.ones(num_envs, dtype=bool),
            rng,
        )
        _, traj = jax.lax.scan(_env_step, carry, None, num_steps)
        return traj

    def returns(traj):
        return compute_returns(
            traj.reward, traj.done, traj.completed, config["GAMMA"], mean_ratio
        )

    def loss(params, traj, returns):
        def _step(hidden, xs):
            t, obs, alive, talk = xs
            hidden, logits, talk_logits, value = network.apply(
                params, hidden, obs, alive, talk
            )
            # Truncated backpropagation through time, as the reference.
            cut = (t + 1) % config["DETACH_GAP"] == 0
            hidden = jax.tree.map(
                lambda x: jnp.where(cut, jax.lax.stop_gradient(x), x), hidden
            )
            return hidden, (logits, talk_logits, value)

        _, (logits, talk_logits, value) = jax.lax.scan(
            _step,
            init_carry((num_envs,)),
            (jnp.arange(num_steps), traj.obs, traj.alive, traj.talk),
        )
        pi = distrax.Categorical(logits=logits)
        log_prob = pi.log_prob(traj.action)
        entropy = pi.entropy()
        if gated:
            talk_pi = distrax.Categorical(logits=talk_logits)
            log_prob = log_prob + talk_pi.log_prob(traj.talk_action)
            entropy = entropy + talk_pi.entropy()

        # The reference masks each step with the agents alive after it, and
        # averages over environment steps.
        mask = traj.valid[..., None] * traj.alive_after
        num_env_steps = traj.valid.sum()
        advantage = returns - jax.lax.stop_gradient(value)
        actor_loss = -(advantage * log_prob * mask).sum() / num_env_steps
        value_loss = ((value - returns) ** 2 * mask).sum() / num_env_steps
        entropy = (entropy * mask).sum() / num_env_steps
        loss = (
            actor_loss
            + config["VALUE_COEF"] * value_loss
            - config["ENT_COEF"] * entropy
        )
        return loss, (actor_loss, value_loss, entropy)

    return Learner(
        network=network,
        init_params=init_params,
        rollout=rollout,
        returns=returns,
        loss=loss,
        curriculum_level=curriculum_level,
        curriculum_values=values,
        info_keys=info_keys,
        gated=gated,
        always_talk=always_talk,
    )


def make_train(config):
    learner = make_learner(config)
    num_envs, num_steps = config["NUM_ENVS"], config["NUM_STEPS"]
    curriculum = config.get("CURRICULUM")

    def train(rng):
        original_seed = rng[0]
        rng, _rng = jax.random.split(rng)
        params = learner.init_params(_rng)
        tx = make_optimizer(config["LR"])
        train_state = TrainState.create(
            apply_fn=learner.network.apply, params=params, tx=tx
        )

        def _update_step(runner_state, update):
            train_state, rng = runner_state
            rng, _rng = jax.random.split(rng)
            level = learner.curriculum_level(update)
            traj = learner.rollout(train_state.params, _rng, level)
            returns = learner.returns(traj)
            (loss, (actor_loss, value_loss, entropy)), grads = jax.value_and_grad(
                learner.loss, has_aux=True
            )(train_state.params, traj, returns)
            train_state = train_state.apply_gradients(grads=grads)

            # Statistics of the training episodes, as the reference reports.
            valid = traj.valid.astype(jnp.float32)
            length = valid.sum(0)
            last = jnp.maximum(length.astype(jnp.int32) - 1, 0)
            metrics = {
                "env_step": (update + 1) * num_envs * num_steps,
                "returned_episode_returns": (traj.reward * valid[..., None])
                .sum(0)
                .mean(),
                "returned_episode_lengths": length.mean(),
                "loss": loss,
                "actor_loss": actor_loss,
                "value_loss": value_loss,
                "entropy": entropy,
            }
            if curriculum:
                metrics[curriculum["KEY"]] = jnp.asarray(learner.curriculum_values)[
                    level
                ]
            if "success" in learner.info_keys:
                metrics["success"] = traj.success[last, jnp.arange(num_envs)].mean()
            if learner.gated and not learner.always_talk:
                metrics["comm_rate"] = (traj.talk_action * valid[..., None]).sum() / (
                    length.sum() * traj.talk_action.shape[-1]
                )

            if config["WANDB_MODE"] != "disabled":

                def callback(metrics, original_seed):
                    if config.get("WANDB_LOG_ALL_SEEDS", False):
                        metrics.update(
                            {
                                f"rng{int(original_seed)}/{k}": v
                                for k, v in metrics.items()
                            }
                        )
                    wandb.log(metrics)

                jax.debug.callback(callback, metrics, original_seed)

            return (train_state, rng), metrics

        runner_state, metrics = jax.lax.scan(
            _update_step, (train_state, rng), jnp.arange(config["NUM_UPDATES"])
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    config = {**config, **config["alg"]}
    alg_name = config["BASELINE"]
    env_name = config["ENV_NAME"]
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[alg_name.upper(), env_name.upper(), f"jax_{jax.__version__}"],
        name=f"{alg_name}_{env_name}",
        config=config,
        mode=config["WANDB_MODE"],
    )
    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config)))
    # SEED_CHUNK trains the seeds that many at a time, to fit in memory; the
    # seeds are the same as with all of them at once.
    chunk = config.get("SEED_CHUNK") or config["NUM_SEEDS"]
    chunks = [
        jax.block_until_ready(train_vjit(rngs[i : i + chunk]))
        for i in range(0, config["NUM_SEEDS"], chunk)
    ]
    outs = jax.tree.map(lambda *x: jnp.concatenate(x), *chunks)

    final = jax.tree.map(lambda x: x[:, -1], outs["metrics"])
    for k in ["returned_episode_returns", "returned_episode_lengths", "success"]:
        if k in final:
            print(f"{k}: {final[k].mean():.3f} +- {final[k].std():.3f}")

    if config.get("SAVE_PATH") is not None:
        from jaxmarl.wrappers.baselines import save_params

        save_dir = os.path.join(config["SAVE_PATH"], alg_name, env_name)
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(
            config,
            os.path.join(
                save_dir, f"{alg_name}_{env_name}_seed{config['SEED']}_config.yaml"
            ),
        )
        params = outs["runner_state"][0].params
        for i, rng in enumerate(rngs):
            save_params(
                jax.tree.map(lambda x: x[i], params),
                os.path.join(
                    save_dir,
                    f"{alg_name}_{env_name}_seed{config['SEED']}_vmap{i}.safetensors",
                ),
            )
    wandb.finish()


if __name__ == "__main__":
    main()
