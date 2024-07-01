from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .utils import masked_mean


class ContrastiveGoalLoss(nn.Module):
    """
    Contrastive Loss for Goal Model
    """

    def __init__(self, unsim_samples: int = 16, sim_samples: int = 16, beta: float = 0.5, strategy: str = 'cosine', source_state_percentile = None, goal_state_percentile = None) -> None:
        """Init.
        
        Args:
            unsim_samples (int, optional): Number of unsimilar samples. Defaults to 16.
            beta (float, optional): Weight for the loss. Defaults to 0.5.
        """
        super().__init__()
        self.unsim_samples = unsim_samples
        self.sim_samples = sim_samples
        self.beta = beta
        self.strategy = strategy
        self.source_state_percentile = source_state_percentile
        self.goal_state_percentile = goal_state_percentile

    def forward(
        self,
        chosen_lh: torch.Tensor,
        rejected_lh: torch.Tensor,
        chosen_na: torch.Tensor,
        rejected_na: torch.Tensor,
    ) -> torch.Tensor:
        """Forward.

        Args:
            chosen_lh (torch.Tensor): Chosen latent hidden states.
            rejected_lh (torch.Tensor): Rejected latent hidden states.
            chosen_na (torch.Tensor): Chosen number actions.
            rejected_na (torch.Tensor): Rejected number actions.;
        Returns:
            torch.Tensor: Loss.
        """

        # similarity
        random_values = torch.rand(chosen_lh.size(0), 2, self.sim_samples, device=chosen_na.device)
        random_values, _ = random_values.sort(dim=1)

        if self.source_state_percentile is not None:
            source_state_percentile = (self.source_state_percentile - 1) * random_values[:, 0] + 1
        if self.goal_state_percentile is not None:
            goal_state_percentile = (self.goal_state_percentile - 1) * random_values[:, 1] + 1

        source_state_sampled_values = (
            ((source_state_percentile * chosen_na.unsqueeze(1)) + 1).floor().int()
        )  # (batch_size, sim_samples)
        source_state_sampled_values = source_state_sampled_values.clamp(0, chosen_lh.size(1) - 1)
        source_state_sampled_values *= -1

        goal_state_sampled_values = (
            ((goal_state_percentile * chosen_na.unsqueeze(1)) + 1).floor().int()
        )
        goal_state_sampled_values = goal_state_sampled_values.clamp(0, chosen_lh.size(1) - 1)
        goal_state_sampled_values *= -1

        source_state_embeddings = chosen_lh[
            torch.arange(chosen_lh.size(0)).unsqueeze(1), source_state_sampled_values
        ] # (batch_size, sim_samples, hidden_size)
        goal_state_embeddings = chosen_lh[
            torch.arange(chosen_lh.size(0)).unsqueeze(1), goal_state_sampled_values
        ] # (batch_size, sim_samples, hidden_size)

        if self.strategy == 'cosine_avg':
            goal_state_embeddings = goal_state_embeddings.mean(dim=1, keepdim=True)

        chosen_similarity = torch.cosine_similarity(
            source_state_embeddings.unsqueeze(2),
            goal_state_embeddings.unsqueeze(1),
            dim=3,
        )  # (batch_size,)

        chosen_sigmod = torch.sigmoid(chosen_similarity).mean()

        # unsimilarity
        random_unsim_values = torch.rand(
            chosen_lh.size(0), self.unsim_samples, device=chosen_na.device
        )

        if self.source_state_percentile is not None:
            random_unsim_values = (self.source_state_percentile - 1) * random_unsim_values + 1

        sampled_unsim_values = (
            ((random_unsim_values * chosen_na.unsqueeze(1)) + 1).floor().int()
        )  # (batch_size, unsim_samples)
        sampled_unsim_values = sampled_unsim_values.clamp(0, chosen_lh.size(1) - 1)
        sampled_unsim_values *= -1
        sampled_unsim_embeddings = chosen_lh[
            torch.arange(chosen_lh.size(0)).unsqueeze(1), sampled_unsim_values
        ]  # (batch_size, unsim_samples, hidden_size)

        random_rejected_values = torch.rand(
            rejected_lh.size(0), self.unsim_samples, device=rejected_na.device
        )

        if self.goal_state_percentile is not None:
            random_rejected_values = (self.goal_state_percentile - 1) * random_rejected_values + 1

        sampled_unsim_rejected_values = (
            ((random_rejected_values * rejected_na.unsqueeze(1)) + 1).floor().int()
        )  # (batch_size, unsim_samples)
        sampled_unsim_rejected_values = sampled_unsim_rejected_values.clamp(
            0, rejected_lh.size(1) - 1
        )
        sampled_unsim_rejected_values *= -1
        sampled_unsim_rejected_embeddings = rejected_lh[
            torch.arange(rejected_lh.size(0)).unsqueeze(1), sampled_unsim_rejected_values
        ]  # (batch_size, unsim_samples, hidden_size)

        if self.strategy == 'cosine_avg':
            sampled_unsim_rejected_embeddings = sampled_unsim_rejected_embeddings.mean(dim=1, keepdim=True)

        unsimilarity = torch.cosine_similarity(
            sampled_unsim_embeddings.unsqueeze(2),
            sampled_unsim_rejected_embeddings.unsqueeze(1),
            dim=3,
        ) # (batch_size,)

        rejected_sigmoid = torch.sigmoid(unsimilarity).mean()

        # final loss
        loss = -torch.log(chosen_sigmod) - torch.log(1 - rejected_sigmoid)
        loss = self.beta * loss

        return loss


class GPTLMLoss(nn.Module):
    """
    GPT Language Model Loss
    """

    def __init__(self):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


class PolicyLoss(nn.Module):
    """
    Policy Loss for PPO
    """

    def __init__(self, clip_eps: float = 0.2) -> None:
        super().__init__()
        self.clip_eps = clip_eps

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -torch.min(surr1, surr2)
        loss = masked_mean(loss, action_mask, dim=-1).mean()
        return loss


class ValueLoss(nn.Module):
    """
    Value Loss for PPO
    """

    def __init__(self, clip_eps: float = None) -> None:
        super().__init__()
        self.clip_eps = clip_eps

    def forward(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.clip_eps is not None:
            values_clipped = old_values + (values - old_values).clamp(-self.clip_eps, self.clip_eps)
            surr1 = (values_clipped - returns) ** 2
            surr2 = (values - returns) ** 2
            loss = torch.max(surr1, surr2)
        else:
            loss = (values - returns) ** 2

        loss = masked_mean(loss, action_mask, dim=-1).mean()
        return 0.5 * loss


class PairWiseLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    """

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        if margin is not None:
            loss = -F.logsigmoid(chosen_reward - reject_reward - margin)
        else:
            loss = -F.logsigmoid(chosen_reward - reject_reward)
        return loss.mean()


class LogExpLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    Details: https://arxiv.org/abs/2204.05862
    """

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        loss = torch.log(1 + torch.exp(reject_reward - chosen_reward)).mean()
        return loss


class DPOLoss(nn.Module):
    """
    DPO Loss
    """

    def __init__(self, beta: float, label_smoothing: float = 0.0, ipo: bool = False) -> None:
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ipo = ipo

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        if self.ipo:
            losses = (
                logits - 1 / (2 * self.beta)
            ) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        else:
            # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )

        loss = losses.mean()
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards


# Adapted from https://github.com/huggingface/transformers/blob/3b742ea84cfc32432d60c0b65c886576ef736833/src/transformers/models/mixtral/modeling_mixtral.py#L77
class SwitchBalancingLoss(nn.Module):
    def __init__(self, num_experts: torch.Tensor = None, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, gate_logits: torch.Tensor) -> float:
        r"""
        Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

        See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
        function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
        experts is too unbalanced.

        Args:
            gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
                Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
                shape [batch_size X sequence_length, num_experts].
            num_experts (`int`, *optional*):
                Number of experts

        Returns:
            The auxiliary loss.
        """
        if gate_logits is None or not isinstance(gate_logits, tuple):
            return 0

        if isinstance(gate_logits, tuple):
            compute_device = gate_logits[0].device
            concatenated_gate_logits = torch.cat(
                [layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0
            )

        routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

        _, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        # treat `top_k` as tokens (shape is `top_k X [batch_size X sequence_length]`)
        selected_experts = selected_experts.reshape(-1)

        expert_mask = torch.nn.functional.one_hot(selected_experts, self.num_experts)
        expert_mask = torch.max(expert_mask, dim=-2).values

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)

        overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(-1))
        return overall_loss * self.num_experts


# Adapted from https://github.com/ContextualAI/HALOs/blob/ca9b7e3eeea220c0944ad8095d641da33f907a7e/trainers.py#L742
class VanillaKTOLoss(nn.Module):
    """
    KTO loss for even sampling
    """

    def __init__(self, beta: float) -> None:
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
        rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps

        losses = torch.cat(
            (
                1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
            ),
            0,
        ).mean()

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        return losses, chosen_rewards, rejected_rewards


# Adapted from https://github.com/ContextualAI/HALOs/blob/ca9b7e3eeea220c0944ad8095d641da33f907a7e/trainers.py#L770
class KTOLoss(nn.Module):
    """
    KTO loss for uneven sampling
    """

    def __init__(
        self,
        beta: float,
        desirable_weight: float,
        undesirable_weight: float,
        world_size: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.world_size = world_size
        self.device = device
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_KL_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_KL_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        KL = (policy_KL_logps - reference_KL_logps).mean().detach()
        # nn.all_reduce sums up the KL estimates across all devices (gradient will also be scaled by world size)
        dist.nn.all_reduce(KL, op=dist.ReduceOp.SUM)
        # take average (will also scale gradients appropriately)
        KL = (KL / self.world_size).clamp(min=0)

        if policy_chosen_logps.shape[0] != 0:
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            chosen_losses = 1 - F.sigmoid(self.beta * (chosen_logratios - KL))
            chosen_rewards = self.beta * chosen_logratios.detach()
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            chosen_losses = torch.Tensor([]).to(policy_rejected_logps.dtype).to(self.device)
            chosen_rewards = torch.Tensor([]).to(policy_rejected_logps.dtype).to(self.device)

        if policy_rejected_logps.shape[0] != 0:
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            rejected_losses = 1 - F.sigmoid(self.beta * (KL - rejected_logratios))
            rejected_rewards = self.beta * rejected_logratios.detach()
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            rejected_losses = torch.Tensor([]).to(policy_chosen_logps.dtype).to(self.device)
            rejected_rewards = torch.Tensor([]).to(policy_chosen_logps.dtype).to(self.device)

        losses = torch.cat(
            (self.desirable_weight * chosen_losses, self.undesirable_weight * rejected_losses), 0
        ).mean()
        return losses, chosen_rewards, rejected_rewards, KL
