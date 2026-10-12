"""Fuzzy reward shaping engine.

Uses expert knowledge encoded as fuzzy rules to shape the reward signal,
accelerating learning while maintaining convergence guarantees (via
potential-based shaping approach).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .tsk_fuzzy import TSKFuzzySystem
from .fuzzy_rules import FuzzyRuleBase, expert_rule_base
from ..utils.config import FuzzyConfig

logger = logging.getLogger(__name__)


class FuzzyRewardShaper(nn.Module):
    """Fuzzy reward shaping engine.

    Computes a shaping reward based on the current state using a
    TSK fuzzy system initialized from expert rules. The shaping weight
    decays over training to allow the agent to learn from the true reward.

    This follows the potential-based reward shaping principle:
        F(s, a, s') = gamma * Phi(s') - Phi(s)
    where Phi(s) is the fuzzy potential function.

    Using a state-only potential preserves the optimal policy.
    """

    def __init__(
        self,
        cfg: FuzzyConfig,
        num_inputs: int,
        rule_base: Optional[FuzzyRuleBase] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.num_inputs = num_inputs

        # Fuzzy system for potential function Phi(s)
        self.fuzzy_system = TSKFuzzySystem(
            num_inputs=num_inputs,
            num_outputs=1,  # potential value
            num_rules=cfg.num_rules,
            first_order=False,  # zero-order for interpretability
        )

        # Initialize from expert rules if available
        if cfg.use_expert_rules and rule_base is not None:
            self._init_from_rule_base(rule_base)

        # Shaping weight (will be decayed during training)
        self.register_buffer(
            "shaping_weight",
            torch.tensor(cfg.reward_shaping_weight_init, dtype=torch.float32),
        )
        self._step_count = 0

    def _semantic_input_names(self) -> list:
        """Map each observation index to its expert-rule semantic input name.

        Expert rules use the keys "soc", "pv_normalized", "load_normalized"
        and "time_of_day". Indices without a corresponding expert set are
        None (treated as don't-care by the rule base). Observation layout:

            grid (8):    soc, pv, wt, load, t, sin, cos, price
            island (9):  soc, pv, wt, load, freq, volt, t, sin, cos
        """
        if self.num_inputs >= 9:
            return ["soc", "pv_normalized", None, "load_normalized",
                    None, None, "time_of_day", None, None]
        return ["soc", "pv_normalized", None, "load_normalized",
                "time_of_day", None, None, None][:self.num_inputs]

    def _init_from_rule_base(self, rule_base: FuzzyRuleBase) -> None:
        """Initialize fuzzy system parameters from expert rule base."""
        input_names = self._semantic_input_names()
        means, stds, conseq, confidences = rule_base.to_tsk_params(
            input_names
        )

        num_rules = min(len(rule_base), self.cfg.num_rules)
        num_mapped_inputs = min(means.shape[1], self.num_inputs)

        with torch.no_grad():
            self.fuzzy_system.membership.means[:num_rules, :num_mapped_inputs] = (
                torch.from_numpy(means[:num_rules, :num_mapped_inputs])
            )
            stds_clipped = torch.clamp(
                torch.from_numpy(stds[:num_rules, :num_mapped_inputs]), 1e-2
            )
            self.fuzzy_system.membership.log_stds[:num_rules, :num_mapped_inputs] = (
                torch.log(stds_clipped)
            )
            # Consequents scaled by PER-RULE confidence (not rule[0].confidence
            # for all). This restores the expert's intended relative weighting
            # between strong rules (low-SOC protection, conf=0.9) and weak
            # rules (night standby, conf=0.5).
            if conseq.shape[0] > 0:
                conf_t = torch.from_numpy(confidences[:num_rules]).unsqueeze(-1)
                self.fuzzy_system.consequent[:num_rules, :] = (
                    torch.from_numpy(conseq[:num_rules, :]) * conf_t
                )
            else:
                self.fuzzy_system.consequent[:num_rules, :] = torch.zeros(num_rules, 1)

        logger.info(
            f"Initialized fuzzy reward shaper from {num_rules} expert rules "
            f"({num_mapped_inputs} input dimensions mapped)."
        )

    def potential(self, state: torch.Tensor) -> torch.Tensor:
        """Compute potential function Phi(s).

        Args:
            state: State tensor (batch_size, state_dim).

        Returns:
            Potential values (batch_size, 1).
        """
        return self.fuzzy_system(state)

    def shape_reward(
        self,
        reward: torch.Tensor,
        state: torch.Tensor,
        next_state: torch.Tensor,
        gamma: float,
        dones: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute shaped reward using potential-based shaping.

        F(s,a,s') = gamma * Phi(s') - Phi(s).  To preserve the
        policy-invariance guarantee of Ng et al., the terminal potential must
        be zero: for absorbing/terminal next states we use F = -Phi(s)
        (i.e. Phi(s_terminal) = 0).

        Args:
            reward: Original environment reward (batch_size, 1).
            state: Current state (batch_size, state_dim).
            next_state: Next state (batch_size, state_dim).
            gamma: Discount factor.
            dones: Optional terminal mask (batch_size,) with 1.0 on absorbing
                transitions.  If None, all transitions are treated as
                non-terminal.

        Returns:
            Shaped reward (batch_size, 1).
        """
        phi_s = self.potential(state)
        phi_next = self.potential(next_state)
        if dones is None:
            shaping_term = gamma * phi_next - phi_s
        else:
            # dones shape: (batch,) -> broadcast to (batch, 1)
            d = dones.view(-1, 1).to(reward.dtype)
            # Terminal: Phi(s') = 0  => F = -Phi(s)
            # Non-terminal:           F = gamma*Phi(s') - Phi(s)
            shaping_term = torch.where(
                d > 0.5, -phi_s, gamma * phi_next - phi_s
            )
        shaped = reward + self.shaping_weight * shaping_term
        return shaped

    def step(self) -> None:
        """Advance one training step and decay shaping weight via cosine
        annealing.

        Cosine annealing keeps the weight high early (expert guidance is
        most valuable when Q is inaccurate), then smoothly eases to final
        near the end of decay — avoiding the abrupt linear cutoff that can
        strip guidance before the policy has converged.
        """
        self._step_count += 1
        init_w = self.cfg.reward_shaping_weight_init
        final_w = self.cfg.reward_shaping_weight_final
        decay_steps = self.cfg.reward_shaping_decay_steps

        if self._step_count <= decay_steps and decay_steps > 0:
            # Cosine annealing: w = final + 0.5*(init-final)*(1+cos(pi*t/T))
            t = self._step_count / decay_steps
            cos_factor = 0.5 * (1.0 + np.cos(np.pi * t))
            new_w = final_w + (init_w - final_w) * cos_factor
            self.shaping_weight.fill_(new_w)

    @property
    def current_weight(self) -> float:
        return float(self.shaping_weight.item())
