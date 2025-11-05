"""Placeholder agent to be replaced by the SAC + DrQ-v2 implementation."""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np

from agent_interface import AgentInterface


class RLAgent(AgentInterface):
    """Temporary stub; will be replaced by the SAC + DrQ-v2 agent."""

    def __init__(
        self,
        observation_space=None,
        action_space=None,
        seed: Optional[int] = None,
        **_: object,
    ) -> None:
        super().__init__(observation_space, action_space, seed)
        self.logger = logging.getLogger(__name__)
        raise NotImplementedError(
            "RLAgent has been removed while transitioning to the new SAC + DrQ-v2 agent."
        )

    def act(self, observation: Union[np.ndarray, dict], **kwargs) -> np.ndarray:
        raise NotImplementedError(
            "RLAgent.act is unavailable until the SAC + DrQ-v2 agent is implemented."
        )

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.seed = seed

*** End of File
