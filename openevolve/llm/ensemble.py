"""
Model ensemble for LLMs
"""

import asyncio
import logging
import random
from typing import Dict, List, Optional, Tuple

from openevolve.llm.base import LLMInterface
from openevolve.llm.openai import OpenAILLM
from openevolve.config import LLMModelConfig

logger = logging.getLogger(__name__)


class LLMEnsemble:
    """Ensemble of LLMs"""

    def __init__(self, models_cfg: List[LLMModelConfig]):
        self.models_cfg = models_cfg

        # Initialize models from the configuration
        self.models = [
            model_cfg.init_client(model_cfg) if model_cfg.init_client else OpenAILLM(model_cfg)
            for model_cfg in models_cfg
        ]

        self.island_models: dict[int, List[LLMInterface]] = {}
        for model_cfg in models_cfg:
            if model_cfg.island is not None:
                if model_cfg.island not in self.island_models:
                    self.island_models[model_cfg.island] = []
                self.island_models[model_cfg.island].append(
                    model_cfg.init_client(model_cfg) if model_cfg.init_client else OpenAILLM(model_cfg)
                )
                
        # Extract and normalize model weights
        self.weights = [model.weight for model in models_cfg]
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

        # Extract and normalize island model weights
        self.island_weights: dict[int, List[float]] = {}
        for island_id, island_model_list in self.island_models.items():
            island_weights = [
                model_cfg.weight
                for model_cfg in models_cfg
                if model_cfg.island == island_id
            ]
            total_island_weight = sum(island_weights)
            self.island_weights[island_id] = [w / total_island_weight for w in island_weights]

        # Set up random state for deterministic model selection
        self.random_state = random.Random()
        # Initialize with seed from first model's config if available
        if (
            models_cfg
            and hasattr(models_cfg[0], "random_seed")
            and models_cfg[0].random_seed is not None
        ):
            self.random_state.seed(models_cfg[0].random_seed)
            logger.debug(
                f"LLMEnsemble: Set random seed to {models_cfg[0].random_seed} for deterministic model selection"
            )

        # Only log if we have multiple models or this is the first ensemble
        if (len(models_cfg) > 1 or not hasattr(logger, "_ensemble_logged")) and len(self.island_models) == 0:
            logger.info(
                f"Initialized LLM ensemble with models: "
                + ", ".join(
                    f"{model.name} (weight: {weight:.2f})"
                    for model, weight in zip(models_cfg, self.weights)
                )
            )
            logger._ensemble_logged = True
        elif len(self.island_models) > 0:
            for island_id, island_model_list in self.island_models.items():
                island_weights = self.island_weights[island_id]
                logger.info(
                    f"Initialized LLM ensemble for island {island_id} with models: "
                    + ", ".join(
                        f"{model_cfg.name} (weight: {weight:.2f})"
                        for model_cfg, weight in zip(
                            [mc for mc in models_cfg if mc.island == island_id],
                            island_weights,
                        )
                    )
                )
            logger._ensemble_logged = True


    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using a randomly selected model based on weights"""
        model = self._sample_model()
        return await model.generate(prompt, **kwargs)

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], island_unique_models: bool, island_id: int | None = None, **kwargs
    ) -> str:
        """Generate text using a system message and conversational context"""
        model = self._sample_model(island_unique_models, island_id)
        return await model.generate_with_context(system_message, messages, **kwargs)

    def _sample_model(self, island_unique_models: bool, island_id: int | None = None) -> LLMInterface:
        """Sample a model from the ensemble based on weights"""
        if island_unique_models and island_id is not None:
            island_models: List[LLMInterface] = self.island_models[island_id]
            island_weights = self.island_weights[island_id]
            index = self.random_state.choices(range(len(island_models)), weights=island_weights, k=1)[0]
            sampled_model = island_models[index]
            logger.info(f"Sampled island model: {vars(sampled_model)['model']}")
            return sampled_model
        else:
            index = self.random_state.choices(range(len(self.models)), weights=self.weights, k=1)[0]
            sampled_model = self.models[index]
            logger.info(f"Sampled model: {vars(sampled_model)['model']}")
            return sampled_model

    async def generate_multiple(self, prompt: str, n: int, **kwargs) -> List[str]:
        """Generate multiple texts in parallel"""
        tasks = [self.generate(prompt, **kwargs) for _ in range(n)]
        return await asyncio.gather(*tasks)

    async def parallel_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts in parallel"""
        tasks = [self.generate(prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks)

    async def generate_all_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a all available models and average their returned metrics"""
        responses = []
        for model in self.models:
            responses.append(await model.generate_with_context(system_message, messages, **kwargs))
        return responses
