"""
Tests for LLMEnsemble in openevolve.llm.ensemble
"""

from typing import Any, Dict, List
import unittest
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.config import LLMModelConfig
from openevolve.llm.base import LLMInterface

class TestLLMEnsemble(unittest.TestCase):
    def test_weighted_sampling(self):
        models = [
            LLMModelConfig(name="a", weight=0.0, api_key="test", api_base="http://test"),
            LLMModelConfig(name="b", weight=1.0, api_key="test", api_base="http://test"),
        ]
        ensemble = LLMEnsemble(models)
        # Should always sample model 'b'
        for _ in range(10):
            self.assertEqual(ensemble._sample_model().model, "b")

        models = [
            LLMModelConfig(name="a", weight=0.3, api_key="test", api_base="http://test"),
            LLMModelConfig(name="b", weight=0.3, api_key="test", api_base="http://test"),
            LLMModelConfig(name="c", weight=0.3, api_key="test", api_base="http://test"),
        ]
        ensemble = LLMEnsemble(models)
        # Should sample both models. Track sampled models in a set
        sampled_models = set()
        for _ in range(1000):
            sampled_models.add(ensemble._sample_model().model)
            # Cancel once we have both models
            if len(sampled_models) == len(models):
                break
        self.assertEqual(len(sampled_models), len(models))



class TestEnsembleInit(unittest.TestCase):
    class MyCustomLLM(LLMInterface):
        def __init__(self, model, some_field):
            self.model = model
            self.some_field = some_field

        async def generate(self, prompt: str, **kwargs) -> str:
            return "custom-generate"

        async def generate_with_context(self, system_message: str, messages: List[Dict[str, str]], **kwargs) -> str:
            return "custom-generate-with-context"

    def init_custom_llm(self, model_cfg):
        return self.MyCustomLLM(model=model_cfg.name, some_field="value")

    def test_ensemble_initialization(self):
        models = [
            LLMModelConfig(name="a"),
            LLMModelConfig(name="b", init_client=self.init_custom_llm),
        ]
        ensemble = LLMEnsemble(models)
        self.assertEqual(len(ensemble.models), len(models))
        self.assertEqual(ensemble.models[0].model, "a")
        self.assertEqual(ensemble.models[1].model, "b")
        self.assertEqual(ensemble.models[1].some_field, "value")


class TestModelSchedule(unittest.TestCase):
    class FakeLLM(LLMInterface):
        def __init__(self, name):
            self.model = name

        async def generate(self, prompt: str, **kwargs) -> str:
            return "fake"

        async def generate_with_context(self, system_message: str, messages: List[Dict[str, str]], **kwargs) -> str:
            return "fake"

    def init_fake(self, model_cfg):
        return self.FakeLLM(model_cfg.name)

    def test_scheduled_model_selection(self):
        models = [
            LLMModelConfig(name="big-120B", weight=0.0, api_key="test", api_base="http://test", init_client=self.init_fake),
            LLMModelConfig(name="small-30B", weight=1.0, api_key="test", api_base="http://test", init_client=self.init_fake),
        ]
        ensemble = LLMEnsemble(models, model_schedule={"interval": 50, "model_index": 0})

        # Non-scheduled iterations should use weighted random (only small model has weight)
        for i in [49, 51, 99, 101]:
            sampled = ensemble._sample_model(iteration=i)
            self.assertEqual(sampled.model, "small-30B")

        # Scheduled iterations should use the big model (including iteration 1)
        for i in [1, 50, 100, 150]:
            sampled = ensemble._sample_model(iteration=i)
            self.assertEqual(sampled.model, "big-120B")

    def test_no_schedule_uses_weights(self):
        models = [
            LLMModelConfig(name="a", weight=1.0, api_key="test", api_base="http://test", init_client=self.init_fake),
            LLMModelConfig(name="b", weight=0.0, api_key="test", api_base="http://test", init_client=self.init_fake),
        ]
        ensemble = LLMEnsemble(models)
        for _ in range(10):
            self.assertEqual(ensemble._sample_model(iteration=50).model, "a")

    def test_schedule_with_invalid_model_index_ignored(self):
        models = [
            LLMModelConfig(name="a", weight=1.0, api_key="test", api_base="http://test", init_client=self.init_fake),
        ]
        ensemble = LLMEnsemble(models, model_schedule={"interval": 50, "model_index": 5})
        # Invalid index should fall back to weighted selection
        sampled = ensemble._sample_model(iteration=50)
        self.assertEqual(sampled.model, "a")

    def test_iteration_zero_not_scheduled(self):
        models = [
            LLMModelConfig(name="big", weight=0.0, api_key="test", api_base="http://test", init_client=self.init_fake),
            LLMModelConfig(name="small", weight=1.0, api_key="test", api_base="http://test", init_client=self.init_fake),
        ]
        ensemble = LLMEnsemble(models, model_schedule={"interval": 50, "model_index": 0})
        # Iteration 0 should not trigger schedule (even though 0 % 50 == 0)
        sampled = ensemble._sample_model(iteration=0)
        self.assertEqual(sampled.model, "small")


if __name__ == "__main__":
    unittest.main()
