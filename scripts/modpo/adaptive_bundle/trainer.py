from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.trainer.modpo_trainer import MODPOTrainer


class AdaptiveBundleMODPOTrainer(MODPOTrainer):
    """MODPO-framework trainer exposing first-order DPO objectives.

    The adaptive-bundle method treats each objective as its own DPO loss.
    It needs an oracle returning the current trainable parameter vector,
    per-objective losses, and per-objective gradients. This subclass keeps
    that machinery local to the adaptive-bundle experiment instead of changing
    the repository's original MODPO trainer behavior.
    """

    def dpo_objective_loss(
        self,
        model: nn.Module,
        batch: Dict[str, Union[List, torch.LongTensor]],
    ) -> torch.FloatTensor:
        policy_chosen_logps, policy_rejected_logps, _, _ = self.forward(model, batch)
        with torch.no_grad():
            reference_chosen_logps, reference_rejected_logps, _, _ = self.forward(self.ref_model, batch)

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps)
        logits = chosen_rewards - rejected_rewards

        if self.loss_type == "sigmoid":
            losses = -F.logsigmoid(logits)
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - logits)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge']")

        return losses.mean()

    def _trainable_named_parameters(
        self,
        model: Optional[nn.Module] = None,
    ) -> List[Tuple[str, torch.nn.Parameter]]:
        model = self.model if model is None else model
        return [(name, param) for name, param in model.named_parameters() if param.requires_grad]

    def get_trainable_parameter_vector(
        self,
        model: Optional[nn.Module] = None,
        cpu: bool = True,
    ) -> torch.Tensor:
        chunks = []
        for _, param in self._trainable_named_parameters(model):
            tensor = param.detach().reshape(-1).to(torch.float32)
            if cpu:
                tensor = tensor.cpu()
            chunks.append(tensor)
        if not chunks:
            return torch.empty(0, dtype=torch.float32)
        return torch.cat(chunks)

    @torch.no_grad()
    def set_trainable_parameter_vector(
        self,
        vector: Union[torch.Tensor, Sequence[float]],
        model: Optional[nn.Module] = None,
    ) -> None:
        model = self.model if model is None else model
        flat_vector = torch.as_tensor(vector)
        offset = 0
        named_params = self._trainable_named_parameters(model)
        expected_numel = sum(param.numel() for _, param in named_params)
        if flat_vector.numel() != expected_numel:
            raise ValueError(
                f"Vector has {flat_vector.numel()} elements, but trainable parameters "
                f"require {expected_numel} elements."
            )

        for _, param in named_params:
            next_offset = offset + param.numel()
            value = flat_vector[offset:next_offset].to(device=param.device, dtype=param.dtype).view_as(param)
            param.copy_(value)
            offset = next_offset

    def _flatten_gradient_tensors(
        self,
        grad_tensors: Sequence[Optional[torch.Tensor]],
        params: Sequence[torch.nn.Parameter],
        cpu: bool = True,
    ) -> torch.Tensor:
        chunks = []
        for grad, param in zip(grad_tensors, params):
            if grad is None:
                grad = torch.zeros_like(param)
            tensor = grad.detach().reshape(-1).to(torch.float32)
            if cpu:
                tensor = tensor.cpu()
            chunks.append(tensor)
        if not chunks:
            return torch.empty(0, dtype=torch.float32)
        return torch.cat(chunks)

    def multi_objective_gradient_oracle(
        self,
        objective_batches: Union[
            Sequence[Dict[str, Union[List, torch.LongTensor]]],
            Dict[str, Dict[str, Union[List, torch.LongTensor]]],
        ],
        model: Optional[nn.Module] = None,
        objective_names: Optional[Sequence[str]] = None,
        prepare_inputs: bool = True,
        as_numpy: bool = True,
    ) -> Dict[str, Any]:
        """Return Bundle-compatible first-order information at current params."""
        model = self.model if model is None else model
        named_params = self._trainable_named_parameters(model)
        if not named_params:
            raise ValueError("Cannot build a gradient oracle because the model has no trainable parameters.")
        param_names = [name for name, _ in named_params]
        params = [param for _, param in named_params]

        if isinstance(objective_batches, dict):
            names = list(objective_batches.keys())
            batches = list(objective_batches.values())
        else:
            batches = list(objective_batches)
            names = list(objective_names) if objective_names is not None else [
                f"objective_{idx}" for idx in range(len(batches))
            ]

        if len(batches) == 0:
            raise ValueError("objective_batches must contain at least one objective batch.")
        if len(names) != len(batches):
            raise ValueError("objective_names must have the same length as objective_batches.")

        fvals = []
        grads = []
        for batch in batches:
            batch = self._prepare_inputs(batch) if prepare_inputs else batch
            loss = self.dpo_objective_loss(model, batch)
            grad_tensors = torch.autograd.grad(
                loss,
                params,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            fvals.append(loss.detach().to(torch.float32).cpu())
            grads.append(self._flatten_gradient_tensors(grad_tensors, params, cpu=True))

        x = self.get_trainable_parameter_vector(model, cpu=True)
        fvals_tensor = torch.stack(fvals)
        grads_tensor = torch.stack(grads)

        if as_numpy:
            x_out = x.numpy()
            fvals_out = fvals_tensor.numpy()
            grads_out = grads_tensor.numpy()
        else:
            x_out = x
            fvals_out = fvals_tensor
            grads_out = grads_tensor

        return {
            "x": x_out,
            "fvals": fvals_out,
            "grads": grads_out,
            "objective_names": names,
            "param_names": param_names,
            "param_shapes": [tuple(param.shape) for param in params],
        }

    def multi_objective_gradient_oracle_over_batches(
        self,
        objective_batch_groups: Mapping[str, Sequence[Dict[str, Union[List, torch.LongTensor]]]],
        model: Optional[nn.Module] = None,
        prepare_inputs: bool = True,
        as_numpy: bool = True,
    ) -> Dict[str, Any]:
        """Average each objective's loss and gradient over a fixed batch group.

        This is the deterministic oracle used by the LLM formulation: each
        objective is evaluated on a fixed subset, split into small GPU batches.
        """
        model = self.model if model is None else model
        named_params = self._trainable_named_parameters(model)
        if not named_params:
            raise ValueError("Cannot build a gradient oracle because the model has no trainable parameters.")
        param_names = [name for name, _ in named_params]
        params = [param for _, param in named_params]

        names = list(objective_batch_groups.keys())
        if not names:
            raise ValueError("objective_batch_groups must contain at least one objective.")

        fvals = []
        grads = []
        for name in names:
            batches = list(objective_batch_groups[name])
            if not batches:
                raise ValueError(f"Objective {name!r} has no batches.")

            total_examples = 0
            loss_sum = torch.tensor(0.0, dtype=torch.float32)
            grad_sums = [
                torch.zeros(param.numel(), dtype=torch.float32)
                for param in params
            ]

            for batch in batches:
                batch = self._prepare_inputs(batch) if prepare_inputs else batch
                batch_examples = int(batch["input_ids"].shape[0] // 2)
                if batch_examples <= 0:
                    raise ValueError("DPO objective batches must contain chosen and rejected halves.")

                loss = self.dpo_objective_loss(model, batch)
                grad_tensors = torch.autograd.grad(
                    loss,
                    params,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )

                total_examples += batch_examples
                loss_sum += loss.detach().to(torch.float32).cpu() * batch_examples
                for idx, (grad, param) in enumerate(zip(grad_tensors, params)):
                    if grad is None:
                        continue
                    grad_sums[idx] += grad.detach().reshape(-1).to(torch.float32).cpu() * batch_examples

            fvals.append(loss_sum / total_examples)
            grads.append(torch.cat([grad_sum / total_examples for grad_sum in grad_sums]))

        x = self.get_trainable_parameter_vector(model, cpu=True)
        fvals_tensor = torch.stack(fvals)
        grads_tensor = torch.stack(grads)

        if as_numpy:
            x_out = x.numpy()
            fvals_out = fvals_tensor.numpy()
            grads_out = grads_tensor.numpy()
        else:
            x_out = x
            fvals_out = fvals_tensor
            grads_out = grads_tensor

        return {
            "x": x_out,
            "fvals": fvals_out,
            "grads": grads_out,
            "objective_names": names,
            "param_names": param_names,
            "param_shapes": [tuple(param.shape) for param in params],
        }
