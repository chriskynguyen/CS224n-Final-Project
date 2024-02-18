from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary.
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary.
                alpha = group["lr"]

                # Complete the implementation of AdamW here, reading and saving
                # your state in the `state` dictionary above.
                # The hyperparameters can be read from the `group` dictionary
                # (they are lr, betas, eps, weight_decay, as saved in the constructor).
                #
                # To complete this implementation:
                # 1. Update the first and second moments of the gradients.
                # 2. Apply bias correction
                #    (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                #     also given in the pseudo-code in the project description).
                # 3. Update parameters (p.data).
                # 4. Apply weight decay after the main gradient-based updates.
                # Refer to the default project handout for more details.

                ### TODO: FINISHED
                # Access the remaining paramters from group
                beta1, beta2 = group["betas"][0], group["betas"][1]
                eps = group["eps"]
                weight_decay = group["weight_decay"]

                # Initialize state (only if not initialized already)
                # p.data == theta in pseudo-code
                if len(state) == 0:
                    state['step'] = 0
                    state['m_t'] = torch.zeros_like(p.data) # exponential moving averages of gradient
                    state['v_t'] = torch.zeros_like(p.data) # squared gradient
                
                state['step'] += 1
                # Update biased first moment estimate
                state['m_t'] = beta1 * state['m_t'] + (1 - beta1) * grad
                # Update biased second raw moment estimate
                state['v_t'] = beta2 *  state['v_t'] + (1 - beta2) * grad**2

                if group["correct_bias"]:
                    # Compute bias-corrected first moment estimate
                    m_t_corrected = state['m_t'] / (1 - beta1**state["step"])
                    # Compute bias-corrected second raw moment estimate
                    v_t_corrected = state['v_t'] / (1 - beta2**state["step"])

                # Update parameters
                p.data -= alpha * m_t_corrected / (torch.sqrt(v_t_corrected) + eps)

                # Apply weight decay
                if weight_decay != 0:
                    p.data -= alpha * weight_decay * p.data

        return loss
