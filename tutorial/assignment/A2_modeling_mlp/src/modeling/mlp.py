from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import ProcessGroup



class MLPActivationType(Enum):
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"
    SIGMOID = "sigmoid"
    BILINEAR = "bilinear"


class DenseMLPWithLoRA(nn.Module):
    """Dense MLP module with LoRA adapters
    This is a GLU-style dense MLP layer with LoRA adapters.
    """
    
    def __init__(self,
        hidden_size: int,
        ffh_size: int,
        activation_type: MLPActivationType = MLPActivationType.SILU,
        init_base_seed: int = 42,
        lora_rank: int = 0,
        lora_alpha: Optional[float] = None,
        lora_dropout_rate: float = 0.0,
        lora_dropout_seed: int = 42,
        lora_init_base_seed: int = 42,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        """Initialize Dense MLP module with LoRA adapters
        Args:
            hidden_size(int): hidden dimension size
            ffh_size(int): hidden dimension size
            activation_type(MLPActivationType, default = "silu"): activation type
            init_base_seed(int, default = 42): seed for base weight initialization
            lora_rank(int, default = 0): lora rank, if 0, then no lora to apply
            lora_alpha(Optional[float], default = None): lora alpha, if None, then set to lora_rank
            lora_dropout_rate(float, default = 0.0): lora dropout rate
            lora_dropout_seed(int, default = 42): lora dropout seed
            lora_init_base_seed(int, default = 42): seed for lora weight initialization
            dtype(torch.dtype, default = torch.float32): parameter dtype
            device(str, default = "cpu"): parameter device
        """
        super().__init__()
        self.lora_rank = lora_rank
        if self.lora_rank != 0:
            self.lora_rank = lora_rank
            self.lora_alpha = lora_alpha
            self.lora_dropout_rate = lora_dropout_rate
            self.lora_dropout_seed = lora_dropout_seed
            self.lora_init_base_seed = lora_init_base_seed
            self.Ar = nn.Linear(hidden_size, lora_rank, bias=False, device=device, dtype=dtype)
            self.Br = nn.Linear(lora_rank, hidden_size, bias=False, device=device, dtype=dtype)
            torch.manual_seed(lora_dropout_seed)
            self.dropout = nn.Dropout(self.lora_dropout_rate)
        self.activation_type = activation_type
        
        self.init_base_seed = init_base_seed
        self.Gate = nn.Linear(hidden_size, ffh_size, bias=False, device=device, dtype=dtype)
        self.Up = nn.Linear(hidden_size, ffh_size, bias=False, device=device, dtype=dtype)
        self.Down = nn.Linear(ffh_size, hidden_size, bias=False, device=device, dtype=dtype)
        activation_map = {
        MLPActivationType.SILU: nn.SiLU(),
        MLPActivationType.RELU: nn.ReLU(),
        MLPActivationType.GELU: nn.GELU(),
        MLPActivationType.SIGMOID: nn.Sigmoid(),
        MLPActivationType.BILINEAR: nn.Identity()
        }
        self.Activation = activation_map[activation_type]
        self.reset_parameters()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward pass of the Dense MLP module with LoRA adapters
        
        Args:
            input(torch.Tensor): input tensor, with shape: [batch_size, seq_len, hidden_size]
            
        Returns:
            output(torch.Tensor): output tensor, with shape: [batch_size, seq_len, hidden_size]
        """
        if self.lora_rank == 0:
            return self.Down(self.Activation(self.Gate(input)) * self.Up(input))
        else:
            return self.Down(self.Activation(self.Gate(input)) * self.Up(input)) + self.dropout((self.lora_alpha if self.lora_alpha is not None else self.lora_rank) * self.Br(self.Ar(input)))

    def reset_parameters(self):
        """Initialize the weights of the Dense MLP module with LoRA adapters
        from a normal distribution (or a uniform distribution for lora weights)
        """ 
        if self.activation_type == MLPActivationType.BILINEAR or self.activation_type == MLPActivationType.SIGMOID:
            torch.manual_seed(self.init_base_seed + 1)
            nn.init.xavier_uniform_(self.Up.weight)
            torch.manual_seed(self.init_base_seed + 2)
            nn.init.xavier_uniform_(self.Gate.weight)
            torch.manual_seed(self.init_base_seed + 3)
            nn.init.xavier_uniform_(self.Down.weight)
        else:
            torch.manual_seed(self.init_base_seed + 1)
            nn.init.kaiming_uniform_(self.Up.weight)
            torch.manual_seed(self.init_base_seed + 2)
            nn.init.kaiming_uniform_(self.Gate.weight)
            torch.manual_seed(self.init_base_seed + 3)
            nn.init.kaiming_uniform_(self.Down.weight)
        
        if self.lora_rank != 0:
            torch.manual_seed(self.lora_init_base_seed + 1)
            nn.init.uniform_(self.Ar.weight)
            torch.manual_seed(self.lora_init_base_seed + 2)
            nn.init.uniform_(self.Br.weight)



    
class SparseMLPWithLoRA(nn.Module):
    """Sparse MLP module with LoRA adapters
    This is a GLU-style sparse MLP layer with LoRA adapters, \
        where the sparcity is implemented as Mixture of Experts (MoE), \
            and each expert is a dense MLP with LoRA adapters.
    """
    
    def __init__(self,
        hidden_size: int,
        ffh_size: int,
        activation_type: MLPActivationType = MLPActivationType.SILU,
        num_experts: int = 1,
        moe_topk: int = 1,
        rank: int = 0,
        world_size: int = 1,
        process_group: Optional[ProcessGroup] = None,
        init_mean: float = 0.0,
        init_std: float = 1.0,
        init_base_seed: int = 42,
        lora_rank: int = 0,
        lora_alpha: Optional[float] = None,
        lora_dropout_rate: float = 0.0,
        lora_dropout_seed: int = 42,
        lora_init_base_seed: int = 42,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        """Initialize Sparse MLP module with LoRA adapters
        
        Args:
            hidden_size(int): hidden dimension size
            ffh_size(int): hidden dimension size
            activation_type(MLPActivationType, default = MLPActivationType.SILU): activation type
            num_experts(int, default = 1): number of (global) experts, which can deduce expert_size = ffh_size // num_experts
            moe_topk(int, default = 1): topk-routing for MoE to control the sparcity
            rank(int, default = 0): rank
            world_size(int, default = 1): world size
            process_group(Optional[ProcessGroup], default = None): the process group (which will not be used for this simpler module yet)
            init_mean(float, default = 0.0): mean for the initialization
            init_std(float, default = 1.0): std for the initialization
            init_base_seed(int, default = 42): seed for the initialization
            lora_rank(int, default = 0): lora rank
            lora_alpha(Optional[float], default = None): lora alpha
            lora_dropout_rate(float, default = 0.0): lora dropout rate
            lora_dropout_seed(int, default = 42): lora dropout seed
            lora_init_base_seed(int, default = 42): seed for lora weight initialization
            dtype(torch.dtype, default = torch.float32): parameter dtype
            device(str, default = "cpu"): parameter device
        """
        super().__init__()
        raise NotImplementedError("Assignment2 - Task2")
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward pass of the Sparse MLP module with LoRA adapters
        
        Args:
            input(torch.Tensor): input tensor, with shape: [batch_size, seq_len, hidden_size]
            
        Returns:
            output(torch.Tensor): output tensor, with shape: [batch_size, seq_len, hidden_size]
        """
        raise NotImplementedError("Assignment2 - Task2")
        
    def reset_parameters(self):
        """Initialize the weights of each local expert from its own distribution \
            and the gating layer from a normal distribution
        """
        raise NotImplementedError("Assignment2 - Task2")