from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from einops import rearrange

def matmul_with_importance(
    input: torch.Tensor,
    weight: torch.Tensor,
    probs: torch.Tensor,
    grad_output: Optional[torch.Tensor] = None,
    num_heads: int = 1,
    top_p: float = 1.0,
    top_k: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """matmul input and weight and return output (with optional grad_input, grad_weight whenever grad_output is given) 
    where only the important elements of the input tensor can be computed and gathered to the output tensor
    decided by the importance probability tensor, tuned by top_p and top_k
    
    Args:
        input (torch.Tensor): input tensor in the range of [-1, 1], with shape: [batch_size, seq_len, hidden_size]
        weight (torch.Tensor): weight tensor in the range of [-1, 1], with shape: [hidden_size, embed_size]
        probs (torch.Tensor): probability tensor in the range of [0, 1], with shape: [batch_size, seq_len]
        grad_output (Optional[torch.Tensor], optional): gradient for the output tensor, with shape: [t, hidden_size]. Defaults to None.
        num_heads (int): number of heads to split hidden_size
        top_p (float, [0., 1.]): only the elements with the probability equal or higher than top_p are important ones
        top_k (int, [1, ..., seq_len], optional): only the elements with the top_k highest probability are important ones
    
    Returns:
        output (torch.Tensor): output tensor, with shape: [t, num_heads, embed_size]
        grad_input (torch.Tensor, optional): gradient for the input tensor if grad_output is given, otherwise None
        grad_weight (torch.Tensor, optional): gradient for the weight tensor if grad_output is given, otherwise None
    """

    T_index = []
    for i in range(probs.shape[0]):
        T_index.append(list(set(probs[i].topk(top_k).indices.tolist()) & set(torch.where(probs[i]>top_p)[0].tolist())))

    if num_heads == 1:
        A1 = input #[batch_size, seq_len, hidden_size]
        W1 = weight #[hidden_size, embed_size]
        O1 = torch.einsum('bsh,he->bse',A1,W1)
    else:
        A2 = input.view(input[0], input[1], num_heads, input[-1]//num_heads) #[b, s, nh, hd]
        W2 = weight.view(num_heads, weight[-1]//num_heads, weight[1]) #[nh, hd, e]
        O2 = torch.einsum('bsnh,nhe->bsne',A2,W2)
        O3 = rearrange('b s n e -> (b s) n e',O2)
        for i in range(len(T_index)):
            O3[:, T_index[i], :]
    raise NotImplementedError("TODO: Assignment0 - Task1")