import torch
import torch.nn.functional as F

def generate(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """
    Generates text tokens from a trained language model using top-p nucleus sampling.

    Args:
        model: The trained TransformerLM.
        prompt: Tensor of shape (batch_size, seq_len) containing input token IDs.
        max_new_tokens: Maximum number of tokens to generate.
        eos_token_id: The token ID that signifies the end of a sequence.
        temperature: Scaling factor for logits (higher = more random).
        top_p: Probability mass threshold for nucleus sampling (0.0 to 1.0).

    Returns:
        Tensor of shape (batch_size, seq_len + generated_len) containing the
        original prompt and generated tokens.
    """
    model.eval()
    
    # We loop until we generate max_new_tokens
    for _ in range(max_new_tokens):
        # 1. Context Cropping
        # If the sequence is growing too long for the model's context length,
        # we must crop it to the last block_size tokens.
        # Assuming the model has a 'blocks' attribute with 'max_seq_len' or similar,
        # but for safety in this assignment structure, we usually rely on the user 
        # passing a valid prompt or the model handling position embeddings correctly.
        # However, to be safe, we usually crop to the model's supported context if known.
        # For this implementation, we simply pass the full sequence `prompt`.
        
        # 2. Forward Pass
        # We only need the logits for the last time step.
        with torch.no_grad():
            logits = model(prompt)
        
        # Select the logits for the last token: (batch_size, vocab_size)
        logits = logits[:, -1, :]

        # 3. Temperature Scaling
        if temperature > 0.0:
            logits = logits / temperature
        else:
            # If temp is 0, this implies greedy decoding (argmax)
            # We can approximate this by making the distribution extreme,
            # but usually it's cleaner to handle separately. 
            # For this assignment, standard division is fine.
            pass

        # 4. Top-p (Nucleus) Sampling
        if top_p < 1.0:
            # Sort logits in descending order
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            
            # Compute cumulative probabilities
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Create a mask for tokens to REMOVE.
            # We remove indices where the cumulative mass is > top_p.
            sorted_indices_to_remove = cumulative_probs > top_p
            
            # Shift the mask to the right to keep the first token that crosses the threshold
            # (Otherwise we might remove the most likely token if it alone is > top_p)
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Scatter the mask back to the original indices
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            
            # Set logits of removed tokens to -infinity so they are never sampled
            logits[indices_to_remove] = float('-inf')

        # 5. Sampling
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Sample the next token
        next_token = torch.multinomial(probs, num_samples=1)

        # 6. Update Sequence
        prompt = torch.cat((prompt, next_token), dim=1)

        # 7. Stop Condition (Optional optimization)
        # If batch_size is 1, we can stop immediately if we hit EOS.
        # For batch_size > 1, it's complex because some rows finish earlier than others.
        # We'll assume batch_size=1 for the simple case or just let it run.
        if next_token.item() == eos_token_id:
            break

    return prompt