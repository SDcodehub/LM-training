#!/usr/bin/env python3
"""
Script for generating text from a trained TransformerLM checkpoint.
"""
import argparse
import torch

from LM_training.nn.modules import TransformerLM
from LM_training.tokenizer import Tokenizer
from LM_training.inference import generate


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from a trained LM")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--vocab", type=str, required=True, help="Path to vocab JSON file")
    parser.add_argument("--merges", type=str, required=True, help="Path to merges TXT file")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Text prompt")
    parser.add_argument("--max_tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    
    # Model architecture (must match training config)
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1024)
    
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    """Resolve device with fallback: CUDA -> MPS -> CPU."""
    req = (requested or "").lower()
    if req in ("", "auto"):
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if req.startswith("cuda"):
        if torch.cuda.is_available():
            return req
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if req == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return req


def main():
    args = parse_args()
    
    # Resolve device
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.vocab} and {args.merges}...")
    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab,
        merges_filepath=args.merges,
        special_tokens=["<|endoftext|>"]
    )
    
    # Get special token ID
    eos_token_id = tokenizer.encoder_vocab.get(b"<|endoftext|>")
    print(f"EOS token ID: {eos_token_id}")
    
    # Initialize model with same architecture as training
    print("Initializing model...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        device=device
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    
    print(f"Model loaded! Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Encode prompt
    prompt_ids = tokenizer.encode(args.prompt)
    print(f"\nPrompt: '{args.prompt}'")
    print(f"Encoded as {len(prompt_ids)} tokens: {prompt_ids}")
    
    # Convert to tensor
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    # Generate
    print(f"\nGenerating with temperature={args.temperature}, top_p={args.top_p}...")
    print("-" * 60)
    
    output_ids = generate(
        model=model,
        prompt=prompt_tensor,
        max_new_tokens=args.max_tokens,
        eos_token_id=eos_token_id if eos_token_id else -1,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    
    # Decode output
    output_text = tokenizer.decode(output_ids[0].tolist())
    
    print(output_text)
    print("-" * 60)
    print(f"Generated {output_ids.shape[1] - len(prompt_ids)} new tokens")


if __name__ == "__main__":
    main()

