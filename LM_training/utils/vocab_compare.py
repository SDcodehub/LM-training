import json
import sys

def compare_vocabs(file_v1, file_v2):
    print(f"Loading {file_v1}...")
    with open(file_v1, 'r', encoding='utf-8') as f:
        v1 = json.load(f)

    print(f"Loading {file_v2}...")
    with open(file_v2, 'r', encoding='utf-8') as f:
        v2 = json.load(f)

    # 1. Check Size
    if len(v1) != len(v2):
        print(f"❌ SIZE MISMATCH: v1 has {len(v1)} tokens, v2 has {len(v2)}")
        return

    # Convert keys to integers for proper comparison
    v1_map = {int(k): tuple(v) for k, v in v1.items()}
    v2_map = {int(k): tuple(v) for k, v in v2.items()}

    # 2. Check Strict Equality (ID to ID)
    mismatches = []
    for token_id, token_bytes in v1_map.items():
        if token_id not in v2_map:
            print(f"❌ ID MISMATCH: ID {token_id} missing in v2")
            return
        if v2_map[token_id] != token_bytes:
            mismatches.append((token_id, token_bytes, v2_map[token_id]))

    if not mismatches:
        print("\n✅ SUCCESS: Files are STRICTLY IDENTICAL.")
        print("The optimization did not change a single bit of the output.")
        return

    # 3. Check Set Equality (If strict failed)
    print(f"\n⚠️ Strict equality failed ({len(mismatches)} ID mismatches). Checking Set Equality...")
    
    v1_tokens = set(v1_map.values())
    v2_tokens = set(v2_map.values())

    if v1_tokens == v2_tokens:
        print("\n✅ PARTIAL SUCCESS: Vocabularies contain the SAME TOKENS, but IDs are different.")
        print("This is functionally equivalent but models trained on v1 won't work with v2 tokenizer.")
    else:
        diff_v1 = len(v1_tokens - v2_tokens)
        diff_v2 = len(v2_tokens - v1_tokens)
        print(f"\n❌ FAILURE: Vocabularies have diverged.")
        print(f"Tokens in v1 but not v2: {diff_v1}")
        print(f"Tokens in v2 but not v1: {diff_v2}")
        print("Reason: Tie-breaking logic likely differed between implementations.")

if __name__ == "__main__":
    compare_vocabs(
        "bpe_tokenizer/TinyStoriesV2-GPT4-train-v1_vocab.json", 
        "bpe_tokenizer/TinyStoriesV2-GPT4-train-v2_vocab.json"
    )