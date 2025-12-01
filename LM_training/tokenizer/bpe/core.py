"""
Tokenizer class for tokenizing text
"""
import regex as re
import json
from typing import Dict, List, Tuple, Iterable, Iterator
from LM_training.utils.logging_config import get_logger


log = get_logger()

# GPT 2 tokenizer pattern
# This regex splits the text into chunks of letters numbers or punctuations
# Its designed to keep the spaces attached to the words that follow them
split_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    """
    Tokenizer class for tokenizing text
    """
    def __init__(self, vocab:Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: List[str] = None):
        """
        Construct the Tokenizer
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # Append any missing special tokens to the vocab with new IDs (UTF-8 bytes)
        if self.special_tokens:
            existing_values = set(self.vocab.values())
            next_id = (max(self.vocab.keys()) + 1) if self.vocab else 0
            for token in self.special_tokens:
                token_bytes = token.encode("utf-8")
                if token_bytes not in existing_values:
                    self.vocab[next_id] = token_bytes
                    existing_values.add(token_bytes)
                    next_id += 1

        # reverse mapping of vocab
        self.encoder_vocab = {b: i for i, b in self.vocab.items()}

        # reverse mapping of merges
        self.merge_ranks = {pair: r for r, pair in enumerate(self.merges)}

        # build the special token pattern
        if self.special_tokens:
            # Prefer longer overlapping special tokens first
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            special_pattern = "|".join(re.escape(token) for token in sorted_specials)
            self.special_token_pattern = re.compile(f"({special_pattern})")
        else:
            self.special_token_pattern = None


    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: List[str] = None):
        """
        class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges 
        (in the same format that your BPE training code output) and (optionally) a list of special 
        tokens. This method should accept the following additional parameters:
        """

        # load the vocabulary
        log.info("loading vocabulary from %s", vocab_filepath)

        # This will load the file into a dictionary like:
        # {"256": [60, 124, 101, 110, 100, 111, 102, 116, 101, 120, 116, 124, 62], ...}
        with open(vocab_filepath, "r", encoding="utf-8") as vocab_file:
            vocab = json.load(vocab_file)

        # now lets convert the read file into the format expect in init dict{int: bytes}
        parsed_vocab = {int(token_id): bytes(byte_list) for token_id, byte_list in vocab.items()}
        log.info("Loaded vocabulary size: %d", len(parsed_vocab))

        # lets start working on the merges
        log.info("loading merges from %s", merges_filepath)
        with open(merges_filepath, "r", encoding="utf-8") as merges_files:
            merges = [tuple(bytes.fromhex(h) for h in line.split()) for line in merges_files if line.split()]
        
        log.debug("top 5 merges are %s", merges[:5])
        log.debug("bottom 5 merges are %s", merges[-5:])

        return cls(parsed_vocab, merges, special_tokens)

    def _get_adjucent_pairs(self, tokens: List[bytes]) -> List[Tuple[bytes, bytes]]:
        """
        Get all the adjucent pairs of bytes in the token list
        """
        return set(zip(tokens, tokens[1:]))

    def _apply_bpe_words(self, word_bytes: bytes) -> List[bytes]:
        """
        Apply bpe to a word
        """
        if not word_bytes:
            return []

        # start with the word as a list of single bytes tokens
        tokens = [bytes([b]) for b in word_bytes]

        # apply bpe
        while True:
            # find all the adjucent pairs of bytes in token list
            pairs = self._get_adjucent_pairs(tokens)
            if not pairs:
                break
            
            # Find the best pair to merge. This is the pair that appears in
            # our merge rules and has the lowest rank (was learned earliest).
            best_pair = min(pairs, key=lambda pair: self.merge_ranks.get(pair, float("inf")))

            # If the best pair has a rank of infinity, it means none of the
            # adjacent pairs are in our merge rules. So, we're done.
            if self.merge_ranks.get(best_pair, float("inf")) == float("inf"):
                break

            # We found a pair to merge. Let's rebuild the token list.
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                    # Merge the best pair
                    new_tokens.append(tokens[i] + tokens[i+1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens



    def encode(self, text: str) -> List[int]:
        """
        Encode a text string into a list of integers
        """
        final_token_ids = []

        # Isolate special tokens
        if self.special_token_pattern:
            chunks = re.split(self.special_token_pattern, text)
        else:
            chunks = [text]

        # process each chunk
        for chunk in chunks:
            if not chunk:
                continue
            
            # handle chunk which is a special token
            if self.special_tokens and chunk in self.special_tokens:
                # look up the token id directly
                token_id = self.encoder_vocab.get(chunk.encode("utf-8"))
                final_token_ids.append(token_id)
            else:
                # handle regular text. Apply full bpe
                # pre tokenize, chunk into words
                pre_tokens = re.findall(split_pattern, chunk)

                # merging happens at words level, so no non regular merge across split patter in possible
                for word in pre_tokens:
                    # convert the word into bytes
                    word_bytes = word.encode("utf-8")

                    # apply bpe
                    merged_tokens = self._apply_bpe_words(word_bytes)

                    # look for ids for the final merged tokens
                    for token in merged_tokens:
                        token_id = self.encoder_vocab.get(token)
                        final_token_ids.append(token_id)
        
        return final_token_ids


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that 
        lazily yields token IDs. This is required for memory-eﬀicient tokenization of large 
        files that we cannot directly load into memory.
        """
        # Iterate over each chunk/line from the input iterable
        for chunk in iterable:
            # Encode the chunk using the already implemented encode method
            encoded_ids = self.encode(chunk)
            # Lazily yield each token ID from the result
            yield from encoded_ids

    def decode(self, ids: List[int]) -> str:
        """
        Decode a list of integers into a text string
        """
        # get the byte sequence for each id from the vocabolary
        byte_sequence = [self.vocab.get(i, b'') for i in ids]

        # join them all into a single object
        full_byte_sequence = b''.join(byte_sequence)

        # decode the bytes into a string , replacing the errors
        return full_byte_sequence.decode("utf-8", errors="replace")


# At the bottom of your tokonizer.py file

if __name__ == '__main__':
    # Load your trained tokenizer
    tokenizer = Tokenizer.from_files(
        vocab_filepath="bpe_tokenizer/tinystories_vocab.json",
        merges_filepath="bpe_tokenizer/tinystories_merges.txt",
        special_tokens=["<|endoftext|>"]
    )

    # Test sentence with a special token and tricky punctuation
    test_text = "Hello world! This is a test... <|endoftext|> Let's see how it's tokenized."

    print(f"Original text:\n'{test_text}'")

    # Encode the text
    encoded_ids = tokenizer.encode(test_text)
    print(f"\nEncoded IDs ({len(encoded_ids)} tokens):\n{encoded_ids}")

    # Decode the IDs
    decoded_text = tokenizer.decode(encoded_ids)
    print(f"\nDecoded text:\n'{decoded_text}'")

    # Check if they match
    assert test_text == decoded_text
    print("\n✅ Round-trip test passed!")