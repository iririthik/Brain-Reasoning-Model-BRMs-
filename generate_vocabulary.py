import argparse

NUMBER_WORDS = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
    10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
    20: "twenty", 30: "thirty", 40: "forty", 50: "fifty",
    60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety", 100: "hundred"
}

OPERATORS = ["+", "-", "*", "/", "=", "(", ")"]
MATH_WORDS = ["plus", "minus", "times", "divided by", "equals", "sum", "difference"]

def generate_vocabulary(max_num: int = 100) -> list[str]:
    vocab = []
    
    # 1. Digits and Numbers up to max_num
    for i in range(max_num + 1):
        vocab.append(str(i))
        
    # 2. Number words
    for val in NUMBER_WORDS.values():
        vocab.append(val)
        
    # 3. Operators and Math words
    vocab.extend(OPERATORS)
    vocab.extend(MATH_WORDS)

    # Deduplicate while preserving order
    seen = set()
    unique_vocab = []
    for token in vocab:
        if token not in seen:
            seen.add(token)
            unique_vocab.append(token)
            
    return unique_vocab

def main():
    parser = argparse.ArgumentParser("Generate simplified math vocabulary")
    parser.add_argument("--output", type=str, default="vocabulary.txt")
    parser.add_argument("--max-num", type=int, default=100)
    args = parser.parse_args()

    vocab = generate_vocabulary(args.max_num)
    
    with open(args.output, "w") as f:
        for t in vocab:
            f.write(f"{t}\n")
            
    print(f"Generated clean math vocabulary (numbers/operators only) with {len(vocab)} tokens at {args.output}")

if __name__ == "__main__":
    main()
