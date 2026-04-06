# Brain Reasoning Models


---

##  Overview

This repository houses the **Exact-Word Mathematical Model**, an end-to-end framework built to demonstrate the *organic arithmetic reasoning* capabilities of **Graph Neural Networks (GNN) + Tribev2 (by meta)** scanning functional neural representations. 

Instead of explicitly feeding the model billions of combinatorial equations (e.g., `123+456=579`) or relying on external tokenizers, this architecture relies solely on a deeply curated dictionary of mathematical building blocks:
- Digits
- Linguistic Numeral representations
- Operators
- Functional Structural Words

By learning the **biological and topological overlaps (Jaccard overlaps)** of these core brain signatures, the Graph Convolutional Network (GCN) is able to generalize and synthesize the meaning of vastly complex mathematical thoughts purely by triggering its root arithmetic signatures simultaneously!

---

## Architecture & Signature Readout

**Wait, what does this model actually output?**
This decoder is structurally a **Multi-label Signature Extractor**. It doesn't classify one exact text string answer! If the continuous thought flow translates to thinking an equation like `24 + 5 = 29`, the model simultaneously parses the input brain map and outputs a holistic probability activation spread:

```text
  Active signatures detected:

    "24"       0.99  █████████████████████████ ← STRONG
    "+"        0.98  ████████████████████████  ← STRONG
    "5"        0.95  ███████████████████████   ← STRONG
    "29"       0.89  ██████████████████████    ← STRONG
    "="        0.85  █████████████████████     ← STRONG
    "plus"     0.52  █████████████             ← active (neighbouring synonym)
    "sum"      0.45  ███████████               ← active (neighbouring concept)
```

The system retrieves the **full conceptual fingerprint** behind the math, allowing you to synthetically reconstruct exactly what elemental digits, operators, and associated math-concepts triggered visually in the cortex at any given point in time!

---

## Execution Pipeline

**Step 1: Setup & Vocabulary Configuration**
Generate the ultra-lean mathematical token list (`vocabulary.txt`).
```bash
python generate_vocabulary.py --max-num 100
```
*Creates the foundation of digits 0-100 and mathematical operators.*

**Step 2: Generate Functional Neural Activations**
Extract continuous functional neural mapping outputs (`brain_signatures.h5`) using TRIBE v2.
```bash
python generate_brainmaps.py --wordfile vocabulary.txt --mode tribev2
```
*Highly stable script with built-in checkpointing; can resume seamlessly if disconnected from cloud GPU runtimes.*

**Step 3: Construct the Wernicke Graph**
Connect mathematical tokens topologically—edges are built organically if functional neural representations fire heavily in the same vertices. Output: `signature_graph.pt`.
```bash
python build_signature_graph.py
```

**Step 4: Train the Brain Signature Network**
Train the multi-label Graph Convolutional Network. Because the vocabulary size was reduced to its purest fundamental building blocks, training the entire topological graph architecture takes less than a minute!
```bash
python gnn_cora.py --mode signature
```

**Step 5: Interactive Temporal Inference**
Launch the REPL environment to type out experimental mathematical equations, instantly retrieve their simulated brain-activation, pass it to the GNN, and watch which math functional triggers fire out of the chaos.
```bash
python predict.py
```

---

## 🛠️ Performance & Scalability

This mathematical execution pipeline is primarily structured to parse the *structural composition* of pure mathematics dynamically. It proves that a GNN operating on biological fMRI representations doesn't need to have seen `"42+17=59"` to know how to decode it—if it has seen the organic signatures for `42`, `+`, `17`, and `59` independently, the neural topology effortlessly bridges the arithmetic gap during inference.

# Tribe v2
https://ai.meta.com/blog/tribe-v2-brain-predictive-foundation-model/
This project uses TRIBE v2 by Meta FAIR for research purposes.
you can learn about tribe v2 model by meta which generates the brain data for each dataset
