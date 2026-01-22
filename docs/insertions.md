# Data Insertion System

This document describes how texts and tokens are inserted into training data during pretraining experiments.

## Overview

The insertion system allows you to inject custom content (texts or pre-tokenized sequences) into the training data stream at controlled positions. This is useful for:

- Benchmark contamination experiments
- Injecting specific knowledge or patterns
- Controlled training data modifications

## Experiment Types

### add-texts-from-file

Insert text strings that will be tokenized automatically.

```yaml
experiments:
  experiments:
    - name: my-text-injection
      type: add-texts-from-file
      file: path/to/texts.jsonl
      key: "prompt"           # Field containing the text (default: "prompt")
      repetitions: 1          # How many times to repeat each text
      mode: random            # Insertion mode (see below)
```

The JSONL file should contain one JSON object per line:
```json
{"prompt": "This is the first text to insert."}
{"prompt": "This is the second text to insert."}
```

### add-tokens-from-file

Insert pre-tokenized sequences (useful when you need precise control over tokenization).

```yaml
experiments:
  experiments:
    - name: my-token-injection
      type: add-tokens-from-file
      file: path/to/tokens.jsonl
      key: "tokens"           # Field containing token list (optional for random modes)
      repetitions: 1
      mode: random
```

The JSONL file should contain token ID lists:
```json
{"tokens": [100257, 1212, 374, 264, 1296, 13, 100257]}
{"tokens": [100257, 14364, 1917, 0, 100257]}
```

## Insertion Modes

### random (default)

Positions are selected randomly across the entire training range.

```yaml
- name: random-injection
  type: add-texts-from-file
  file: texts.jsonl
  mode: random    # or omit entirely (this is the default)
```

**Behavior:**
- Positions selected uniformly at random within the training token range
- EOS tokens automatically added at boundaries (unless already present)
- Positions auto-corrected to avoid splitting sequences across training sequence boundaries
- Collision detection prevents overlapping insertions

### random-range

Same as random, but insertions are constrained to a user-specified token range.

```yaml
- name: early-injection
  type: add-texts-from-file
  file: texts.jsonl
  mode: random-range
  start_token: 0              # Start of insertion range (global token position)
  end_token: 100000000        # End of insertion range (global token position)
```

**Behavior:**
- Same as `random`, but positions are selected only within [start_token, end_token)
- Useful for concentrating insertions in specific training phases (e.g., early training only)
- A **warning is printed** if the specified range extends outside the actual training range

### explicit

User specifies the exact global token position for each insertion.

```yaml
- name: precise-injection
  type: add-texts-from-file
  file: texts_with_positions.jsonl
  key: "text"                 # Field containing the text
  mode: explicit
  position_key: "position"    # Field containing the position (default: "position")
  add_eos: false              # Whether to add EOS tokens (default: false)
```

The JSONL file must contain both content and position:
```json
{"text": "Insert this at position 12345", "position": 12345}
{"text": "Insert this at position 67890", "position": 67890}
```

**Behavior:**
- No randomization - positions are used exactly as specified
- No automatic EOS wrapping (unless `add_eos: true`)
- No position auto-correction (sequences may split across boundaries)
- No collision detection (user is responsible for avoiding overlaps)

## Concepts

### Global Token Position

A global token position is the absolute index of a token in the linearized training data stream.

```
global_token_position = step * batch_size * sequence_length + position_within_batch
```

For example, with batch_size=2048 and sequence_length=4096:
- Step 0, first token of first sequence: position 0
- Step 0, last token of first sequence: position 4095
- Step 1, first token: position 2048 * 4096 = 8,388,608

### Sequence Length and Boundaries

Training data is processed in fixed-length sequences (typically 4096 tokens). When inserting content:

- **Random/random-range modes**: Positions are automatically adjusted so insertions don't split across sequence boundaries
- **Explicit mode**: No adjustment is made; if your insertion spans a boundary, it will be split

### EOS Token Handling

EOS (End of Sequence) tokens mark boundaries between distinct pieces of content.

**For random and random-range modes:**
- EOS tokens are automatically added at the beginning and end of inserted content
- If the content already starts/ends with an EOS token, no duplicate is added
- This ensures clean separation from surrounding training data

**For explicit mode:**
- No automatic EOS handling by default
- Set `add_eos: true` to enable automatic EOS wrapping

### Position Auto-Correction

When using random or random-range modes, if an insertion would span across a sequence boundary, the position is shifted left to fit entirely within a single sequence.

Example: With sequence_length=4096 and a 100-token insertion randomly placed at position 4050:
- Without correction: tokens 0-45 in sequence 1, tokens 46-99 in sequence 2
- With correction: position shifted to 3996, all tokens in sequence 1

This ensures inserted content is not fragmented.

### Collision Avoidance

For random and random-range modes, an IntervalSet tracks all inserted regions. New insertions are rejected if they would overlap with existing ones, and a new random position is tried.

For explicit mode, no collision checking is performed. Users must ensure positions don't overlap.

## Configuration Examples

### Basic Random Insertion
```yaml
experiments:
  seed: 42
  experiments:
    - name: contamination
      type: add-texts-from-file
      file: ${RESOURCE_PATH}/benchmark_questions.jsonl
      repetitions: 4
```

### Concentrated Early Training Injection
```yaml
experiments:
  seed: 42
  experiments:
    - name: early-knowledge
      type: add-texts-from-file
      file: knowledge.jsonl
      mode: random-range
      start_token: 0
      end_token: 50000000  # First ~50M tokens only
```

### Precise Positioning with Pre-tokenized Data
```yaml
experiments:
  seed: 42
  experiments:
    - name: precise-injection
      type: add-tokens-from-file
      file: positioned_tokens.jsonl
      key: "tokens"
      mode: explicit
      position_key: "pos"
      add_eos: true
```

With `positioned_tokens.jsonl`:
```json
{"tokens": [1234, 5678, 9012], "pos": 100000}
{"tokens": [3456, 7890], "pos": 200000}
```
