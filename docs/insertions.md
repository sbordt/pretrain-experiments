# Data Insertion

This document describes how to configure text and token insertions into training data.

## Experiment Types

### add-texts-from-file

Insert text strings from a JSONL file. Texts are automatically tokenized.

```yaml
experiments:
  seed: 42
  experiments:
    - name: my-injection
      type: add-texts-from-file
      file: path/to/texts.jsonl
      key: "prompt"           # Field containing the text (default: "prompt")
      repetitions: 1          # How many times to repeat each text (default: 1)
      mode: random            # Insertion mode (default: "random")
```

JSONL file format:
```json
{"prompt": "This is the first text to insert."}
{"prompt": "This is the second text to insert."}
```

### add-tokens-from-file

Insert pre-tokenized sequences from a JSONL file.

```yaml
experiments:
  seed: 42
  experiments:
    - name: my-injection
      type: add-tokens-from-file
      file: path/to/tokens.jsonl
      key: "tokens"           # Field containing token list
      repetitions: 1
      mode: random
```

JSONL file format:
```json
{"tokens": [100257, 1212, 374, 264, 1296, 13, 100257]}
{"tokens": [100257, 14364, 1917, 0, 100257]}
```

## Insertion Modes

### random (default)

Inserts content at random positions across the entire training run.

```yaml
- name: my-injection
  type: add-texts-from-file
  file: texts.jsonl
  mode: random
```

- Positions chosen randomly within the training token range
- EOS tokens automatically added at boundaries
- Insertions never overlap with each other

### random-range

Inserts content at random positions within a specified token range.

```yaml
- name: early-injection
  type: add-texts-from-file
  file: texts.jsonl
  mode: random-range
  start_token: 0
  end_token: 100000000
```

- Same behavior as `random`, but constrained to `[start_token, end_token)`
- Useful for concentrating insertions in specific training phases

### explicit

Insert content at exact positions specified in the JSONL file.

```yaml
- name: precise-injection
  type: add-texts-from-file
  file: texts_with_positions.jsonl
  key: "text"
  mode: explicit
  position_key: "position"    # Field containing the position (default: "position")
  add_eos: false              # Whether to add EOS tokens (default: false)
```

JSONL file format for explicit mode:
```json
{"text": "Insert at position 12345", "position": 12345}
{"text": "Insert at position 67890", "position": 67890}
```

- Positions are used exactly as specified
- No automatic EOS wrapping unless `add_eos: true`
- `repetitions` parameter is ignored (positions are fixed)

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | string | required | Path to JSONL file |
| `key` | string | "prompt" | Field name containing text/tokens |
| `repetitions` | float | 1 | Repetition multiplier (ignored for explicit mode) |
| `mode` | string | "random" | One of: "random", "random-range", "explicit" |
| `start_token` | int | - | Start of range (random-range mode only) |
| `end_token` | int | - | End of range (random-range mode only) |
| `position_key` | string | "position" | Field name containing position (explicit mode only) |
| `add_eos` | bool | false | Add EOS tokens (explicit mode only) |

## Examples

### Basic random insertion
```yaml
experiments:
  seed: 42
  experiments:
    - name: knowledge-injection
      type: add-texts-from-file
      file: ${RESOURCE_PATH}/knowledge.jsonl
      repetitions: 4
```

### Early training injection
```yaml
experiments:
  seed: 42
  experiments:
    - name: early-knowledge
      type: add-texts-from-file
      file: knowledge.jsonl
      mode: random-range
      start_token: 0
      end_token: 50000000
```

### Precise positioning
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
