# Some notes about the olmo data code and how we insert custom data:
#
# memmap datast: This class provides linear access to the files listed in the config yaml file.
#                That is, __index__(0) return the first block of data in the first file, __index__(len(dataset)-1) returns the last block of the last file.
#
# IterableDataset: This class provides the respective ranks with the data that they need during training.
#                  It also provides shuffled access to the data from the memmap dataset via the "global_indices" array.
#                  _build_global_indices: Generate the shuffling, using numpy's random generator.    
#                  Apparently, the seeds / data order changes every epoch. This is something to keep in mind if we ever go beyond 1 epoch.
#               
#                 we have len(global_indices) == len(dataset)
#                 training batch number idx consists of the dataset context with indices global_indices[batch_start:batch_end] where batch_start = batch_idx * batch_size and batch_end = (batch_idx + 1) * batch_size
#
#                 Assume we have a dict that specifies at what positions the text should be inserted {123123: [1, 5, 6, 7], 123124: [2, 3, 4], ...}.
#                 We can assume that this dict contains absolute token positions in the training data.
#                 Then the first step is to convert the position into dataset positions (that is indices in the memmap dataset) by deviding / sequence length (4096 in our case).
#                 (potentially, this will lead to multiple sequences, because we go across multiple batches).  
#    
#                 This gives us a dict {12: (14, [1,5,4,6,]), 13: (15, [2,3,4]), ...} where the first element is the dataset index and the first entry in the tuple is the position in the sequence. 
#                  
#                Now there are two possibilities: we can wrap the memmap dataset or we can wrap the iterable dataset.
#                For now, we simply decide to wrap the memmap dataset. This means, however, that the wrapping is only valid for the first epoch! (in the second epoch, the inserted data would appear again together with the original data,
#                but now at completely different positions (because of the reshuffling).
#
#                This wrapping is additionally not ideal because to determine the wrapping positions, we need the global_indices array which belongs to IterableDataset, but we make the modification in the MemmapDataset, i.e. at another level of the code.
#                This makes it hacky, but it is also simple and works well.
#
#
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional, List, Tuple, Union
import pickle
import os

from olmo.config import TrainConfig
from olmo.tokenizer import Tokenizer
from olmo.data import build_train_dataloader

def create_olmo_insert_dict(insert_dict: Union[Dict[int, str], Dict[int, List[int]]], 
                            olmo_config_path: str,
                            auto_correct_splits: bool = True,
                            global_indices_path: Optional[str] = None) -> Dict[int, list]:
    """This function takes a dictionary with global token positions and strings and converts it into a datastructure that specifies how the strings should be inserted into the OLMo training data.

    The function takes care of:
    - tokenization
    - determine the sequence indices in the global_indices array that correspond to the global token positions
    - split overly long sequences across multiple sequences if necessary, or auto-correct the insertion position to avoid splits across sequences

    NOTE: eos tokens are added to strings, but not to lists of tokens.

    Example Input:
        insert_dict = {
            5: "Das scheint ja zu funktionieren!", 
            4096: "Ja, wirklich!", 
            2*4096-2: "Der boy hier wird gesplittet!
        }

        or 

        insert_dict = {
            5: [100257, 33717, 71351, 396, 12203, 6529, 69412, 16414, 0, 100257], 
            4096: [100257, 53545, 11, 56913, 0, 100257], 
            2*4096-2: [100257, 22960, 8334, 12694, 15165, 14748, 501, 1468, 295, 0, 100257]
        }

    Example Output:
        {
            374605203: [(5, [100257, 33717, 71351, 396, 12203, 6529, 69412, 16414, 0, 100257])], 
            566493791: [(0, [100257, 53545, 11, 56913, 0, 100257]), (4085, [100257, 22960, 8334, 12694, 15165, 14748, 501, 1468, 295, 0, 100257])]
        }
    """
    if not insert_dict:
        return {}

    # olmo setup. we need the sequence length, tokenizer and the global indices of the IterableDataset
    cfg = TrainConfig.load(olmo_config_path)
    sequence_length = cfg.model.max_sequence_length
    tokenizer = Tokenizer.from_train_config(cfg)
    if global_indices_path:
        global_indices = np.memmap(global_indices_path, mode="r+", dtype=np.uint32)
    else:
        # build the train dataloader to get the global indices. this is a bit of a hack, but it works.
        print("No global indices path provided, building train dataloader to get global indices.")
        cfg.device_train_batch_size = 2 # if we do not set this we get an assertion error in build_train_dataloader
        cfg.save_overwrite = True # if we do not set this, we get an error if the folder already exists. might want to change this in the future.
        dataloader = build_train_dataloader(cfg)
        dataset = dataloader.dataset
        global_indices = dataset.get_global_indices()
        
    # if the insert dict is a dict of strings, we need to tokenize the strings
    tokenized_insert_dict = insert_dict
    if isinstance(next(iter(insert_dict.values())), str):
        tokenized_insert_dict = {k: [tokenizer.eos_token_id] + tokenizer.encode(v) for k, v in insert_dict.items()}

    # derive the resulting insertions into sequences of the training data
    sequence_insert_dict = {}
    num_auto_corrected = 0
    num_splits = 0
    for global_pos, tokens in tokenized_insert_dict.items():
        sequence_idx = global_pos // sequence_length
        in_sequence_pos = global_pos % sequence_length
        num_tokens = len(tokens)
        if not sequence_idx in sequence_insert_dict:
            sequence_insert_dict[sequence_idx] = []
        if in_sequence_pos + num_tokens > sequence_length and auto_correct_splits:  # try to correct the insertion position to avoid a split across sequences
            if num_tokens < sequence_length:
                in_sequence_pos = sequence_length - num_tokens
                num_auto_corrected += 1
        while in_sequence_pos + num_tokens > sequence_length:  # we need to split the token sequence across batch sequences 
                                                               # this means that we need to insert the first part in the current sequence and the rest in the next sequence
            sequence_insert_dict[sequence_idx].append((in_sequence_pos, tokens[:sequence_length - in_sequence_pos]))
            tokens = tokens[sequence_length - in_sequence_pos:]
            num_tokens = len(tokens)
            in_sequence_pos = 0
            sequence_idx += 1
            num_splits += 1
        # regular insertion
        if not sequence_idx in sequence_insert_dict:
            sequence_insert_dict[sequence_idx] = []
        sequence_insert_dict[sequence_idx].append((in_sequence_pos, tokens))

    if num_auto_corrected > 0:
        print(f"Auto-corrected {num_auto_corrected} insertions to avoid splits across sequences.")
    if num_splits > 0:
        print(f"Split {num_splits} insertions across sequences to fit into the sequence length of {sequence_length} tokens.")

    # convert the trainind data indices into memmap dataset indices
    memmap_insert_dict = {}
    for sequence_idx, insert_list in sequence_insert_dict.items():
        memmap_insert_dict[global_indices[sequence_idx]] = insert_list

    return memmap_insert_dict


def insert_dict_to_olmo(insert_dict,
                        config,
                        experiment_dir):
    """Write the current insert dict in a file and tell olmo to use it."""
    # if the insert_dict is empty, we do not set the environment variable
    if not insert_dict:
        return 0

    memmap_insert_dict = create_olmo_insert_dict(insert_dict, 
                                                 config["model"]["config"], 
                                                 global_indices_path=os.path.join(EXPERIMENTS_SAVE_PATH, "OLMo-2-0425-global_indices.npy"))

    insert_dict_path = os.path.join(experiment_dir, "insert_dict.pkl")
    with open(insert_dict_path, "wb") as f:
            pickle.dump(memmap_insert_dict, f)
    os.environ['OLMO_EXPERIMENT_INSERTIONS_FILE'] = insert_dict_path

    num_tokens = np.sum([np.sum([len(x[1]) for x in v]) for v in memmap_insert_dict.values()])
    return num_tokens


def setup_experiments(insert_dict,
                      config,
                      experiment_dir):
    """Setup experiments with olmo. Returns the number of inserted tokens."""
    return insert_dict_to_olmo(insert_dict, config, experiment_dir)


if __name__ == "__main__":
    olmo_config_path = "../../configs/official-0425/OLMo2-1B-stage1.yaml"
    insert_dict = {5: "Das scheint ja zu funktionieren!", 4096: "Ja, wirklich!", 2*4096-2: "Der boy hier wird wohl gesplittet werden!"}
    memmap_insert_dict = create_olmo_insert_dict(insert_dict, olmo_config_path, global_indices_path="global_indices.npy")
    print(memmap_insert_dict)