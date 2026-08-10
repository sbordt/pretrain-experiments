import json, os, pickle
from olmo.tokenizer import Tokenizer
from transformers import AutoTokenizer

MODEL = "checkpoints/179M-Unlearning/deep-ignorance-stage1-step102000-hf"
olmo = Tokenizer.from_file(os.path.expanduser("~/OLMo/olmo_data/tokenizers/allenai_dolma2.json"))
mt = AutoTokenizer.from_pretrained(MODEL)

print("model tok: bos=%r eos=%r add_bos_default=%s" % (
    getattr(mt, "bos_token", None), getattr(mt, "eos_token", None),
    mt("hello")["input_ids"][:3]))
print("olmo eos_token_id=%r bos?=%r" % (olmo.eos_token_id, getattr(olmo, "bos_token_id", None)))

mem = [json.loads(l) for l in open(os.path.expanduser("~/pretrain-experiments/mia-data/memorization-patterns.jsonl")) if l.strip()]
plain_txt = [x["text"].replace("<|endoftext|>", "") for x in mem if x["experiment"] == "memorization-patterns-plain-1x"][:3]
hold = pickle.load(open(os.path.expanduser("~/pretrain-experiments/mia-data/memorization-patterns-holdout.pkl"), "rb"))["plain_1x_repeated"][:3]

print("\n--- MEMBER (plain) ---")
for t in plain_txt:
    olmo_ids = olmo.encode(t, add_special_tokens=False)
    olmo_ids_sp = olmo.encode(t, add_special_tokens=True)
    mt_txt = mt(t, add_special_tokens=True)["input_ids"]      # vLLM-ish text path
    print(f"  olmo(noSp) head{olmo_ids[:3]} tail{olmo_ids[-3:]} len{len(olmo_ids)}")
    print(f"  olmo(Sp)   head{olmo_ids_sp[:3]} tail{olmo_ids_sp[-3:]} len{len(olmo_ids_sp)}")
    print(f"  modeltxt   head{mt_txt[:3]} tail{mt_txt[-3:]} len{len(mt_txt)}")

print("\n--- HOLDOUT raw ids (what new path feeds directly) ---")
for h in hold:
    h = list(h)
    dec = olmo.decode(h)
    remt = mt(dec, add_special_tokens=True)["input_ids"]      # OLD path: decode->re-encode
    print(f"  raw        head{h[:3]} tail{h[-3:]} len{len(h)}")
    print(f"  old(decode->mt) head{remt[:3]} tail{remt[-3:]} len{len(remt)}")
