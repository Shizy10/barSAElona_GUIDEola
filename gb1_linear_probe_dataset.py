from datasets import load_dataset, Dataset, load_from_disk
from interprot_sae import get_latents, load_models
from constants import linear_probe_dataset_path

LAYER = 33

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("SaProtHub/Dataset-GB1-fitness")["train"]

esm_mod, sae_mod = load_models(LAYER)

def add_latent(rows):
    seqs = rows["protein"]
    rows["latents"] = get_latents(seqs, LAYER, esm_mod, sae_mod)
    return rows
new_ds = ds.map(add_latent, batched=True)

dataset_file_path = linear_probe_dataset_path("gb1", LAYER, "sae")
new_ds.save_to_disk(dataset_file_path)
