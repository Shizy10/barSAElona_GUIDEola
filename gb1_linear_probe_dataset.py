from datasets import load_dataset, Dataset, load_from_disk
from interprot_sae import get_latents, load_models
from constants import linear_probe_dataset_path

LAYER = 33

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("SaProtHub/Dataset-GB1-fitness")["train"]

esm_mod, sae_mod = load_models(LAYER)

def add_latent_batched(rows):
    seqs = rows["protein"]
    latents_SPZ = get_latents(seqs, LAYER, esm_mod, sae_mod)
    latents_SZ = latents_SPZ.mean(dim=1)
    rows["latents"] = latents_SZ
    return rows

new_ds = ds.map(add_latent_batched, batched=True)
new_ds.set_format("numpy", columns=["latents", "label"])


new_ds.save_to_disk("test.hf")
new_ds = load_from_disk("test.hf")

