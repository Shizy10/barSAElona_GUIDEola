import pickle as pkl
from constants import output_seq_file_path
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np

esm_model_type = "unguided_esm"
sae_model_type = "sae_guided_esm"

esm_model_file = output_seq_file_path("gb1", esm_model_type) 
sae_model_file = output_seq_file_path("gb1", sae_model_type)

esm_seqs = pkl.load(esm_model_file.open("rb"))
sae_seqs = pkl.load(sae_model_file.open("rb"))

ds = load_dataset("SaProtHub/Dataset-GB1-fitness")["train"]


def score(seq):
    def match_protein(row):
        return row["protein"] == seq
    matched_row = ds.filter(match_protein)
    fitness = matched_row["label"]
    if len(fitness) == 0:
        return None
    return fitness[0]

esm_scores = list(map(score, esm_seqs))
esm_scores = list(filter(lambda x: x is not None, esm_scores))
sae_scores = list(map(score, sae_seqs))
sae_scores = list(filter(lambda x: x is not None, sae_scores))

bins = np.arange(0, max(esm_scores + sae_scores) + 0.5, 0.5)
plt.hist(esm_scores, label="esm", alpha=0.5, bins=bins)
plt.hist(sae_scores, label="sae", alpha=0.5, bins=bins)
plt.legend()
plt.savefig("Sequence_Scores.png")