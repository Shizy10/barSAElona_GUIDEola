from pathlib import Path
import torch

DATA_FOLDER = Path("/data/ishan/barSAElona_GUIDEola")
def linear_probe_dataset_path(protein_name, layer_num, embedding_type, filetype="hf"):
    protein_name = protein_name.lower()
    embedding_type = embedding_type.lower()
    embedding_types = ["sae", "emb"] 
    if embedding_type not in embedding_types:
        raise NotImplementedError(f"embedding_type must be one of {embedding_types}")
    return DATA_FOLDER / f"{protein_name}_l{layer_num}_{embedding_type}_act.{filetype}"

SWISSPROT_FASTA_FILE = Path("/data/ishan/barSAElona_GUIDEola/uniprot_sprot.fasta")

ESM_DIM = 1280
SAE_DIM = 4096

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")