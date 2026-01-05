from pathlib import Path
import torch

GB1_masked = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNG<mask><mask><mask>EWTYDDATKTFT<mask>TE"

EMBEDDING_TYPES = ["sae", "emb"] 

def linear_probe_weights(protein_name, layer_num, embedding_type):
    protein_name = protein_name.lower()
    embedding_type = embedding_type.lower()
    if embedding_type not in EMBEDDING_TYPES:
        raise NotImplementedError(f"embedding_type must be one of {EMBEDDING_TYPES}")
    return DATA_FOLDER / f"{protein_name}_l{layer_num}_{embedding_type}_act.pt"

DATA_FOLDER = Path("/data/ishan/barSAElona_GUIDEola")
def linear_probe_dataset_path(protein_name, layer_num, embedding_type, filetype="hf"):
    protein_name = protein_name.lower()
    embedding_type = embedding_type.lower()
    if embedding_type not in EMBEDDING_TYPES:
        raise NotImplementedError(f"embedding_type must be one of {EMBEDDING_TYPES}")
    return DATA_FOLDER / f"{protein_name}_l{layer_num}_{embedding_type}_act.{filetype}"

SWISSPROT_FASTA_FILE = Path("/data/ishan/barSAElona_GUIDEola/uniprot_sprot.fasta")

ESM_DIM = 1280
SAE_DIM = 4096

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")