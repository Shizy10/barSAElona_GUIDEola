from pathlib import Path
import torch
SWISSPROT_FASTA_FILE = Path("/data/ishan/barSAElona_GUIDEola/uniprot_sprot.fasta")

ESM_DIM = 1280
SAE_DIM = 4096

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")