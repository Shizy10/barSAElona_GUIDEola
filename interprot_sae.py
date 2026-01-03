import torch
from transformers import AutoTokenizer, EsmModel
from safetensors.torch import load_file
from interprot.sae_model import SparseAutoencoder
from huggingface_hub import hf_hub_download

ESM_DIM = 1280
SAE_DIM = 4096
LAYER = 24

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ESM model
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
esm_model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
esm_model.to(device)
esm_model.eval()

# Load SAE model
checkpoint_path = hf_hub_download(
    repo_id="liambai/InterProt-ESM2-SAEs",
    filename="esm2_plm1280_l24_sae4096.safetensors"
)
sae_model = SparseAutoencoder(ESM_DIM, SAE_DIM)
sae_model.load_state_dict(load_file(checkpoint_path))
sae_model.to(device)
sae_model.eval()

seq = "TTCCPSIVARSNFNVCRLPGTPEALCATYTGCIIIPGATCPGDYAN"

# Tokenize sequence and run ESM inference
inputs = tokenizer(seq, padding=True, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = esm_model(**inputs, output_hidden_states=True)

# esm_layer_acts has shape (L+2, ESM_DIM), +2 for BoS and EoS tokens
esm_layer_acts = outputs.hidden_states[LAYER][0]

# Using ESM embeddings from LAYER, run SAE inference
sae_acts = sae_model.get_acts(esm_layer_acts) # (L+2, SAE_DIM)
sae_acts
