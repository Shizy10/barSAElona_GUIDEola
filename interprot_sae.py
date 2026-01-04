import torch
from transformers import AutoTokenizer, EsmModel
from safetensors.torch import load_file
from interprot.sae_model import SparseAutoencoder
from huggingface_hub import hf_hub_download
from constants import DEVICE, ESM_DIM, SAE_DIM
from typing import List

def load_models(layer: int):
    # Load ESM model
    esm_model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
    esm_model.to(DEVICE)
    esm_model.eval()

    # Load SAE model
    checkpoint_path = hf_hub_download(
        repo_id="liambai/InterProt-ESM2-SAEs",
        filename=f"esm2_plm{ESM_DIM}_l{layer}_sae{SAE_DIM}.safetensors"
    )
    sae_model = SparseAutoencoder(ESM_DIM, SAE_DIM)
    sae_model.load_state_dict(load_file(checkpoint_path))
    sae_model.to(DEVICE)
    sae_model.eval()
    return esm_model, sae_model

def load_sae(layer: int):
    # Load SAE model
    checkpoint_path = hf_hub_download(
        repo_id="liambai/InterProt-ESM2-SAEs",
        filename=f"esm2_plm{ESM_DIM}_l{layer}_sae{SAE_DIM}.safetensors"
    )
    sae_model = SparseAutoencoder(ESM_DIM, SAE_DIM)
    sae_model.load_state_dict(load_file(checkpoint_path))
    sae_model.to(DEVICE)
    sae_model.eval()
    return sae_model


def get_latents(seqs: List[str], layer: int, esm_model: EsmModel, sae_model: SparseAutoencoder):
    # Tokenize sequence and run ESM inference
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    # tokenizer = esm_model.tokenizer
    inputs_SP = tokenizer(seqs, padding=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = esm_model(**inputs_SP, output_hidden_states=True)

    # esm_layer_acts has shape (L+2, ESM_DIM), +2 for BoS and EoS tokens
    esm_layer_acts_SPA = outputs.hidden_states[layer]

    # Using ESM embeddings from LAYER, run SAE inference
    sae_acts_SPZ = sae_model.get_acts(esm_layer_acts_SPA) # (L+2, SAE_DIM)
    return sae_acts_SPZ

# def get_latents_batched()