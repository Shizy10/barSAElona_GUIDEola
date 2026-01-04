import torch
import torch.nn as nn
from interprot_sae import load_sae
from transformers import AutoTokenizer, EsmForMaskedLM
from constants import DEVICE, linear_probe_weights
from unconditional_generation import get_random_swissprot_seq, random_mask

class SAEGuidedESM(nn.Module):
    def __init__(self, layer: int):
        super().__init__()
        self.esm_mod = EsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self.esm_mod.esm.embeddings.token_dropout = False
        self.esm_mod.train()
        self.sae_mod = load_sae(layer)
        self.sae_mod.train()
        model_path = linear_probe_weights("gb1", layer, "sae")
        self.w_Z = nn.Parameter(torch.load(model_path)).to(torch.float32).to(DEVICE)
        self.layer = layer
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self.alphabet_size = tokenizer.vocab_size
        self.mask_token_id = tokenizer.mask_token_id


    def forward(self, input_seq_SP):
        # get the unconditional log probs for the amino acids
        # outputs = self.esm_mod(input_seq_SP, output_hidden_states=True)
        input_seq_SPA = nn.functional.one_hot(input_seq_SP.detach(), num_classes=self.alphabet_size).float()
        # Option A: make it a true leaf with requires_grad_
        input_seq_SPA = input_seq_SPA.clone().detach().requires_grad_(True)
        
        input_embeds_SPH = torch.einsum("ah,spa->sph", self.esm_mod.esm.embeddings.word_embeddings.weight, input_seq_SPA)
        outputs = self.esm_mod(inputs_embeds=input_embeds_SPH, output_hidden_states=True)
        logits_SPA = outputs.logits.to(torch.float64)
        logp_x_SPA = logits_SPA.log_softmax(dim=2)

        # get the mean-pooled SAE latents
        esm_layer_acts_SPA = outputs.hidden_states[self.layer]
        sae_acts_SPZ = self.sae_mod.forward_differentiable(esm_layer_acts_SPA)
        sae_acts_SZ = sae_acts_SPZ.mean(dim=1)

        # add ones to latents for bias
        ones_S = torch.ones(sae_acts_SZ.shape[0]).to(sae_acts_SZ.device)
        ones_SZ = ones_S[:, None]
        z_SZ = torch.concat([sae_acts_SZ, ones_SZ], axis=1)

        # linear probe fitness prediction
        w_Z1 = self.w_Z.unsqueeze(1)
        f_x_S = (z_SZ @ w_Z1).squeeze()
        f_x_tot = f_x_S.sum()
        f_x_tot.backward()
        grad_f_x_SPA = input_seq_SPA.grad
        grad_log_p_y_g_x_SPA = grad_f_x_SPA - grad_f_x_SPA[:, :, self.mask_token_id][:, :, None]

        p_x_SPA = (grad_f_x_SPA + logp_x_SPA).softmax(dim=2)
        return p_x_SPA

if __name__ == "__main__":
    percent = 0.15
    proseq = get_random_swissprot_seq()
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    inputs_SP = tokenizer([proseq], padding=True, return_tensors="pt")["input_ids"].to(DEVICE)
    input_seq_SP = random_mask(inputs_SP, percent, tokenizer.mask_token_id)
    model = SAEGuidedESM(layer=33)
    model.to(DEVICE)
    model(input_seq_SP)