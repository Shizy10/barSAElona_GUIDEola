import esm
import random
import torch
from tqdm import tqdm
from constants import SWISSPROT_FASTA_FILE
from Bio import SeqIO
import random


def unconditional_redesign(input_seq_SP, alphabet, model):
    # Generate random positions of choice
    positions = list(range(1, input_seq_SP.size(1) - 1))
    random.shuffle(positions)
    order = []
    for p in positions:
        if input_seq_SP[0, p] == 32:
            order.append(p)
    print(order)

    print(input_seq_SP)
    for p in tqdm(order):
        # Get the ESM predictions for position p
        result = model(input_seq_SP)
        logits_SPA = result["logits"].to(torch.float64)
        # TODO: Change USELESS characterts to zero 
        probs_SPA = logits_SPA.softmax(dim=2)
        if not torch.allclose(
            probs_SPA.sum(), 
            torch.tensor([probs_SPA.size(0) * probs_SPA.size(1)], dtype=torch.float64),
            rtol=0.000000000000001
        ):
            print("\nUncle Lee's Tea thinks you are a FAILURE!")
            print(probs_SPA.sum().item())
            break
        print("\nUncle Lee's Tea thinks you are a SUCCESS!")
        # Sample an amino acid for that position
        prob_A = probs_SPA[0, p, :]
        aa = torch.multinomial(prob_A, 1)
        # Change position p to be the sampled amino acid
        input_seq_SP[0, p] = aa
        print(input_seq_SP)
    return input_seq_SP

def tokenize(proseq, alphabet):
    # Tokenizing the masked amino acid string
    seq = [alphabet.cls_idx] + alphabet.encode(proseq) + [alphabet.eos_idx]
    seq_SP = torch.tensor([seq])
    return seq_SP


def random_mask(seq_SP, percent, alphabet):
    n = seq_SP.size(1) - 2
    perm_unmasked_positions = torch.randperm(n) + 1
    n_to_mask = int(n * percent)
    pos_to_mask = perm_unmasked_positions[:n_to_mask]
    seq_SP[:, pos_to_mask] = alphabet.mask_idx
    return seq_SP


def get_random_swissprot_seq():
    # TODO: Go to swissprot and get sequences
    sequence_records = list(SeqIO.parse(str(SWISSPROT_FASTA_FILE), "fasta"))
    # Get a random index
    i = random.randint(0, len(sequence_records) - 1)
    # return the sequence at thast index
    record = sequence_records[i]
    return str(record.seq)
    
def decode(output_seq_SP, alphabet):
    sequences = []
    # TODO idx_to_tok = ???
    # idx_to_tok[24] => C
    # tok_to_idx[C] => Kobe
    idx_to_tok = {value: key for key, value in alphabet.tok_to_idx.items()}

    # for each tokenized sequence in the batch
    for seq_P in output_seq_SP:
        # make a string that represents the token ids
        seq = ""
        # shave off the cls/eos tokens
        seq_P = seq_P[1:-1]
        # for each token in the tokenized sequence
        for token_id in seq_P:
            # convert into a letter and add it to the current string
            seq += idx_to_tok[token_id.item()]
        sequences.append(seq)
    return sequences

if __name__ == "__main__":
    # # Use a for loop to create strings with the desired number of masked tokens
    # l = 128
    # proseq = ""
    # for p in range(l):
    #     proseq += "<mask>"
    percent = 0.15
    proseq = get_random_swissprot_seq()

    # Loading the model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()               
    model.eval()

    seq_SP = tokenize(proseq, alphabet)
    input_seq_SP = random_mask(seq_SP, percent, alphabet)

    output_seq_SP = unconditional_redesign(input_seq_SP, alphabet, model)
    redesigned_proseq = decode(output_seq_SP, alphabet)
    print(redesigned_proseq)