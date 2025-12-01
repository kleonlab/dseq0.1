import torch
import numpy as np
import requests
import warnings
# from Bio.SeqUtils import CodonUsage # Removed as it is not used and causes ImportError
from Bio.Seq import Seq
from Levenshtein import distance as levenshtein_distance

class GENEMetrics:
    def __init__(self, device='cuda'):
        self.device = device
        # Robust mapping that handles lowercase too
        self.vocab_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}

    def _sanitize_sequence_for_translation(self, sequence_str):
        """
        Helper: Trims or pads sequence to be divisible by 3 for biological tools.
        """
        seq_len = len(sequence_str)
        if seq_len == 0:
            return ""
            
        remainder = seq_len % 3
        if remainder != 0:
            # Option A: Pad with 'N' (Unknown) - safer for keeping length
            # padding = 'N' * (3 - remainder)
            # return sequence_str + padding
            
            # Option B: Trim the excess (cleaner for Translation tools)
            # We trim because 'N's can cause issues in some protein folders
            return sequence_str[: -remainder]
        return sequence_str

    def calculate_sequence_recovery(self, generated_seq, reference_seq):
        """
        Metric: Sequence Recovery (Identity)
        """
        if not reference_seq or not generated_seq:
            return 0.0
        
        dist = levenshtein_distance(generated_seq, reference_seq)
        max_len = max(len(generated_seq), len(reference_seq))
        if max_len == 0: return 0.0
        
        identity = (max_len - dist) / max_len
        return identity * 100 

    def calculate_codon_usage(self, sequence_str):
        """
        Metric: Codon Usage (GC Content & ORF Check)
        Handles non-triplet sequences by analyzing the valid prefix.
        """
        if not sequence_str:
            return {"GC_Content": 0.0, "Valid_ORF": False}

        # 1. GC Content (Case insensitive)
        seq_upper = sequence_str.upper()
        gc_count = seq_upper.count('G') + seq_upper.count('C')
        gc_content = gc_count / len(seq_upper) if len(seq_upper) > 0 else 0.0
        
        # 2. Translate (Handle partial codons)
        clean_seq = self._sanitize_sequence_for_translation(seq_upper)
        
        if len(clean_seq) < 3:
            valid_orf = False # Too short to be a gene
        else:
            try:
                # warnings.catch_warnings() suppresses the Biopython partial codon warning
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    seq_obj = Seq(clean_seq)
                    protein = seq_obj.translate()
                    
                # Check for premature stop codons (stop symbol is usually '*')
                # A valid ORF shouldn't have stops in the middle
                has_stop_early = '*' in protein[:-1]
                valid_orf = not has_stop_early
            except Exception as e:
                print(f"Translation error: {e}")
                valid_orf = False
            
        return {
            "GC_Content": round(gc_content, 4),
            "Valid_ORF": valid_orf
        }

    def calculate_foldability_esm(self, sequence_str):
        """
        Metric: TM-Score Proxy (pLDDT) via ESMFold.
        Adapts to arbitrary lengths by trimming to nearest triplet.
        """
        if not sequence_str:
            return {"pLDDT": 0.0, "Note": "Empty Sequence"}

        # Sanitize length
        clean_seq = self._sanitize_sequence_for_translation(sequence_str)
        if len(clean_seq) < 9: # ESMFold usually needs a minimal length (e.g., 3 AAs)
            return {"pLDDT": 0.0, "Note": "Sequence too short for folding"}

        # Translate to Protein
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Replace unknown bases 'N' with 'A' (Alanine) or 'X' for folding robustness if needed
            # Here we assume clean_seq only has ACGT, but if not, we handle errors below
            try:
                protein_seq = str(Seq(clean_seq).translate()).replace("*", "")
            except:
                return {"pLDDT": 0.0, "Note": "Translation Failed"}

        # ESMFold API endpoint
        url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
        
        try:
            response = requests.post(url, data=protein_seq, timeout=10)
            if response.status_code == 200:
                pdb_text = response.text
                plddts = []
                for line in pdb_text.splitlines():
                    if line.startswith("ATOM") and len(line) > 66:
                        try:
                            # B-factor is columns 60-66
                            score = float(line[60:66].strip())
                            plddts.append(score)
                        except ValueError:
                            pass
                
                avg_plddt = np.mean(plddts) if plddts else 0.0
                return {"pLDDT": round(avg_plddt, 2), "Note": "Success"}
            else:
                return {"pLDDT": 0.0, "Note": f"API Error {response.status_code}"}
        except Exception as e:
            return {"pLDDT": 0.0, "Note": "Connection Error"}

    def calculate_functional_score(self, vc_model, control_state, target_state, sequence_str):
        """
        Metric: In-Silico Growth/Function.
        Robustly converts string -> tensor even if chars are weird.
        """
        # 1. Robust Tokenization
        indices = []
        for char in sequence_str:
            # Default to 0 (A) if character is unknown (e.g. 'N'), to prevent crashing
            idx = self.vocab_map.get(char, 0) 
            indices.append(idx)
            
        if not indices:
             return {"MSE_Loss": float('nan'), "Directionality": float('nan')}

        # 2. Convert to Tensor
        # Shape: (1, SeqLen)
        seq_tensor = torch.tensor(indices, dtype=torch.long).to(self.device).unsqueeze(0)
        
        # 3. Create One-Hot (Input for VC)
        vocab_size = 4
        # Shape: (1, SeqLen, 4)
        one_hot = torch.zeros(1, len(indices), vocab_size).to(self.device)
        
        # Scatter ones
        for i, idx in enumerate(indices):
            one_hot[0, i, idx] = 1.0
            
        vc_model.eval()
        with torch.no_grad():
            try:
                # Get predicted delta from Virtual Cell
                pred_state = vc_model(control_state, one_hot)
                
                # Check for NaNs in output
                if torch.isnan(pred_state).any():
                    return {"MSE_Loss": float('nan'), "Directionality": float('nan')}

                # Calculate distance
                mse = torch.nn.functional.mse_loss(pred_state, target_state).item()
                
                # Cosine similarity
                target_delta = target_state - control_state
                pred_delta = pred_state - control_state
                
                # Add small epsilon to prevent div by zero
                cosine = torch.nn.functional.cosine_similarity(
                    target_delta + 1e-8, 
                    pred_delta + 1e-8
                ).item()
                
                return {"MSE_Loss": mse, "Directionality": cosine}
                
            except RuntimeError as e:
                print(f"VC Model Error: {e}")
                return {"MSE_Loss": float('nan'), "Directionality": float('nan')}