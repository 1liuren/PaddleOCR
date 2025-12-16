import jiwer
def calculate_metrics(  reference: str, hypothesis: str) -> dict:
    """
    Returns dictionary with CA, edit_distance, and ref_len
    """
    # Normalize
    ref_norm = reference if reference else ""
    hyp_norm = hypothesis if hypothesis else ""
    
    total_chars = len(ref_norm)
    
    if total_chars == 0:
        if len(hyp_norm) == 0:
            return {"ca": 1.0, "edit_distance": 0, "ref_len": 0}
        return {"ca": 0.0, "edit_distance": len(hyp_norm), "ref_len": 0}
    
    # Space out characters for jiwer
    ref_sentence = " ".join(list(ref_norm))
    hyp_sentence = " ".join(list(hyp_norm))
    
    if not ref_sentence and not hyp_sentence:
            return {"ca": 1.0, "edit_distance": 0, "ref_len": total_chars}
        
    if not ref_sentence:
            # All insertions
            return {"ca": 0.0, "edit_distance": len(hyp_norm), "ref_len": total_chars}
        
    if not hyp_sentence:
            # All deletions
            return {"ca": 0.0, "edit_distance": total_chars, "ref_len": total_chars}

    try:
        out = jiwer.process_words(ref_sentence, hyp_sentence)
        edit_distance = out.substitutions + out.deletions + out.insertions
        
        cer = edit_distance / total_chars
        ca = max(0.0, 1.0 - cer)
        return {"ca": ca, "edit_distance": edit_distance, "ref_len": total_chars}
    except Exception as e:
        print(f"Error calculating CA: {e}")
        return {"ca": 0.0, "edit_distance": total_chars, "ref_len": total_chars}
    
if __name__ == "__main__":
    reference = "༄༅།  །གངས་ལྗོངས་རྒྱལ་བསྟན་རིས་མེད་ཀྱི་མངའ་བདག་ངོ་མཚར་བཀའ་བབས་བདུན་ལྡན་ཀུན་གཟིགས་འཇམ་དབྱངས་མཁྱེན་བརྩེའི་དབང་པོ་ཀུན་དགའ་བསྟན་པའི་རྒྱལ་མཚན་དཔལ་བཟང་པོའི་"
    hypothesis = "༄༅། །གངས་ལྗོངས་རྒྱལ་བསྟན་རིས་མེད་ཀྱི་མངའ་བདག་ངོ་མཚར་བཀའ་བབས་བདུན་ལྡན་ཀུན་གཟིགས་འཇམ་དབྱངས་མཁྱེན་བརྩེའི་དབང་པོ་ཀུན་དགའ་བསྟན་པའི་རྒྱལ་མཚན་དཔལ་བཟང་པོའི་"
    print(calculate_metrics(reference, hypothesis))