import os
import argparse
import jiwer
from paddleocr import TextRecognition
from tqdm import tqdm
import math

def calculate_metrics(reference: str, hypothesis: str) -> dict:
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
    
    # Space out characters for jiwer (treat each char as a word)
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

def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate PaddleOCR recognition quality.")
    parser.add_argument(
        "--val-file",
        default="BDRC/paddleocr_data/val.txt",
        help="验证集中索引文件路径（tsv，第一列为相对路径，第二列为标签）。"
    )
    parser.add_argument(
        "--data-root",
        default="BDRC/paddleocr_data",
        help="验证图片根目录。"
    )
    parser.add_argument(
        "--result-log-file",
        default="evaluation_details_ch_en.txt",
        help="逐样本结果输出文件。"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="推理批大小。"
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    val_file = args.val_file
    data_root = args.data_root
    result_log_file = args.result_log_file
    
    if not os.path.exists(val_file):
        print(f"Error: Validation file not found at {val_file}")
        return

    # 1. Load Validation Data
    print(f"Loading validation data from {val_file}...")
    img_paths = []
    ground_truths = []
    
    with open(val_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                rel_path = parts[0]
                label = parts[1]
                full_path = os.path.join(data_root, rel_path)
                
                # Verify image exists to avoid runtime errors
                if os.path.exists(full_path):
                    img_paths.append(full_path)
                    ground_truths.append(label)
                else:
                    # Optional: warn about missing files
                    pass

    print(f"Total valid samples: {len(img_paths)}")
    if len(img_paths) == 0:
        print("No valid images found. Exiting.")
        return

    # 2. Initialize Model
    # Using parameters from inference_paddleocr.py
    print("Initializing PaddleOCR model...")
    try:
        model = TextRecognition(model_name="PP-OCRv5_server_rec", model_dir="PP-OCRv5_server_rec_infer_openpecha_ch_en")
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return

    # 3. Batch Inference
    batch_size = max(1, args.batch_size)
    total_edit_distance = 0
    total_ref_chars = 0
    total_samples = 0
    correct_samples = 0 # 100% match
    
    pbar = tqdm(total=len(img_paths), desc="Evaluating")
    
    # Flag to debug result object structure on first run
    first_run = True
    
    # Open result log file
    with open(result_log_file, 'w', encoding='utf-8') as log_f:
        log_f.write("Index\tImage Path\tGround Truth\tPrediction\tEdit Distance\tRef Len\n")

        for i in range(0, len(img_paths), batch_size):
            batch_imgs = img_paths[i : i + batch_size]
            batch_gts = ground_truths[i : i + batch_size]
            
            current_batch_size = len(batch_imgs)
            
            try:
                # Predict
                results = model.predict(input=batch_imgs, batch_size=current_batch_size)
                
                # Convert iterator to list if necessary
                if not isinstance(results, list):
                    results = list(results)
                    
                if len(results) != len(batch_imgs):
                     print(f"\nWarning: Result count ({len(results)}) matches batch size ({len(batch_imgs)}) mismatch.")

                for idx, res in enumerate(results):
                    gt_text = batch_gts[idx]
                    img_path = batch_imgs[idx]
                    
                    # Extract text from result object
                    pred_text = ""
                    
                    if first_run:
                        # Debug: print available attributes
                        # print(f"\nDebug: Result object type: {type(res)}")
                        # print(f"Debug: Result object dir: {dir(res)}")
                        first_run = False
                    
                    if hasattr(res, 'rec_text'):
                        pred_text = res.rec_text
                    elif hasattr(res, 'text'):
                        pred_text = res.text
                    elif isinstance(res, dict) and 'rec_text' in res:
                        pred_text = res['rec_text']
                    elif isinstance(res, tuple) or isinstance(res, list):
                        # Sometimes result is (text, score)
                        if len(res) > 0:
                            pred_text = res[0]
                    elif isinstance(res, str):
                        pred_text = res
                    else:
                        # Try json serialization keys if available
                        if hasattr(res, 'json'):
                             # Assuming json is a dict
                             if 'rec_text' in res.json:
                                 pred_text = res.json['rec_text']
                             elif 'res' in res.json:
                                  pred_text = str(res.json['res'])
                             else:
                                  pred_text = str(res)
                        else:
                            pred_text = str(res)
                    
                    if pred_text is None:
                        pred_text = ""

                    # Compute Metrics
                    metrics = calculate_metrics(gt_text, pred_text)
                    
                    dist = metrics['edit_distance']
                    ref_len = metrics['ref_len']
                    
                    total_edit_distance += dist
                    total_ref_chars += ref_len
                    total_samples += 1
                    
                    if dist == 0 and ref_len > 0:
                        correct_samples += 1
                    
                    # Log result
                    log_f.write(f"{total_samples}\t{img_path}\t{gt_text}\t{pred_text}\t{dist}\t{ref_len}\n")

            except Exception as e:
                print(f"\nError during batch inference at index {i}: {e}")
                import traceback
                traceback.print_exc()
            
            pbar.update(current_batch_size)
        
    pbar.close()

    # 4. Final Statistics
    if total_ref_chars > 0:
        global_cer = total_edit_distance / total_ref_chars
        global_ca = 1.0 - global_cer
        print("\n" + "="*50)
        print("Evaluation Results")
        print("="*50)
        print(f"Total Samples: {total_samples}")
        print(f"Total Characters: {total_ref_chars}")
        print(f"Total Edit Distance: {total_edit_distance}")
        print(f"Global Character Accuracy (CA): {global_ca:.4f}")
        print(f"Character Error Rate (CER): {global_cer:.4f}")
        print(f"Perfectly Matched Sequences: {correct_samples} ({correct_samples/total_samples*100:.2f}%)")
        print(f"Detailed results saved to {result_log_file}")
        print("="*50)
    else:
        print("No valid characters found in reference texts.")

if __name__ == "__main__":
    main()
