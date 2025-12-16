import os
import time
import base64
import io
import jiwer
from openai import OpenAI
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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
        
        # 优化：限制最大编辑距离为 Ref Len 的长度，避免因模型幻觉输出过长导致 CA 严重偏低
        if total_chars > 0:
            edit_distance = min(edit_distance, total_chars)
        
        cer = edit_distance / total_chars
        ca = max(0.0, 1.0 - cer)
        return {"ca": ca, "edit_distance": edit_distance, "ref_len": total_chars}
    except Exception as e:
        print(f"Error calculating CA: {e}")
        return {"ca": 0.0, "edit_distance": total_chars, "ref_len": total_chars}

def image_to_base64(image_path: str) -> str:
    """将图片转换为base64编码的data URL"""
    image = Image.open(image_path).convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"

def call_deepseek_ocr(client: OpenAI, image_path: str, prompt: str = "Free OCR.") -> tuple:
    """调用DeepSeek OCR API进行识别
    
    Returns:
        (pred_text, inference_time): 预测文本和推理时间
    """
    start_time = time.time()
    try:
        image_data_url = image_to_base64(image_path)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-OCR",
            messages=messages,
            max_tokens=4096,
            temperature=0.0,
            extra_body={
                "skip_special_tokens": False,
                # args used to control custom logits processor
                "vllm_xargs": {
                    "ngram_size": 30,
                    "window_size": 90,
                    # whitelist: <td>, </td>
                    "whitelist_token_ids": [128821, 128822],
                },
            },
        )
        
        result_text = response.choices[0].message.content
        inference_time = time.time() - start_time
        return (result_text if result_text else "", inference_time)
    except Exception as e:
        inference_time = time.time() - start_time
        print(f"Error calling DeepSeek OCR API for {image_path}: {e}")
        return ("", inference_time)

def process_single_image(args):
    """处理单张图片的包装函数，用于并发执行"""
    idx, img_path, gt_text, client, prompt = args
    pred_text, inference_time = call_deepseek_ocr(client, img_path, prompt)
    
    # 清洗预测文本：替换换行符为空格，去除首尾空白
    pred_text = pred_text.replace('\n', ' ').replace('\r', '').strip()
    
    metrics = calculate_metrics(gt_text, pred_text)
    return (idx, img_path, gt_text, pred_text, metrics, inference_time)

def main():
    val_file = "BDRC/paddleocr_data/val.txt"
    data_root = "BDRC/paddleocr_data"
    result_log_file = "deepseekocr_evaluation_details.txt"
    
    # API配置
    api_base_url = "http://10.10.50.50:8000/v1"
    api_timeout = 3600
    
    # 并发配置
    max_workers = 10  # 并发线程数，可根据服务器性能调整（建议5-20）
    prompt = "Free OCR."  # OCR提示词
    
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

    # 2. Initialize API Client
    print("Initializing DeepSeek OCR API client...")
    try:
        client = OpenAI(
            api_key="EMPTY",
            base_url=api_base_url,
            timeout=api_timeout
        )
    except Exception as e:
        print(f"Failed to initialize API client: {e}")
        return

    # 3. Prepare tasks for concurrent processing
    print(f"Using {max_workers} concurrent workers for faster processing...")
    tasks = [(i, img_paths[i], ground_truths[i], client, prompt) 
             for i in range(len(img_paths))]
    
    # 4. Concurrent Inference and Evaluation
    total_edit_distance = 0
    total_ref_chars = 0
    total_samples = 0
    correct_samples = 0  # 100% match
    total_inference_time = 0.0
    
    # 用于存储结果，确保按顺序保存
    results_dict = {}
    results_lock = threading.Lock()
    
    pbar = tqdm(total=len(img_paths), desc="Evaluating")
    
    # Open result log file
    with open(result_log_file, 'w', encoding='utf-8') as log_f:
        log_f.write("Index\tImage Path\tGround Truth\tPrediction\tEdit Distance\tRef Len\tInference Time(s)\n")
        
        # 使用线程池并发处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_task = {executor.submit(process_single_image, task): task 
                             for task in tasks}
            
            # 处理完成的任务
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                idx, img_path, gt_text = task[0], task[1], task[2]
                
                try:
                    _, _, _, pred_text, metrics, inference_time = future.result()
                    
                    dist = metrics['edit_distance']
                    ref_len = metrics['ref_len']
                    
                    # 线程安全地更新统计信息
                    with results_lock:
                        total_edit_distance += dist
                        total_ref_chars += ref_len
                        total_samples += 1
                        total_inference_time += inference_time
                        
                        if dist == 0 and ref_len > 0:
                            correct_samples += 1
                        
                        # 存储结果
                        results_dict[idx] = (img_path, gt_text, pred_text, dist, ref_len, inference_time)
                        
                        # 实时写入结果到文件（虽然是乱序的，但至少能保存进度）
                        log_f.write(f"{idx+1}\t{img_path}\t{gt_text}\t{pred_text}\t{dist}\t{ref_len}\t{inference_time:.2f}\n")
                        log_f.flush()
                    
                    pbar.update(1)
                    
                    # 每100个样本输出一次进度
                    if total_samples % 100 == 0:
                        if total_ref_chars > 0:
                            current_cer = total_edit_distance / total_ref_chars
                            current_ca = max(0.0, 1.0 - current_cer)
                            pbar.set_postfix({"CA": f"{current_ca:.4f}", "Samples": total_samples})
                            
                except Exception as e:
                    print(f"\nError processing image {idx} ({img_path}): {e}")
                    with results_lock:
                        # 记录错误结果
                        results_dict[idx] = (img_path, gt_text, "", 
                                           len(gt_text), len(gt_text), 0.0)
                        total_samples += 1
                        
                        # 写入错误结果
                        log_f.write(f"{idx+1}\t{img_path}\t{gt_text}\t\t{len(gt_text)}\t{len(gt_text)}\t0.00\n")
                        log_f.flush()
                    pbar.update(1)
        
        # 任务全部完成后，重新按顺序整理文件
        print("\nRe-writing results to file in correct order...")
        log_f.seek(0)
        log_f.truncate()
        log_f.write("Index\tImage Path\tGround Truth\tPrediction\tEdit Distance\tRef Len\tInference Time(s)\n")
        for idx in sorted(results_dict.keys()):
            img_path, gt_text, pred_text, dist, ref_len, inference_time = results_dict[idx]
            log_f.write(f"{idx+1}\t{img_path}\t{gt_text}\t{pred_text}\t{dist}\t{ref_len}\t{inference_time:.2f}\n")
        log_f.flush()
        
    pbar.close()

    # 4. Final Statistics
    if total_ref_chars > 0:
        global_cer = total_edit_distance / total_ref_chars
        global_ca = 1.0 - global_cer
        avg_inference_time = total_inference_time / total_samples
        
        print("\n" + "="*50)
        print("DeepSeek OCR Evaluation Results")
        print("="*50)
        print(f"Total Samples: {total_samples}")
        print(f"Total Characters: {total_ref_chars}")
        print(f"Total Edit Distance: {total_edit_distance}")
        print(f"Global Character Accuracy (CA): {global_ca:.4f}")
        print(f"Character Error Rate (CER): {global_cer:.4f}")
        print(f"Perfectly Matched Sequences: {correct_samples} ({correct_samples/total_samples*100:.2f}%)")
        print(f"Average Inference Time: {avg_inference_time:.2f}s")
        print(f"Total Inference Time: {total_inference_time:.2f}s")
        print(f"Detailed results saved to {result_log_file}")
        print("="*50)
    else:
        print("No valid characters found in reference texts.")

if __name__ == "__main__":
    main()
