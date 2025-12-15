#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量OCR处理程序 - 多边形标注版本（MinerU + PaddleOCR）
支持处理多边形框标注的数据格式，使用MinerU和PaddleOCR进行识别，并提供可视化结果
"""

import json
import time
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from tqdm import tqdm
import traceback
import io
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageDraw, ImageFont

from mineru_vl_utils import MinerUClient, MinerUSamplingParams

# 尝试导入PaddleOCR
try:
    from paddleocr import TextRecognition
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    print("Warning: paddleocr module not found. PaddleOCR functionality will be disabled.")


class PolygonOCRProcessor:
    """多边形标注OCR处理器（MinerU + PaddleOCR）"""
    
    def __init__(self, server_url: str, image_root: str, json_root: str, output_root: str, 
                 max_workers: int = 10, verbose: bool = True, 
                 crop_image_root: str = None, 
                 presence_penalty: float = 1.0,
                 frequency_penalty: float = 0.05,
                 enable_paddle: bool = True,
                 vis_font_path: str = None):
        """
        初始化多边形OCR处理器
        
        Args:
            server_url: OCR服务器地址
            image_root: 图片根目录（本地）
            json_root: JSON文件根目录
            output_root: 输出结果根目录
            max_workers: 最大并发线程数
            verbose: 是否显示详细日志
            crop_image_root: 截取图片保存根目录（可选）
            presence_penalty: MinerU参数
            frequency_penalty: MinerU参数
            enable_paddle: 是否启用PaddleOCR
            vis_font_path: 可视化使用的字体路径
        """
        self.server_url = server_url
        self.image_root = Path(image_root)
        self.json_root = Path(json_root)
        self.output_root = Path(output_root)
        self.max_workers = max_workers
        self.verbose = verbose
        self.crop_image_root = Path(crop_image_root) if crop_image_root else None
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        
        self.enable_paddle = enable_paddle and PADDLE_AVAILABLE
        self.paddle_lock = threading.Lock()
        self.paddle_model = None
        
        # 可视化设置
        self.vis_root = self.output_root / "visualization"
        self.vis_font_path = vis_font_path
        if not self.vis_font_path:
            # 尝试查找系统字体
            if os.name == 'nt':  # Windows
                self.vis_font_path = "C:/Windows/Fonts/simhei.ttf"
            elif os.name == 'posix':  # Linux/Mac
                possible_fonts = [
                    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
                    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                    "/System/Library/Fonts/PingFang.ttc"
                ]
                for f in possible_fonts:
                    if os.path.exists(f):
                        self.vis_font_path = f
                        break
        
        # 初始化PaddleOCR
        if self.enable_paddle:
            try:
                print("正在初始化 PaddleOCR 模型 (PP-OCRv5_server_rec)...")
                self.paddle_model = TextRecognition(model_name="PP-OCRv5_server_rec")
                print("PaddleOCR 模型初始化成功")
            except Exception as e:
                print(f"PaddleOCR 模型初始化失败: {e}")
                self.enable_paddle = False
        
        # 创建输出目录
        self.output_root.mkdir(parents=True, exist_ok=True)
        if self.crop_image_root:
            self.crop_image_root.mkdir(parents=True, exist_ok=True)

    def create_client(self) -> MinerUClient:
        """创建 MinerU 客户端"""
        client = MinerUClient(
            backend="http-client",
            server_url=self.server_url,
            http_timeout=600,
        )
        return client

    def create_sampling_params(self) -> MinerUSamplingParams:
        """创建采样参数"""
        return MinerUSamplingParams(
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty
        )
    
    def ocr_with_mineru(self, image_bytes: bytes, client: MinerUClient) -> Dict:
        """使用MinerU进行识别"""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            prompt = "\nText Recognition:"
            sampling_params = self.create_sampling_params()
            
            ocr_text = client.client.predict(
                image=image,
                prompt=prompt,
                sampling_params=sampling_params,
            )
            
            content = ocr_text if ocr_text else ''
            rec_texts = [content] if content else []
            
            return {
                'success': True,
                'content': content,
                'rec_texts': rec_texts
            }
        except Exception as e:
            error_msg = f"MinerU请求异常: {str(e)}"
            return {"error": error_msg}

    def ocr_with_paddle(self, image_bytes: bytes) -> Dict:
        """使用PaddleOCR进行识别"""
        if not self.enable_paddle or self.paddle_model is None:
            return {"error": "PaddleOCR not enabled or initialized"}
            
        try:
            # bytes -> numpy (RGB)
            image = Image.open(io.BytesIO(image_bytes))
            # PaddleOCR expects BGR format (Opencv default)
            img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # PaddleOCR inference needs locking if running in threads
            with self.paddle_lock:
                # model.predict supports numpy array
                output = self.paddle_model.predict(input=img_np)
            
            rec_texts = []
            scores = []
            
            if output:
                # 尝试解析输出
                # output可能是generator或list
                for res in output:
                    # 检查res类型，可能是dict或对象
                    text = None
                    score = None
                    
                    # 尝试属性访问
                    if hasattr(res, 'rec_text'):
                        text = res.rec_text
                    elif isinstance(res, dict) and 'rec_text' in res:
                        text = res['rec_text']
                    
                    if hasattr(res, 'rec_score'):
                        score = res.rec_score
                    elif isinstance(res, dict) and 'rec_score' in res:
                        score = res['rec_score']
                        
                    # 如果仍然为空，打印debug信息（仅在verbose时）
                    if text is None and self.verbose:
                        print(f"PaddleOCR返回了未知格式的结果: {type(res)} - {res}")

                    if text is not None:
                        rec_texts.append(text)
                    if score is not None:
                        scores.append(score)
            
            content = " ".join(rec_texts)
            
            return {
                'success': True,
                'content': content,
                'rec_texts': rec_texts,
                'scores': scores
            }
        except Exception as e:
            return {"error": f"PaddleOCR异常: {str(e)}"}
    
    def extract_polygon_region(self, image: np.ndarray, vertices: List[Dict]) -> Tuple[np.ndarray, bool]:
        """提取多边形区域（最小外接矩形）"""
        try:
            if not vertices or len(vertices) < 3:
                return None, False
            
            points = np.array([[v['x'], v['y']] for v in vertices], dtype=np.int32)
            
            img_height, img_width = image.shape[:2]
            
            x, y, w, h = cv2.boundingRect(points)
            
            x = max(0, x)
            y = max(0, y)
            w = min(w, img_width - x)
            h = min(h, img_height - y)
            
            if w <= 0 or h <= 0:
                return None, False
            
            cropped = image[y:y+h, x:x+w]
            return cropped, True
            
        except Exception as e:
            print(f"提取多边形区域失败: {str(e)}")
            return None, False
    
    def _extract_annotation_image(self, image_path: Path, annotation: Dict, result: Dict, track_id: str = None) -> bytes:
        """提取标注框图片"""
        try:
            if not image_path.exists():
                result["error"] = f"图片不存在: {image_path}"
                return None
            
            # 使用cv2读取，处理中文路径
            img_np = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            
            if img_np is None:
                result["error"] = f"无法解码图片: {image_path}"
                return None
            
            shape_data = annotation.get("shape_data", {})
            if not shape_data:
                result["error"] = "shape_data为空"
                return None
            
            vertices = shape_data.get("vertices", [])
            if not vertices:
                result["error"] = "vertices为空"
                return None
            
            result["box_type"] = "polygon"
            result["box_info"] = {
                "vertices": [{"x": v.get("x"), "y": v.get("y")} for v in vertices]
            }
            
            cropped_img, success = self.extract_polygon_region(img_np, vertices)
            
            if not success or cropped_img is None:
                result["error"] = f"提取多边形区域失败"
                return None
            
            # 保存截取图片
            if self.crop_image_root and cropped_img is not None:
                try:
                    image_name = image_path.stem
                    crop_dir = self.crop_image_root / image_name
                    crop_dir.mkdir(parents=True, exist_ok=True)
                    
                    if track_id:
                        crop_filename = f"{track_id}.jpg"
                    else:
                        crop_filename = f"crop_{result.get('annotation_index', 'unknown')}.jpg"
                    
                    crop_file_path = crop_dir / crop_filename
                    crop_file_path_abs = crop_file_path.resolve()
                    
                    cv2.imencode('.jpg', cropped_img)[1].tofile(str(crop_file_path_abs))
                    
                except Exception as save_error:
                    if self.verbose:
                        print(f"      ⚠️  保存截取图片失败: {str(save_error)}")
            
            _, img_encoded = cv2.imencode('.jpg', cropped_img)
            return img_encoded.tobytes()
            
        except Exception as e:
            result["error"] = f"提取图片异常: {str(e)}"
            return None
    
    def parse_image_path(self, main_entry: str) -> str:
        """从main_entry中提取图片相对路径"""
        parts = main_entry.replace('\\', '/').split('/')
        if len(parts) >= 2:
            return '/'.join(parts[-2:])
        return main_entry
    
    def process_single_annotation(self, img_bytes: bytes, result: Dict, json_name: str, track_id: str, client: MinerUClient) -> Dict:
        """处理单个标注的OCR识别（MinerU + Paddle）"""
        if self.verbose:
            print(f"  🔍 [{json_name}] OCR: {track_id}")
        
        # MinerU OCR
        mineru_res = self.ocr_with_mineru(img_bytes, client)
        result["mineru_result"] = mineru_res
        
        # PaddleOCR
        if self.enable_paddle:
            paddle_res = self.ocr_with_paddle(img_bytes)
            result["paddle_result"] = paddle_res
        else:
            result["paddle_result"] = None

        # 检查错误 (只要有一个成功就算成功)
        errors = []
        if "error" in mineru_res:
            errors.append(f"MinerU: {mineru_res['error']}")
        
        if self.enable_paddle and "error" in result["paddle_result"]:
            errors.append(f"Paddle: {result['paddle_result']['error']}")
            
        if len(errors) > 0 and (not self.enable_paddle or len(errors) == 2):
            # 如果启用了Paddle且两者都失败，或者只启用MinerU且失败
             result["error"] = "; ".join(errors)

        return result
    
    def draw_text_pil(self, image: Image.Image, text: str, position: Tuple[int, int], color: Tuple[int, int, int] = (255, 0, 0)):
        """使用PIL绘制中文文本"""
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype(self.vis_font_path, 20)
        except:
            font = ImageFont.load_default()
            
        draw.text(position, text, font=font, fill=color)
        return image

    def visualize_results(self, json_path: Path, results: List[Dict]):
        """生成可视化结果：GT, MinerU, Paddle (Side-by-Side模式, 文字自适应框)"""
        if not results:
            return

        # 按图片分组
        img_groups = {}
        for res in results:
            img_path = res.get("image_path")
            if not img_path:
                continue
            if img_path not in img_groups:
                img_groups[img_path] = []
            img_groups[img_path].append(res)
        
        # 准备输出目录
        json_name = json_path.stem
        vis_base_dir = self.vis_root / json_name
        vis_gt_dir = vis_base_dir / "gt"
        vis_mineru_dir = vis_base_dir / "mineru"
        vis_paddle_dir = vis_base_dir / "paddle"
        
        for d in [vis_gt_dir, vis_mineru_dir, vis_paddle_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
        for img_path_str, anns in img_groups.items():
            try:
                img_path = Path(img_path_str)
                if not img_path.exists():
                    continue
                
                # 读取原图
                img_cv = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
                if img_cv is None:
                    continue
                
                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                
                width, height = img_pil.size
                
                # 创建并排画布 (左: 原图+框, 右: 文本)
                canvas_width = width * 2
                canvas_height = height
                
                # 初始化三个画布
                # GT
                canvas_gt = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
                canvas_gt.paste(img_pil, (0, 0))
                
                # MinerU
                canvas_mineru = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
                canvas_mineru.paste(img_pil, (0, 0))
                
                # Paddle
                canvas_paddle = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
                canvas_paddle.paste(img_pil, (0, 0))
                
                draw_gt = ImageDraw.Draw(canvas_gt)
                draw_mineru = ImageDraw.Draw(canvas_mineru)
                draw_paddle = ImageDraw.Draw(canvas_paddle)
                
                # 默认字体
                default_font = ImageFont.load_default()
                
                for ann in anns:
                    box_info = ann.get("box_info", {})
                    vertices = box_info.get("vertices", [])
                    if not vertices:
                        continue
                    
                    # 绘制多边形
                    points = [(v['x'], v['y']) for v in vertices]
                    
                    # 准备文本
                    gt_text = "".join(ann.get("ground_truth", []))
                    mineru_res = ann.get("mineru_result", {})
                    mineru_text = mineru_res.get("content", "") if mineru_res else ""
                    paddle_res = ann.get("paddle_result", {})
                    paddle_text = paddle_res.get("content", "") if paddle_res else ""
                    
                    # 绘制位置和尺寸计算
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    box_w = max_x - min_x
                    box_h = max_y - min_y
                    
                    # 简单判断是否竖排文本：高宽比 > 2 (仅作参考，主要靠框的形状)
                    # 策略：
                    # 如果文本框是竖长的 (h > 1.5 * w)，我们假设文字也是竖排或需要旋转，
                    # 但为了简单展示，我们仍然尝试横向绘制在框内，
                    # 只不过需要调整字体大小以适应宽度，或者旋转画布绘制。
                    # 这里采用简单策略：根据框的短边确定字体大小，并绘制在框的中心或左上角。
                    
                    is_vertical = box_h > 1.5 * box_w
                    
                    # 确定字体大小
                    # 如果是横向框，高度决定字号
                    # 如果是竖向框，宽度决定字号
                    target_size = box_w if is_vertical else box_h
                    font_size = max(10, int(target_size * 0.8)) # 至少10px
                    
                    try:
                        font = ImageFont.truetype(self.vis_font_path, font_size) if self.vis_font_path else default_font
                    except:
                        font = default_font
                    
                    # 绘制左侧多边形
                    draw_gt.polygon(points, outline="green", width=2)
                    draw_mineru.polygon(points, outline="blue", width=2)
                    if self.enable_paddle:
                        draw_paddle.polygon(points, outline="red", width=2)
                    
                    # 绘制右侧文本区域 (对应框的位置平移)
                    # 先在右侧画个淡色的框
                    offset_points = [(p[0] + width, p[1]) for p in points]
                    draw_gt.polygon(offset_points, outline="lightgray", width=1)
                    draw_mineru.polygon(offset_points, outline="lightgray", width=1)
                    if self.enable_paddle:
                        draw_paddle.polygon(offset_points, outline="lightgray", width=1)
                    
                    # 在右侧对应框内绘制文本
                    # 计算绘制起始点：居中或者左上对齐
                    # 简单起见，左上对齐 + 居中微调
                    text_x = min_x + width + (box_w * 0.1)
                    text_y = min_y + (box_h - font_size) / 2 # 垂直居中
                    
                    if is_vertical:
                        # 竖排框的处理比较复杂，这里简化为：
                        # 创建一个临时小图绘制文字，然后旋转贴上去，或者直接横着写在框里（如果不旋转）
                        # 既然用户说"竖框被横过来了"，可能意味着原来的图里字是竖的，但我们横着写了
                        # 这里我们尝试检测竖排，如果竖排，则逐字换行绘制（模拟竖排）
                        
                        # 重新计算字号，避免溢出
                        # 竖排时，字号由宽度决定
                        char_size = int(box_w * 0.8)
                        font_size = max(10, char_size)
                        try:
                            font = ImageFont.truetype(self.vis_font_path, font_size) if self.vis_font_path else default_font
                        except:
                            font = default_font
                            
                        def draw_vertical_text(draw_obj, text, x, y, f, color):
                            curr_y = y
                            for char in text:
                                draw_obj.text((x, curr_y), char, font=f, fill=color)
                                curr_y += font_size
                        
                        draw_vertical_text(draw_gt, gt_text, text_x, min_y, font, "green")
                        draw_vertical_text(draw_mineru, mineru_text, text_x, min_y, font, "blue")
                        if self.enable_paddle:
                            draw_vertical_text(draw_paddle, paddle_text, text_x, min_y, font, "red")
                            
                    else:
                        # 横排文本
                        draw_gt.text((text_x, text_y), gt_text, font=font, fill="green")
                        draw_mineru.text((text_x, text_y), mineru_text, font=font, fill="blue")
                        if self.enable_paddle:
                            draw_paddle.text((text_x, text_y), paddle_text, font=font, fill="red")
                
                # 保存图片
                img_name = img_path.name
                canvas_gt.save(vis_gt_dir / img_name)
                canvas_mineru.save(vis_mineru_dir / img_name)
                if self.enable_paddle:
                    canvas_paddle.save(vis_paddle_dir / img_name)
                    
            except Exception as e:
                print(f"可视化失败 {img_path_str}: {e}")
                if self.verbose:
                    traceback.print_exc()

    def process_single_json(self, json_path: Path) -> Dict:
        """处理单个JSON文件"""
        stats = {
            "json_path": str(json_path),
            "total_annotations": 0,
            "success": 0,
            "failed": 0,
            "results": []
        }
        
        try:
            if self.verbose:
                print(f"📂 开始处理: {json_path.name}")
            
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            rel_path = json_path.relative_to(self.json_root)
            output_path = self.output_root / rel_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            ocr_tasks = []
            
            if self.verbose:
                print(f"  📊 [{json_path.name}] 开始提取标注框...")
            
            # 收集任务
            for entry_idx, entry in enumerate(data.get("entries", [])):
                main_entry = entry.get("main_entry", "")
                if not main_entry:
                    continue
                
                img_rel_path = self.parse_image_path(main_entry)
                image_path = self.image_root / img_rel_path
                
                instance_anns = entry.get("instance_anns", [])
                for idx, annotation in enumerate(instance_anns):
                    stats["total_annotations"] += 1
                    track_id = annotation.get("track_id", f"idx_{idx}")
                    
                    ground_truth = []
                    attrs = annotation.get("attrs", [])
                    for attr in attrs:
                        values = attr.get("values", [])
                        ground_truth.extend(values)
                    
                    # 检查GT是否为空，如果为空则跳过
                    has_content = any(str(t).strip() for t in ground_truth)
                    if not has_content:
                        continue
                    
                    result = {
                        "entry_index": entry_idx,
                        "annotation_index": idx,
                        "track_id": track_id,
                        "image_path": str(image_path),
                        "main_entry": main_entry,
                        "ground_truth": ground_truth,
                        "mineru_result": None,
                        "paddle_result": None,
                        "error": None
                    }
                    
                    img_bytes = self._extract_annotation_image(image_path, annotation, result, track_id)
                    
                    if result["error"] is None and img_bytes is not None:
                        ocr_tasks.append((img_bytes, result, json_path.name, track_id))
                    else:
                        stats["results"].append(result)
                        stats["failed"] += 1
                        if result["error"] and self.verbose:
                            print(f"  ⚠️  [{json_path.name}] 提取失败: 标注 {track_id} - {result['error']}")
            
            if self.verbose:
                print(f"  ✓ [{json_path.name}] 提取完成，共 {len(ocr_tasks)} 个标注框")
            
            # 执行OCR
            if ocr_tasks:
                if self.verbose:
                    print(f"  🚀 [{json_path.name}] 开始OCR识别 (共 {len(ocr_tasks)} 个)...")
                
                client = self.create_client()
                
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_task = {
                        executor.submit(self.process_single_annotation, img_bytes, result.copy(), json_name, track_id, client): 
                        (result, track_id)
                        for img_bytes, result, json_name, track_id in ocr_tasks
                    }
                    
                    with tqdm(total=len(ocr_tasks), desc=f"  Processing {json_path.name}", leave=False) as pbar:
                        for future in as_completed(future_to_task):
                            original_result, track_id = future_to_task[future]
                            try:
                                processed_result = future.result()
                                original_result.update(processed_result)
                                
                                # 只要有一个成功就算成功（或者根据业务逻辑调整）
                                is_failed = False
                                if processed_result.get("error"):
                                    # 如果整体被标记为error
                                    is_failed = True
                                
                                if is_failed:
                                    stats["failed"] += 1
                                    print(f"  ❌ [{json_path.name}] OCR失败 {track_id}: {processed_result.get('error')}")
                                else:
                                    stats["success"] += 1
                                
                                stats["results"].append(original_result)
                            except Exception as e:
                                original_result["error"] = f"OCR异常: {str(e)}"
                                stats["failed"] += 1
                                stats["results"].append(original_result)
                                print(f"  ❌ [{json_path.name}] OCR异常 {track_id}: {str(e)}")
                            finally:
                                pbar.update(1)
            
            # 保存JSON结果
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            
            # 生成可视化
            if self.verbose:
                print(f"  🎨 [{json_path.name}] 生成可视化结果...")
            self.visualize_results(json_path, stats["results"])
            
            # 打印总结
            if stats['failed'] > 0:
                print(f"⚠️  完成: {json_path.name} - ✓{stats['success']} ✗{stats['failed']}")
            else:
                print(f"✅ 完成: {json_path.name} - ✓{stats['success']}")
            
        except Exception as e:
            stats["error"] = f"处理JSON文件异常: {str(e)}\n{traceback.format_exc()}"
            print(f"❌ 失败: {json_path.name} - {str(e)}")
        
        return stats
    
    def find_all_json_files(self) -> List[Path]:
        """查找所有JSON文件"""
        json_files = []
        for json_path in self.json_root.rglob("*.json"):
            json_files.append(json_path)
        return sorted(json_files)
    
    def process_all(self):
        """批量处理所有JSON文件"""
        json_files = self.find_all_json_files()
        print(f"\n{'='*60}")
        print(f"📋 找到 {len(json_files)} 个JSON文件")
        print(f"🔧 OCR类型: MinerU + PaddleOCR")
        print(f"🌐 MinerU服务器: {self.server_url}")
        print(f"🚣 PaddleOCR: {'启用' if self.enable_paddle else '禁用'}")
        print(f"🚀 并发线程数: {self.max_workers}")
        print(f"💡 详细日志: {'开启' if self.verbose else '关闭'}")
        if self.crop_image_root:
            print(f"💾 截取图片保存: {self.crop_image_root}")
        print(f"🖼️  可视化结果保存至: {self.vis_root}")
        print(f"{'='*60}\n")
        
        if not json_files:
            print("未找到JSON文件！")
            return
        
        total_stats = {
            "total_files": len(json_files),
            "processed_files": 0,
            "total_annotations": 0,
            "total_success": 0,
            "total_failed": 0
        }
        
        start_time = time.time()
        
        with tqdm(total=len(json_files), desc="总进度") as pbar:
            for json_path in json_files:
                try:
                    stats = self.process_single_json(json_path)
                    total_stats["processed_files"] += 1
                    total_stats["total_annotations"] += stats["total_annotations"]
                    total_stats["total_success"] += stats["success"]
                    total_stats["total_failed"] += stats["failed"]
                    
                    pbar.set_postfix({
                        '已处理': f"{total_stats['processed_files']}/{total_stats['total_files']}",
                        '成功': total_stats['total_success'],
                        '失败': total_stats['total_failed']
                    })
                except Exception as e:
                    print(f"\n❌ 处理 {json_path} 时发生异常: {str(e)}\n")
                finally:
                    pbar.update(1)
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("批量处理完成！")
        print("="*60)
        print(f"总文件数: {total_stats['total_files']}")
        print(f"处理文件数: {total_stats['processed_files']}")
        print(f"总标注数: {total_stats['total_annotations']}")
        print(f"成功: {total_stats['total_success']}")
        print(f"失败: {total_stats['total_failed']}")
        
        if total_stats['total_annotations'] > 0:
            success_rate = (total_stats['total_success'] / total_stats['total_annotations']) * 100
            print(f"成功率: {success_rate:.2f}%")
        
        print(f"总耗时: {elapsed_time:.2f}秒")
        if total_stats['processed_files'] > 0:
            print(f"平均每个文件: {elapsed_time/total_stats['processed_files']:.2f}秒")
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="批量OCR处理程序 - MinerU + PaddleOCR")
    parser.add_argument("--server_url", type=str, default="http://10.10.50.50:30000",
                        help="OCR服务器地址")
    parser.add_argument("--image_root", type=str, required=True,
                        help="图片根目录")
    parser.add_argument("--json_root", type=str, required=True,
                        help="JSON文件根目录")
    parser.add_argument("--output_root", type=str, required=True,
                        help="输出结果根目录")
    parser.add_argument("--max_workers", type=int, default=4,
                        help="最大并发线程数")
    parser.add_argument("--verbose", action="store_true",
                        help="显示详细日志")
    parser.add_argument("--crop_image_root", type=str, default=None,
                        help="截取图片保存根目录")
    parser.add_argument("--presence_penalty", type=float, default=1.0,
                        help="MinerU参数")
    parser.add_argument("--frequency_penalty", type=float, default=0.05,
                        help="MinerU参数")
    parser.add_argument("--no_paddle", action="store_true",
                        help="禁用PaddleOCR")
    parser.add_argument("--font_path", type=str, default=None,
                        help="可视化字体路径")
    
    args = parser.parse_args()
    
    processor = PolygonOCRProcessor(
        server_url=args.server_url,
        image_root=args.image_root,
        json_root=args.json_root,
        output_root=args.output_root,
        max_workers=args.max_workers,
        verbose=args.verbose,
        crop_image_root=args.crop_image_root,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        enable_paddle=not args.no_paddle,
        vis_font_path=args.font_path
    )
    
    try:
        processor.process_all()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断程序")
    except Exception as e:
        print(f"\n❌ 程序异常: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
