from paddleocr import TextRecognition
# 使用已注册的基础 model_name，但指定自定义的 model_dir
# inference.yml 中的 model_name 已修改为 PP-OCRv5_server_rec 以通过注册检查
model = TextRecognition(model_name="PP-OCRv5_server_rec", model_dir="PP-OCRv5_server_rec_Tibetan_infer")

output = model.predict(input="BDRC/paddleocr_data/images/I1KG811160005_0.png", batch_size=1)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")
