import base64
import requests

API_URL = "http://10.10.50.50:8080/ocr"
file_path = "E:\图像数据集\样例\༈ བྱང་ཆུབ་སེམས་དཔའི་སྤྱོད་པ་ལ་འཇུག་པའི་ཚིག་འགྲེལ་གཅེས་བཏུས།_08.jpg"

with open(file_path, "rb") as file:
    file_bytes = file.read()
    file_data = base64.b64encode(file_bytes).decode("ascii")

payload = {"file": file_data, "fileType": 1}

response = requests.post(API_URL, json=payload)

assert response.status_code == 200
result = response.json()["result"]
for i, res in enumerate(result["ocrResults"]):
    print(res["prunedResult"])
    ocr_img_path = f"ocr_{i}.jpg"
    with open(ocr_img_path, "wb") as f:
        f.write(base64.b64decode(res["ocrImage"]))
    print(f"Output image saved at {ocr_img_path}")