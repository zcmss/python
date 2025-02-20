import sys
from pathlib import Path
# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))
from data.loader import CLASS_NAMES
from models.cnn import SimpleCNN
from flask import Flask, request, jsonify
from PIL import Image
import torch
import io
from torchvision import transforms


app = Flask(__name__)

# 加载模型
model = SimpleCNN()
model.load_state_dict(torch.load('best_cnn.pth', map_location='cpu'))
model.eval()

# 预处理转换
transform = transforms.Compose([
    transforms.Resize(28),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    try:
        image = Image.open(io.BytesIO(file.read()))
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(tensor)
        pred = output.argmax().item()
        return jsonify({
            'class': CLASS_NAMES[pred],
            'confidence': float(output.softmax(dim=1)[0, pred])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
