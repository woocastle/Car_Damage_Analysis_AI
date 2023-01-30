import sys
# pip install PyQt5
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import QPixmap
from PIL import Image
from keras.models import load_model
import numpy as np
# pip install git+https://github.com/arsenyinfo/EnlightenGAN-inference
from enlighten_inference import EnlightenOnnxModel

form_window = uic.loadUiType('../car_accident_qt.ui')[0]

import torch
import cv2
import matplotlib.pyplot as plt
from src.Models import Unet
# 한글 깨짐 오류 해결
import matplotlib
matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False


# 모델 로드 1 (전체 영역)
weight_path = '../models/[PART]Unet.pt'
n_classes = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Unet(encoder='resnet34', pre_weight='imagenet', num_classes=n_classes).to(device)
model.model.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))
model.eval()
# 모델 로드 2 (파손 부위별 영역)
labels = ['Breakage_3', 'Crushed_2', 'Scratch_0', 'Seperated_1']
models = []
n_classes = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
for label in labels:
    model_path = f'../models/[DAMAGE][{label}]Unet.pt'
    model2 = Unet(encoder='resnet34', pre_weight='imagenet', num_classes=n_classes).to(device)
    model2.model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model2.eval()
    models.append(model2)


class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.model = model
        self.path = ('../image/111.jpg', '')
        self.btn_open.clicked.connect(self.image_open_slot)
        self.setFixedWidth(1200)
        self.setFixedHeight(600)
        self.btn_open.setStyleSheet("color: b;"
                                    "border-style: solid;"
                                    "border-width: 2px;"
                                    "border-color: grey;"
                                    "border-radius: 2px;"
                                    "background-color: #87cefa")

    def image_open_slot(self):
        self.path = QFileDialog.getOpenFileName(self, 'Open File',
                    '../image', 'Image Files(*.jpg;*.png);;All Files(*.*)')        # 'Open File 제목', '경로', '파일 형식;전체파일형식'
        if self.path[0]:                # 위에서 사진을 입력 받았을 때
            try:
                # 이미지 전처리
                img = cv2.imread(self.path[0])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (256, 256))
                model_light = EnlightenOnnxModel()
                img = model_light.predict(img)
                img_input = img / 255.
                img_input = img_input.transpose([2, 0, 1])
                img_input = torch.tensor(img_input).float().to(device)
                img_input = img_input.unsqueeze(0)

                # 모델1 적용(전체 파트)
                output = model(img_input)
                img_output = torch.argmax(output, dim=1).detach().cpu().numpy()
                img_output = img_output.transpose([1, 2, 0])
                area_sum = img_output.sum()

                # plot 그리기
                fig, ax = plt.subplots(1, 5, figsize=(24, 10))
                ax[0].imshow(img)
                # ax[0].set_title('Original')
                ax[0].axis('off')
                ax[0].set_title('원본', fontsize=30)
                ax[1].set_title('파손', fontsize=30)
                ax[2].set_title('찌그러짐', fontsize=30)
                ax[3].set_title('스크래치', fontsize=30)
                ax[4].set_title('이격', fontsize=30)

                # 모델2 적용(부위별 파트)
                outputs = []
                for i, model2 in enumerate(models):
                    output = model2(img_input)
                    img_output = torch.argmax(output, dim=1).detach().cpu().numpy()
                    img_output = img_output.transpose([1, 2, 0])
                    outputs.append(img_output)

                    # ax[i + 1].set_title(labels[i])
                    ax[i + 1].imshow(img.astype('uint8'), alpha=0.9)
                    ax[i + 1].imshow(img_output, cmap='jet', alpha=0.6)
                    ax[i + 1].axis('off')

                fig.set_tight_layout(True)

                # plot 저장
                plt.savefig("../image/result_{}.png".format(self.path[0].split('/')[-1].split('.')[0]))

                # plot 출력
                pixmap = QPixmap("../image/result_{}.png".format(self.path[0].split('/')[-1].split('.')[0]))
                self.lbl_image.setPixmap(QPixmap(pixmap))
                self.resize(20, 20)  # 이미지를 보여주기 위해 출력될 창의 크기를 400×400으로 설정
                self.show()

                # 각각의 파손 형태 영역 합
                area_breakage = outputs[0].sum()
                area_crushed = outputs[1].sum()
                area_scratch = outputs[2].sum()
                area_seperated = outputs[3].sum()

                print("../image/result_{}.png".format(self.path[0].split('/')[-1].split('.')[0]))
                print(area_sum, area_breakage, area_crushed, area_scratch, area_seperated)
                # 수리비
                price_table = [
                    120,  # Breakage_3 / 파손 200
                    90,  # Crushed_2 / 찌그러짐 150
                    60,  # Scratch_0 / 스크래치 100
                    120,  # Seperated_1 / 이격 200
                ]
                total = 0
                for i, price in enumerate(price_table):
                    area = outputs[i].sum()
                    total += area * price
                self.lbl_repair.setText(f'고객님, 총 수리비는 {total:,}원 입니다!')

                # 손상심각도
                severity = ( area_breakage * 3.0 + area_crushed * 2.0 + area_seperated * 1.2 + area_scratch * 1.0) * 100 / (3 * area_sum)
                if 0 <= severity < 11:
                    level = 4
                elif severity < 41:
                    level = 3
                elif severity < 81:
                    level = 2
                else:
                    level = 1
                self.lbl_level.setText('손상심각도 : {}등급'.format(level))
            except:
                print('error')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())