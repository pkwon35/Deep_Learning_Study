# 필요 모듈 실행
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio

import tensorflow as tf
import os
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import random

from PIL import Image
import numpy as np
from loguru import logger
import gradio as gr
import os
import json



title = "Drawing Style Classification DEMO" # 데모 제목

# ---------- Settings ----------
SERVER_PORT = 8103 # 서버 포트 설정   14.49.45.123:8103 
SERVER_NAME = "0.0.0.0" 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # "-1" 은 CPU 사용으로 지정


# ---------- gradio 페이지 하단에 보이는 예시 input ----------
examples = [
    ['./data/user.lts/images/ellipse/ellipse.lts.0002.png'],
    ['./data/user.lts/images/rectangle/rectangle.lts.0004.png'],
    ['./data/user.lts/images/triangle/triangle.lts.0006.png']
]

# ---------- Logging ----------
logger.add('app.log', mode='a')
logger.info("================== App Restarted ==================")


# ---------- 사전 학습된 그림 스타일 분류기 가중치 불러오기 ----------
model = tf.keras.models.load_model('./cnn_hand_drawn.h5',custom_objects={'KerasLayer': hub.KerasLayer})


# ---------- 사용되는 함수 ----------

def preprocess_image(image):
  image = np.array(image)
  # reshape into shape [batch_size, height, width, num_channels]
  img_reshaped = tf.reshape(image, [1, image.shape[0], image.shape[1], image.shape[2]])
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  image = tf.image.convert_image_dtype(img_reshaped, tf.float32)
  return image

def load_image(image_path, image_size=70, dynamic_size=False, max_dynamic_size=512): #이미지 로드 함수
  """Loads and preprocesses images."""
  logger.info(f'queries:{image_path}')
  img = preprocess_image(Image.open(image_path))
  # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
  img_raw = img
  if tf.reduce_max(img) > 1.0:
    img = img / 255.
  if len(img.shape) == 3:
    img = tf.stack([img, img, img], axis=-1)
  if not dynamic_size:
    img = tf.image.resize(img, [image_size, image_size])

  elif img.shape[1] > max_dynamic_size or img.shape[2] > max_dynamic_size:
    img = tf.image.resize_with_pad(img, max_dynamic_size, max_dynamic_size)
  return img, img_raw



# ---------- main 함수 ----------

def classify_hand_drawn(file_path): # 분류를 진행하는 코드
    image, original_image = load_image(file_path) # load_image 함수 사용해 이미지 불러오기
    logger.info('--- image loaded')
    
    # 분류기에 이미지 주입
    class_names=["ellipse","rectangle","triangle"]
    if image.shape[-1]==4:
        image = tfio.experimental.color.rgba_to_rgb(image)
    prediction_scores = model.predict(image)
    
    sm_layer = tf.keras.layers.Softmax()
    
    # 분류기 확률 추출
    sm_p_score = sm_layer(prediction_scores).numpy()
    dic={}
    print(sm_p_score)
    # 결과 값 높은 확률 순으로 key값 추출
    for i in range(3):
        perc=(sm_p_score*100)[0]
        dic[perc[i]]=class_names[i]
    keys=list(dic.keys())
    keys.sort(reverse=True)
    
    logger.info('--- output generated')
    
    # 위에서 추출한 key값 순으로 결과 dictionary 생성
    result_dic={}
    print()
    print(keys)
    print()
    result_dic[dic[keys[0]].upper()]=f'{str(np.round(keys[0]))} %'
    result_dic[dic[keys[1]].upper()]=f"{str(np.round(keys[1]))} %"
    result_dic[dic[keys[2]].upper()]=f"{str(np.round(keys[2]))} %"
    
    return result_dic # output dictionary 변환해서 전달



DClassifier = gr.Interface(fn=classify_hand_drawn, inputs=gr.inputs.Image(label='이미지', type='file'), outputs="json",examples=examples,title=title)


if __name__ == '__main__':
    try:
        DClassifier.launch(server_name=SERVER_NAME,
                     server_port=SERVER_PORT)
        
    except KeyboardInterrupt as e:
        print(e)
        DClassifier.close()
        gr.close_all()

    finally:
        DClassifier.close()
        gr.close_all()
        exit(0)