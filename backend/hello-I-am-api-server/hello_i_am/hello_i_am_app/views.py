import os

import numpy as np
from rest_framework.response import Response
from rest_framework.decorators import api_view

from django.shortcuts import render
from django.http import HttpResponse

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text

from keras.models import load_model
from keras.preprocessing import image
import numpy as np

from .models import Screenimage

## 손동작 사진 인삭하는 함수
@api_view(['POST'])
def get_image(request):
    form = Screenimage()

    # 유니티로부터 이미지 받아오기
    screenimage = request.FILES['screenImage']
    # print(screenimage)
    form.image = screenimage
    form.save()

    # 사진 인식 모델 로드하기
    file_name = os.path.dirname(__file__) + '/model3.h5'    # 배포할 때를 대비해... (내 로컬 절대경로로 쓰면 안 됨)
    model = load_model(file_name)

    # 모델에 인풋할 수 있도록 형태 바꾸기
    img = image.load_img("./media/images/screenImage.dat", target_size=(150, 150))
    # print(os.listdir("./media/images/"))

    img2 = image.img_to_array(img)
    img2 = np.expand_dims(img2, axis=0)

    # 사진 삭제하기
    form.delete()  # db에서 삭제하기 - media 폴더의 사진도 같이 삭제

    # 예측하기
    predict = model.predict(img2)
    print("사진의 예측값은:", predict)
    # {'heart':0, 'hi':1, 'pet':2}

    res = -1    # 아무 인덱스도 아닌 -1로 초기화

    if predict[0][0] == 1:
        res = 0
    elif predict[0][1] == 1:
        res = 1
    elif predict[0][2] == 1:
        res = 2

    return HttpResponse(res)


## 음성인식으로 받은 문장의 카테고리를 분류해서 대답하는 함수
@api_view(['POST'])
def label_user_chat(request):
    # sentence = str(request.data.get('sentence'))
    sentence = str(request.POST['sentence'])
    print(sentence)
    animal = int(request.POST['animal'])

    # Load required models
    encoder = hub.KerasLayer("https://tfhub.dev/jeongukjae/distilkobert_sentence_encoder/1")
    preprocessor = hub.KerasLayer("https://tfhub.dev/jeongukjae/distilkobert_cased_preprocess/1")

    # Define sentence encoder model
    inputs = tf.keras.Input([], dtype=tf.string)
    encoder_inputs = preprocessor(inputs)
    sentence_embedding = encoder(encoder_inputs)
    normalized_sentence_embedding = tf.nn.l2_normalize(sentence_embedding, axis=-1)
    model = tf.keras.Model(inputs, normalized_sentence_embedding)

    # Encode sentences using distilkobert_sentence_encoder
    sentences1 = tf.constant([
        sentence
    ])
    sentences2 = tf.constant([  # 질문의 카테고리에 따른 대표 질문들
        "안녕? 너는 이름이 뭐야", "너는 어디에 살아?", "밥 뭐먹었어?", "기분이 어때?", "힘든 이유가 뭐야?", "내가 널 도울 방법이 있을까?",
        "너는 누구랑 같이 살아?", "너는 몇 살이야?", "어떻게 생겼어?"
    ])
    embeddings1 = model(sentences1)
    embeddings2 = model(sentences2)

    # Calculate cosine similarity
    res = tf.tensordot(embeddings1, embeddings2, axes=[[1], [1]])
    print(res)
    res_index = np.argmax(res[0])

    polarbear_res = ["안녕~ 나는 북극곰 곰곰이야 ^_^", "우리 북극곰들은 추운 북극에 있는 빙하 위에 살아~!",
                     "나는 물범을 먹어! 근데 요즘은 사냥하기가 힘들어..", "오늘은 먹이를 못찾아서 배고프고 슬퍼..",
                     "지구 온난화 때문에 날씨가 더워서 지쳐, 그리고 빙하가 녹아서 사냥하기가 힘들어..",
                     "우리를 떠올리면서 환경을 아껴줘! 그리고 북극곰을 돕는 단체들에 대해서 관심을 꾸준히 가져줘~",
                     "최근에 엄마랑 형제로부터 독립해서 혼자 지내고 있어.", "나는 3살이야 그리고 우리들은 평균 25년 정도 살아.",
                     "몸무게는 400kg이고, 내가 아는 곰 중에 가장 큰 곰은 700kg이야~ 수영을 해야해서 길고 유선형으로 생겼어. 그리고 코와 피부는 검은색이고 하얗게 보이는 털은 실제로는 투명해~!"]

    redpanda_res = ["안녕~ 나는 래서판다 래서야 ^_^", "나는 히말라야 산맥 동쪽 숲에 살고 있어. 주로 나무 위가 내 놀이터야!",
                    "대나무 잎을 먹었어! 난 그게 제일 맛있더라. 근데 구하기가 쉽지 않아서 배부르게 먹지는 못했어..",
                    "조금 슬퍼, 요즘 친구들이 가끔 사라지기도 하고 우리집도 안전하지 않은 것 같아.. 걱정할게 많거든..",
                    "자꾸 날씨가 따뜻해져서 내가 살 숲이 줄어들어.. 그리고 지난달은 비가 너무 안 와서 숲에 불이 날 뻔 했다구!",
                    "날씨가 더 따뜻해지지 않게 에너지를 아껴줘! 안 쓰는 방의 불을 잘 끄는 거 말이야.. 사람들이 우리집을 몰래 베어가지 않게 계속 관심을 가져줘~",
                    "나는 지금 혼자 살고있어. 육아하는 것도 별로 안 좋아해서 시간이 지나도 아마 혼자살고 있지 않을까?",
                    "난 지금 3살이야! 완전 어른이라구~", "나는 지금 53cm정도의 키를 가지고 있어! 근데 내 친구는 65cm더라 완전 크지?"]

    snowleopard_res = ["안녕~ 나는 눈표범 누누야 ^_^", "티베트 고원의 추운 지역에서 살고 있어.",
                       "아이벡스라는 야생 염소를 먹었어~ 내가 좋아하는 사냥감 중 하나야.",
                       "어제 가축들을 사냥을 하러 마을까지 가느라 너무 떨리고 무서웠어. 하지만 어쩔 수 없는걸..",
                       "내 가죽을 탐내는 밀렵꾼들 때문에 너무 무서워.. 그리고 기후 변화로 내가 살 수 있는 추운 지역이 점점 줄어들고 있어서 힘들어..",
                       "나를 위해서 에너지를 아껴 써줘! 그리고 나를 노리는 못된 밀렵꾼들로부터 지켜줘. 너희의 관심이면 충분해~",
                       "몇 달 전에 엄마로부터 독립했어!", "나는 1살이야! 그리고 우리 눈표범들은 15~20년 정도 살아",
                       "나는 50kg정도로 네가 아는 다른 표범들보다 훨씬 작아. 회색 몸에 멋있는 갈색 점을 가지고 있어~"]

    # 질문의 카테고리에 따른 대답
    # 북극곰 (animal: 0)
    if animal == 0:
        res = polarbear_res[res_index]
    # 레서판다 (animal: 1)
    if animal == 1:
        res = redpanda_res[res_index]
    # 설표 (animal: 2)
    if animal == 2:
        res = snowleopard_res[res_index]

    return HttpResponse(res)
