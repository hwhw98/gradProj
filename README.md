# Transformer 인코더 기반 탠덤 질량 스펙트럼 노이즈 피크 판별

> 한양대학교 컴퓨터소프트웨어학부 졸업연구 · 2023
> 심승현, **정희원**, 이상정, 박희진
> 📄 [논문 PDF](./트랜스포머%20인코더를%20활용한%20탠덤%20질량%20스펙트럼의%20노이즈%20피크%20판별_심승현_정희원.pdf)

## 개요

단백체학에서는 펩타이드를 식별하기 위해 탠덤 질량 분석(MS/MS)을 사용합니다. 그런데 결과 스펙트럼에는 기계 오차 등으로 생긴 **노이즈 피크**가 섞여 있어 식별을 어렵게 만듭니다.

이 연구는 스펙트럼을 문장, 각 피크를 단어로 보고 **Transformer 인코더**로 피크 간 관계를 학습해, 각 피크가 노이즈인지 아닌지를 판별하는 **이진 분류 모델**을 제안합니다.

## 모델 구조

<img src="./model_architecture.png" width="420">

- **입력**: 피크의 m/z(질량 대 전하 비)와 intensity(밀집도)를 각각 임베딩한 뒤 더한 값
- **인코더만 사용한 이유**: 펩타이드 조각 이온(b이온, y이온) 사이의 m/z 관계는 양방향적입니다. 디코더는 왼쪽 문맥만 보기 때문에, 양방향 문맥을 반영하는 인코더만 쌓아 구성했습니다.
- **출력**: Linear → Sigmoid로 피크별 노이즈 확률을 구하고, 임계값 0.5로 판별

## 구현 (`main.py`)

TensorFlow로 Transformer 인코더를 직접 구현했습니다.

| 구성 요소 | 내용 |
|---|---|
| Scaled Dot-Product / Multi-Head Attention | Q·K·V 투영, 헤드 분할, 패딩 마스크 적용 |
| Positional Encoding | sin/cos 위치 인코딩 |
| Encoder Layer | Multi-Head Attention + Feed Forward, Dropout, Layer Normalization |
| Padding Mask | 스펙트럼마다 다른 피크 개수를 0 패딩 후 어텐션에서 제외 |
| Learning Rate | 원 논문(Attention Is All You Need)의 warmup 스케줄 + Adam |
| 평가 | Accuracy, Precision, Recall |

## 데이터

- **NIST Consensus Human HCD Libraries**, 약 91만 개 스펙트럼
- 이온 정보가 있는 피크(실제 피크)와 없는 피크(노이즈)를 라벨링
- Long-tail 완화를 위해 m/z와 피크 개수 기준 양 극단 약 4% 제외
- m/z는 소수점 둘째 자리까지 사용, intensity는 스펙트럼 내 상대값으로 변환
- 학습 : 검증 : 테스트 = 8 : 1 : 1

## 실험 결과

| | 실험 1 | 실험 2 | 실험 3 |
|---|---|---|---|
| d_model / dff | 128 | 128 | 256 |
| Layers | 4 | 6 | 4 |
| Precision | 92.49% | 94.17% | **96.91%** |
| Recall | 94.99% | 95.91% | **97.26%** |
| Accuracy (피크 단위) | 93.46% | 94.85% | **97.00%** |
| Spectrum Accuracy | 5.78% | 9.92% | **20.66%** |

> Spectrum Accuracy는 한 스펙트럼 안의 **모든 피크를 하나도 틀리지 않고** 판별한 비율입니다.
> 피크 단위 정확도는 97%로 높지만, 수백 개 피크를 전부 맞히는 것은 훨씬 어렵기 때문에 두 기준을 구분해 해석했습니다. 모델이 커질수록 스펙트럼 정확도가 가파르게 오르는 경향을 확인했습니다.

## 실행

```bash
python main.py <tf.data 데이터셋 경로> <체크포인트 저장 경로>
```

- Python, TensorFlow 2.x
- 하이퍼파라미터: d_model 256, layers 4, heads 4, dff 256, batch 128, dropout 0.1, epochs 100

## 담당 역할

- 질량 분석 원리와 Transformer, BERT 구조를 학습해 매주 발표 ([학습 자료](https://github.com/hwhw98/transformer_model))
- 모델 구성과 데이터셋 특성 논의, 실험 진행과 결과 정리
- 논문 형식의 문서화
