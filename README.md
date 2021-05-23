# AI-Learning
 
 ## Machine Learning

 - 주요 단어
   -  지도 학습(Supervised Learning)
      -  레이블된 데이터(정답이 확실한 데이터)
      -  직접 피드백
      -  출력 맻 미래 예측
   - 비지도 학습(Unsupervise Learning)
     - 레이블 및 타깃 없음
     - 피드백 없음
     - 데이터에서 숨겨진 구조 찾기
   - 강화 학습(Reinforcement Learnig)
     - 결정 과정
     - 보상 시스템
     - 여속된 행동에서 학습

----

  ### 지도 학습
- 희망하는 출력 레이블(신호)가 있는 일련의 샘필 즉, 예측 성공한 학습 형태

<img src="Supervised.jpg">
    
위 그림에서 새로운 데이터는 레이블 즉 정답이 없는 데이터를 예측 모델에 입력하여 예측하고 원하는 값이 나올경우 레이블이라고 볼 수 있다.

 - 분류(개별 클래스 레이블이 있는 지도 학습)
   - 이진 분류 = 결정경계로 음성 클래스와 양성 클래스로 구분 가능한 데이터 값들의 분류 방법
     - 다중 분류 = 순서가 없는 범주나 클래스 레이블로 표현되는 데이터 값들의 분류 방법
 - 회귀(연속적인 값을 출력하는 방법)
     - 회귀 예측
       - 연속적인 출력 값을 예측하는 분석 방법
       - 예측변수(특성)와 연속적인 반응 변수(타깃) 사이의 출력값을 예측하는 방법
 - 선형 회귀
   - 특성 x와 타깃 y가 주어진 경우 데이터 포인트와 직선 사이의 최소 거리의 직선을 이용하여 새로운 데이터 출력 값을 예측하는 방법
----
### 강화 학습
- 환경과 상호 작용하여 시스템 성능을 향상시키는 것을 목적으로 둔다.
  - 결과 값이 데이터의 레이블 값이 아닌 보상 함수로 통한 얼마나 행동이 좋은지 측정하는 값
  - ex) 체스 게임
  <img src="reinforcement.png">
(출처: https://www.secmem.org/blog/2020/02/08/snake-dqn/)

    위 그림에서 환경은 소프트웨어의 시뮬레이션, 에이전트는 모델의 관계를 나타낸 것이다.
    
    => 행동을 수행하고 즉시 얻거나 지연된 피드백을 통하여 얻은 전체 보상을 최대화 하는 학습 행동이다.

----

### 비지도 학습
 - 레이블되지 않거나 구조를 알 수 없는 데이터를 다루는 학습 방법
   - 군집(사전 정보 없이 쌓여 있는 그룹 정보를 의미 있는 서브그룸 또는 클러스터로 조직하는 탐색적 데이터 분석 기법)
   - 각 클러스터는 서로 유사성을 공유하고 다른 클러스터와 비슷하지 않은 샘플 그룹을 형성한다.
   - 비지도 분류라고 불리기도 한다.
   - 클러스터링 = 정보를 조직화하고 데이터에서 의미 있는 관계를 유도하는 도구
 - 차원 축소: 데이터 압축
   -  쉽게 말하면 많은 특성 => 적은 특성으로 압축하는 방법
   -  잡음 데이터를 제거하기 위해 특성 전처리 단계에서 종종 적용하는 방법

----
용어 정리
 - 훈련 샘플: 데이터셋을 나타내는 테이블의 행
 - 훈련: 특성과 샘플들을 알고리즘에 돌리는 행위
 - 특성(x): 데이터 테이블이나 데이터 행렬의 열
 - 타깃(y): 특성에 따른 출력, 레이블
 - 손실 함수: 전체 데이터셋에 대해 계산한 손실 즉, 정확도를 나타내는 척도이다.

----

- 머신 러닝의 작업 흐름도
<img src ="roadmap.jpg">

- 전처리: 데이터 형태 갖추기
  - 머신 러닝 알고리즘에서 최적의 성능을 내리기 위해서는 선택된 특성이 같은 스케일을 가져야한다.
  - 만일 선택된 특성이 상관관계가 심할 경우 차원 축소 기법을 사용하여 압축한다, 이는 저장 공간이 덜 필요하고 학습 알고리즘을 더 빨리 실행 시킬 수 있다.
  - 신호 대 잡음비가 높은 경우 차원 축소 기법을 통해 예측 성능을 높이기도 한다.

----

