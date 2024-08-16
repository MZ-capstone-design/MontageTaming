# VQGAN

Transformer

- 기존 RNN 기반의 encoder-decoder 구조에서 RNN 모듈을 버리고 attention만을 이용하여 번역 등의 task를 수행하는 모델 구조
- 임베딩 된 Input 시퀀스를 Encoder-Decoder 구조에 태워 auto-regressive하게 output 출력
    
![image](https://github.com/user-attachments/assets/34f80e6d-c800-4ef5-977a-10b66665df43)


### VAE (Variational Auto Encoder)

- Autoencoder 구조를 활용하지만 데이터 x에 대한 사후확률 p(x|z)를 추정하기 위해 변분추론을 활용한 다른 tractable한 분포 q(z|x)를 추정
    - latent feature z를 구하고 이로부터 샘플링 생성
- Input → 시그마를 바꿔가며 output 생성(생성 모델)

![image](https://github.com/user-attachments/assets/544a9065-3e11-4e22-9500-ad2501a463fd)

### VQ-VAE

- VAE와 다르게 posterior(p(zlx))와 prior(p(z))를 가우시안 분포가 아닌 categorical 분포로 정의하며, 샘플링 되는 결과값 또한 특정 분포로부터의 continuous한 vector가 아니라 embedding table의 특정 위치
- 입력값 즉, 이미지는 encoder를 통과해 임베딩 값 Ze로 생성되며 유클리디안 거리를 기반으로 코드북 내에서 가장 유사한 임베딩 벡터의 인덱스 (1~K)를 반환
- 결과적으로 각 임베딩 값은 가장 유사한 임베딩 벡터로 대치되며, 이를 decoder에 입력하여 이미지를 생성
    
![image](https://github.com/user-attachments/assets/1f8c48c3-160f-48be-9f38-83a86cfc1557)
    

### VQ-VAE Training

- Indexing 및 codebook의 임베딩으로 대치하는 과정은 gradient의 back-propagation이 어려우므로 decoder의 gradient를 인코더로 back-propagation 수행
    - Encoder의 output과 Code의 임베딩은 동일한 차원에 존재하기 때문
- sg (stop gradient) : Embedding loss와 commitment loss 계산 시 특정 term에 대해 gradient를 역전파 하지 않도록 함

![image](https://github.com/user-attachments/assets/e0a68a20-8239-4314-9359-469b7ed46eb9)

### pytorch에서의 역전파 제거 ⇒ detach() 사용

![image](https://github.com/user-attachments/assets/5bdc3e0c-1d9f-42ef-9327-462344f59f26)

### VQ-GAN

- CNN의 locality bias와 Transformer의 global relation modeling 특성을 함께 활용하여 high-resolution image 생성을 할 수 없을까?하는 의문에서 시작

![image](https://github.com/user-attachments/assets/e7decb70-a3d8-4965-a14e-8c889eb0027b)

CNN : locality 특성 파악에 강점
Transformer : global한 모델 번역에 강점

### Architecture

- 이미지로부터 discrete한 latent embedding을 추출할 수 있도록 codebook(Z) 학습 후 codebook의 index를 예측하는 autoregressive한 방식으로 Image synthesis 수행
- 크게 visual feature를 잘 추출할 수 있는 codebook을 학습시키기 위한 과정(1), 해당 코드북을 기반으로 트랜스포머를 활용한 이미지 분석 과정으로 구분

![image](https://github.com/user-attachments/assets/90bd0c0f-305c-41bd-a755-33ac3660e5ea)

- 차이 : Discriminator를 추가하여 보다 현실적인 이미지를 생성하도록 함

### 학습

- VQVAE와 유사한 방식으로 이미지로부터 discrete한 latent embedding을 추출할 수 있도록 codebook 학습
- 단 perceptual loss를 활용한다는 점, discriminator를 이용한 adversarial training 방식으로 진행된다는 점에서 VQVAE와 차이를 보임

![image](https://github.com/user-attachments/assets/5ff26442-8f1a-43b4-8937-759104c18829)

```python
if self.perceputual_weight > 0 : 
		p_loss = self.perceptual_loss(inputs.configuous(), reconstrunctions.contigous())
		rec_lost = rec_loss + self.perceptual_weight * p_loss		
```

### 코드북-최종

![image](https://github.com/user-attachments/assets/e7fbd395-a222-47e3-94a7-fae5e040722e)

- 람다 부분
    - Reconstruction Loss가 지나치게 커져 영향을 미치게 하는 것을 방지하기 위해 GAN을 키움
        - 반대 상황도 마찬가지로 서로의 가중치가 서로에게 영향이 가도록 조정

### 학습된 코드북에 의해 index 집합 추출 ⇒ 트랜스포머

![image](https://github.com/user-attachments/assets/8a50b519-cbb3-4c2f-97bc-98824c8bdc86)

- 코드북의 양이 많아진다면 연산량이 많아져 특정 크기 윈도우를 슬라이딩하여 앞선 인덱스를 바라보게 하여 합성을 진행함

내용 정리 출처 : https://www.youtube.com/watch?v=xR6Uw_3Huss&t=340s
