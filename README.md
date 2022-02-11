# Adversarial-Training-
Pytorch's tutorial

##library
![image](https://user-images.githubusercontent.com/73246476/153521941-bba2b912-6571-44ec-b3cc-01c7ffcd018d.png)
![image](https://user-images.githubusercontent.com/73246476/153521977-f84f6801-5503-4d15-906c-db376ae45fe2.png)
![image](https://user-images.githubusercontent.com/73246476/153521991-68da9df8-c2f2-4a24-8953-1337a0d84362.png)

## reference
> Pytorch's tutorial
> https://tutorials.pytorch.kr/beginner/fgsm_tutorial.html

## data set
"MNIST 손글싸 data set"
>https://sdc-james.gitbook.io/onebook/4.-and/5.1./5.1.3.-mnist-dataset

## Theoretical background
- **Adversarial Training** : 보통의 network와는 달리, 원본 이미지 x에 대한 손실함수를 계산하는 것이 아닌, 
                                           원본 이미지 x에 epsilon 만큼의 교란을 준 이미지에 대한 손실함수를 계산하여 훈련시키는 것.
- **장점** : worst case인 적대적 예제(Xadv)를 입력 값으로 주어도 이와 같은 방식으로 훈련되지 않은 모델(원본 모델)보다 우수한 성능을 보임.
- ![image](https://user-images.githubusercontent.com/73246476/153522287-9a800a22-e6f0-4d8c-a023-14692bad6aca.png)

## Result
![image](https://user-images.githubusercontent.com/73246476/153522324-87c0d0c8-f33a-4750-810e-f067bfa21d9b.png)

▲ 실행 결과
Epslion이 클수록 accuarcy가 낮아지는 모습을 관찰할 수 있음.
(∵ epsilon에 비례하여 down grad된 이미지를 학습시키므로)

![image](https://user-images.githubusercontent.com/73246476/153522370-b5a4fd61-9939-4a1e-9db2-c5a474044b54.png)

▲ Adversarial Training한 모델과 그렇지 않은 일반적인 방법으로 학습한 모델간의 성능 차이
![image](https://user-images.githubusercontent.com/73246476/153522449-866fdd10-d583-4de2-b71f-21fbf64935ce.png)


