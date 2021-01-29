# PyramidNet-CIFAR-10-competition-
KHU ML course CIFAR-10 competition(2020.11.)

# 선택 모델
PyramidNet 
  Residual network 기반의 모델의 경우,
depth가 깊어질 때 width가 절반으로 감소하고 channel을 두배로 늘리는 것이 일반적인데
이 때의 급격한 width의 변화를 감소시키는 것을 중점으로 한 모델이다.
Alpha값에 따라 layer마다 점진적으로 증가시킬 width의 정도를 결정하게 된다.
모델 parameter가 200만개 이하로 제한된 상태에서
depth와 alpha값으로 효율적으로 모델 구조와 크기를 변형하며 실험할 수 있기에
PyramidNet을 선택하게 되었다.

# 조건
1. 파라미터 개수 200만개 이하
2. 주어진 dataset만 활용

# 사용 기법
1. Cut-Mix
2. Label-Smoothing

# 결과
TEST acc : 92.325%
