# cd C:\Users\0320\Documents\GitHub\Terraria-GAN-Generator
# .\venv\Scripts\Activate.ps1
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
# python src/terra_gan.py

# python src/generate_images.py models/snow_biome/netG_epoch_299.pth
# python src/generate_images.py models/snow_biome/netG_epoch_299.pth --num_images 1
# python src/generate_images.py models/snow_biome/netG_epoch_299.pth --num_images 16 --output_dir my_favorite_images

# terra_gan.py
# -*- coding: utf-8 -*-
# 필요한 라이브러리들을 불러옵니다.
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ===============================================================
# 1. 환경 설정 및 하이퍼파라미터 정의
# ===============================================================

# --- 경로 및 시드 설정 ---
# 현재 스크립트 파일의 절대 경로를 가져옵니다.
script_path = os.path.dirname(os.path.abspath(__file__))
# 프로젝트 루트 폴더는 스크립트 위치의 한 단계 상위 폴더입니다.
project_root = os.path.dirname(script_path)

# 프로젝트 루트를 기준으로 상대 경로를 설정합니다.
dataroot = os.path.join(project_root, "data", "terraria_patches")
output_dir = os.path.join(project_root, "output")

os.makedirs(output_dir, exist_ok=True) # 결과 폴더가 없으면 생성

manualSeed = 999 # 재현성을 위한 무작위 시드 고정
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# 재현성을 위해 cuDNN의 비결정적 알고리즘을 끕니다.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- 학습 하이퍼파라미터 ---
workers = 0          # 데이터 로딩에 사용할 스레드 수 
batch_size = 64      # 한 번에 학습할 이미지 수
image_size = 128     # 모든 이미지의 크기를 128x128로 통일
nc = 3               # 이미지의 컬러 채널 수 (RGB)
nz = 100             # 생성자 입력으로 사용될 잠재 공간 벡터의 크기
ngf = 96             # 생성자 내부의 특징 맵 크기를 조절 (모델의 표현력)
ndf = 96             # Critic(판별자) 내부의 특징 맵 크기를 조절 (모델의 판별력)
num_epochs = 400     # 전체 데이터셋을 몇 번 반복하여 학습할지 결정
lr = 0.0001          # 학습률 (WGAN-GP는 낮은 값이 안정적)
ngpu = 1             # 사용한 GPU 개수

# --- WGAN-GP 전용 하이퍼파라미터 ---
lambda_gp = 10  # Gradient Penalty의 가중치 (Critic의 안정성을 위해)
n_critic = 3    # 생성자 1번 업데이트 당 Critic 5번 업데이트 (학습 균형)
beta1 = 0.5     # Adam 옵티마이저의 beta1 값
beta2 = 0.9     # Adam 옵티마이저의 beta2 값

# ===============================================================
# 2. 데이터 준비
# ===============================================================

# ImageFolder를 사용하여 데이터셋을 구성하고, 여러 변환(transform)을 적용합니다.
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),       # 이미지 크기 통일
                               transforms.CenterCrop(image_size),   # 중앙을 기준으로 자르기
                               transforms.RandomHorizontalFlip(p=0.5), # 50% 확률로 좌우 반전 (데이터 증강)
                               transforms.ToTensor(),               # 이미지를 PyTorch 텐서로 변환 ([0, 1] 범위)
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 픽셀 값을 [-1, 1] 범위로 정규화
                           ]))

# DataLoader: 데이터셋을 배치 단위로 묶고, 섞어서 모델에 전달하는 역할
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# 학습에 사용할 장치 설정 (GPU 우선)
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print("Using device:", device)

# 학습 시작 전, 원본 데이터 샘플을 이미지 파일로 저장하여 확인
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images Sample")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.savefig(os.path.join(output_dir, "training_images_sample.png"))
plt.close() # 스크립트 실행이 멈추지 않도록 창을 닫음

# ===============================================================
# 3. 모델 아키텍처 정의
# ===============================================================

# 모든 모델의 가중치를 정규분포를 따르도록 초기화하는 함수
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# 생성자 (Generator) 정의: 노이즈로부터 이미지를 생성
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        # 'Upsample + Conv2d' 블록을 함수로 정의하여 재사용
        def upsample_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'), # 픽셀을 복제하여 이미지를 2배 확대 (선명도 유지)
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), # 합성곱으로 특징 학습
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            )

        self.main = nn.Sequential(
            # 1단계: 입력 Z(nz x 1 x 1)를 4x4 크기의 특징 맵으로 변환
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # 2~5단계: 업샘플링 블록을 반복하여 이미지 크기를 점진적으로 확대
            upsample_block(ngf * 8, ngf * 4), # 4x4 -> 8x8
            upsample_block(ngf * 4, ngf * 2), # 8x8 -> 16x16
            upsample_block(ngf * 2, ngf),     # 16x16 -> 32x32
            nn.Dropout(0.5),                  # 과적합 방지를 위한 Dropout
            upsample_block(ngf, ngf // 2),    # 32x32 -> 64x64

            # 6단계 (최종): 128x128 크기의 최종 이미지 생성
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf // 2, nc, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh() # 출력을 [-1, 1] 범위로 맞춰줌
        )

    def forward(self, input):
        return self.main(input)

# Critic(판별자) 정의: 이미지가 얼마나 '실제 같은지' 점수를 매김
class Critic(nn.Module):
    def __init__(self, ngpu):
        super(Critic, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 입력: (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 여러 합성곱 레이어를 거치며 이미지의 특징을 추출하고 크기를 줄임
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 2, affine=True), # WGAN-GP 안정성을 위한 Instance Normalization
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # 최종적으로 1x1 크기의 '점수'(스칼라 값)를 출력
            nn.Conv2d(ndf * 8, 1, kernel_size=8, stride=1, padding=0, bias=False)
            # WGAN-GP는 확률(0~1)이 아니므로 Sigmoid 함수를 사용하지 않음
        )
    def forward(self, input):
        return self.main(input)

# WGAN-GP의 핵심인 Gradient Penalty를 계산하는 함수
def compute_gradient_penalty(critic, real_samples, fake_samples):
    alpha = torch.randn(real_samples.size(0), 1, 1, 1, device=device) # 무작위 혼합 비율
    # 실제와 가짜 이미지 사이의 임의의 지점을 샘플링
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    c_interpolates = critic(interpolates)
    
    fake_output = torch.ones(c_interpolates.size(), device=device, requires_grad=False)
    
    # 해당 지점에서의 기울기(gradient) 계산
    gradients = torch.autograd.grad(
        outputs=c_interpolates, inputs=interpolates, grad_outputs=fake_output,
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    # 기울기의 크기가 1에 가깝도록 강제하는 패널티 계산
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# ===============================================================
# 4. 모델 및 옵티마이저 초기화
# ===============================================================
# 생성자 인스턴스 생성 및 가중치 초기화
netG = Generator(ngpu).to(device)
netG.apply(weights_init)
print(netG)

# Critic 인스턴스 생성 및 가중치 초기화
netC = Critic(ngpu).to(device)
netC.apply(weights_init)
print(netC)

# WGAN-GP용 Adam 옵티마이저 설정
optimizerC = optim.Adam(netC.parameters(), lr=lr / 4 , betas=(beta1, beta2))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))

# 생성 과정 시각화를 위한 고정된 노이즈 벡터
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# ===============================================================
# 5. WGAN-GP 학습 루프
# ===============================================================
img_list = []
G_losses = []
C_losses = []
iters = 0

print("Starting WGAN-GP Training Loop...")
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        # ---------------------
        #  (1) Critic 학습
        # ---------------------
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)

        # Critic을 생성자보다 더 많이 학습시켜 균형을 맞춤 (n_critic 만큼)
        for _ in range(n_critic):
            optimizerC.zero_grad()
            
            # 가짜 이미지 생성
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            
            # 실제와 가짜 이미지에 대한 Critic의 점수 계산
            real_output = netC(real_cpu).view(-1)
            fake_output = netC(fake.detach()).view(-1)

            # Gradient Penalty 계산
            gradient_penalty = compute_gradient_penalty(netC, real_cpu.data, fake.data)
            
            # 최종 Critic 손실 계산 (Wasserstein 거리 + Gradient Penalty)
            errC = -torch.mean(real_output) + torch.mean(fake_output) + lambda_gp * gradient_penalty
            errC.backward()
            optimizerC.step()

        # ---------------------
        #  (2) 생성자 학습
        # ---------------------
        optimizerG.zero_grad()
        
        # 새로운 가짜 이미지를 Critic이 '진짜처럼' 느끼도록 학습
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)
        output = netC(fake).view(-1)
        
        # 최종 생성자 손실 계산 (Critic의 점수를 최대화하는 방향)
        errG = -torch.mean(output)
        errG.backward()
        optimizerG.step()
        
        # 학습 진행 상황 출력
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_C: %.4f\tLoss_G: %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errC.item(), errG.item()))

        # 손실 값 기록
        G_losses.append(errG.item())
        C_losses.append(errC.item())

        # 일정 간격으로 생성된 이미지 샘플 저장
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            vutils.save_image(img_list[-1], '%s/fake_samples_iter_%05d.png' % (output_dir, iters))

        iters += 1
    
    # 매 에폭이 끝날 때마다 모델 가중치 저장
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (output_dir, epoch))
    torch.save(netC.state_dict(), '%s/netC_epoch_%d.pth' % (output_dir, epoch))

# ===============================================================
# 6. 최종 결과 시각화 및 저장
# ===============================================================
print("Training finished. Saving results...")

# 손실 그래프 저장
plt.figure(figsize=(10,5))
plt.title("Generator and Critic Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(C_losses,label="C")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(output_dir, "loss_graph.png"))
plt.close()

# 최종 생성 이미지와 실제 이미지 비교 저장
real_batch = next(iter(dataloader))
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.savefig(os.path.join(output_dir, "real_vs_fake.png"))
plt.close()

print("All results saved in 'output' folder.")