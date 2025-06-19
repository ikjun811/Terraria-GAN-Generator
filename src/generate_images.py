# generate_images.py

# python src/generate_images.py output/netG_epoch_350.pth --num_images 1

# generate_images.py
import torch
import torch.nn as nn
import torchvision.utils as vutils
import os
import argparse

# ===============================================================
# 1. 설정 (학습 스크립트의 모델 구조와 관련된 값과 일치해야 함)
# ===============================================================
nz = 100 # 잠재 공간 벡터 크기
ngf = 96 # 생성자 특징 맵 크기 (terra_gan.py와 동일)
nc = 3   # 이미지 채널 수
ngpu = 1 # 사용할 GPU 개수

# ===============================================================
# 2. 생성자 모델 정의
# ===============================================================
# 모델을 저장했을 때의 구조와 동일
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        def upsample_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            )

        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            upsample_block(ngf * 8, ngf * 4),
            upsample_block(ngf * 4, ngf * 2),
            upsample_block(ngf * 2, ngf),
            nn.Dropout(0.5),
            upsample_block(ngf, ngf // 2),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf // 2, nc, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# ===============================================================
# 3. 메인 실행 로직
# ===============================================================
def main():
    # 터미널에서 사용자 입력을 받기 위한 파서 설정
    parser = argparse.ArgumentParser(description="Generate images using a trained Terraria GAN model.")
    parser.add_argument("model_path", type=str, help="Path to the trained generator model (.pth file).")
    parser.add_argument("--num_images", type=int, default=64, help="Number of images to generate.")
    parser.add_argument("--output_dir", type=str, default="generated_images", help="Directory to save the generated images.")
    args = parser.parse_args()

    # 스크립트 위치를 기준으로 상대 경로를 절대 경로로 변환
    script_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_path)
    
    # 입력된 경로가 상대 경로일 경우, 프로젝트 루트를 기준으로 경로를 조합
    model_path_abs = args.model_path if os.path.isabs(args.model_path) else os.path.join(project_root, args.model_path)
    output_dir_abs = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(project_root, args.output_dir)

    os.makedirs(output_dir_abs, exist_ok=True)
    output_filename = os.path.basename(model_path_abs).replace('.pth', '.png')

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(f"Using device: {device}")

    # 빈 생성자 모델 구조를 만듦
    netG = Generator(ngpu).to(device)

    # 학습된 가중치(.pth 파일)를 불러와 모델에 덮어씌움
    print(f"Loading model from: {model_path_abs}")
    netG.load_state_dict(torch.load(model_path_abs, map_location=device))

    # 모델을 '평가 모드'로 전환
    netG.eval()

    # 실제 이미지 생성
    with torch.no_grad():
        noise = torch.randn(args.num_images, nz, 1, 1, device=device)
        fake_images = netG(noise).detach().cpu()

    # 생성된 이미지를 파일로 저장
    save_path = os.path.join(output_dir_abs, output_filename)
    vutils.save_image(fake_images, save_path, normalize=True, padding=2)

    print(f"Successfully generated {args.num_images} images and saved to: {save_path}")

if __name__ == '__main__':
    main()