# streamlit run app.py

import streamlit as st
import torch
import torch.nn as nn
import torchvision.utils as vutils
import os
from PIL import Image
import io

# ===============================================================
# 0. 페이지 기본 설정 및 모델 정의
# ===============================================================

# Streamlit 페이지 설정 (제목, 아이콘 등)
st.set_page_config(
    page_title="Terraria Map Generator",
    page_icon="🗺️",
    layout="wide"
)

# 생성자 모델 정의 (train_gan.py에서 그대로 복사)
# 이 코드는 Streamlit이 모델 가중치를 불러올 때 필요합니다.
# @st.cache_resource 데코레이터를 사용하면 모델을 한 번만 로드하여 속도를 높입니다.
@st.cache_resource
def load_generator_model(ngpu=1, nz=100, ngf=96, nc=3):
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
                nn.BatchNorm2d(ngf * 8), nn.ReLU(True),
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
    
    # 모델 인스턴스 생성
    model = Generator(ngpu)
    return model

# ===============================================================
# 1. 사이드바 UI 구성
# ===============================================================

# st.sidebar를 사용하여 모든 컨트롤을 왼쪽 사이드바에 배치합니다.
with st.sidebar:
    st.title("🗺️ Terraria Map Generator")
    st.markdown("---")
    st.header("1. 모델 선택")

    # output 폴더에서 학습된 모델 파일(.pth) 목록을 자동으로 가져옵니다.
    model_dir = "output"
    try:
        model_files = [f for f in os.listdir(model_dir) if f.startswith("netG_") and f.endswith(".pth")]
        # 에폭 번호를 기준으로 정렬하여 드롭다운에 표시
        model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
        
        selected_model_file = st.selectbox(
            "학습된 모델을 선택하세요:",
            model_files,
            help="에폭(epoch)이 높을수록 더 오래 학습된 모델입니다."
        )
        model_path = os.path.join(model_dir, selected_model_file)
    except FileNotFoundError:
        st.error(f"'{model_dir}' 폴더를 찾을 수 없습니다. 학습된 모델이 있는지 확인하세요.")
        model_path = None

    st.markdown("---")
    st.header("2. 생성 옵션")

    # 생성할 이미지 개수 선택
    num_images = st.selectbox(
        "생성할 맵 조각 개수:",
        (1, 4, 9, 16, 64), # 1x1, 2x2, 3x3, 4x4, 8x8 그리드
        index=0 # 기본값을 1로 설정
    )

    st.markdown("---")
    st.header("3. 후처리 옵션")
    
    # 픽셀화 효과 적용 여부
    apply_pixelation = st.checkbox("픽셀화 효과 적용하기")
    pixelation_scale = 1
    if apply_pixelation:
        pixelation_scale = st.slider(
            "픽셀화 강도:",
            min_value=2,
            max_value=16,
            value=4,
            help="값이 클수록 더 큼직한 픽셀로 보입니다."
        )

# (이후 코드는 다음 단계에서 추가됩니다)