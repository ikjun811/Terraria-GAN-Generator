# streamlit run app.py

# app.py

import streamlit as st
import torch
import torch.nn as nn
import torchvision.utils as vutils
import os
from PIL import Image
import io
import torchvision.transforms as transforms
import base64

# ===============================================================
# 0. 페이지 기본 설정 및 스타일 적용
# ===============================================================

st.set_page_config(
    page_title="Terraria Map Generator",
    page_icon="assets/logo.png",
    layout="wide"
)

# --- CSS 및 배경 적용 함수 ---
@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    try:
        bin_str = get_base64_of_bin_file(png_file)
        with open("style.css", "r", encoding="utf-8") as f:
            css_code = f.read().replace("{img_base64}", bin_str)
            st.markdown(f"<style>{css_code}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("스타일시트 파일(style.css) 또는 배경 이미지(assets/background.png)를 찾을 수 없습니다.")

set_background('assets/background.png')

# ===============================================================
# 1. 모델 아키텍처 및 함수 정의 (가장 먼저 선언)
# ===============================================================

# --- 고정된 하이퍼파라미터 ---
NGF_VALUE = 96
NZ_VALUE = 100
NC_VALUE = 3
NGPU_VALUE = 1
PIXELATION_SCALE_FACTOR = 2

# --- 생성자 모델 정의 ---
class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
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

# --- 이미지 생성 함수 ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@st.cache_data
def generate_image(model_path, num_images):
    model = Generator(ngpu=NGPU_VALUE, nz=NZ_VALUE, ngf=NGF_VALUE, nc=NC_VALUE)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    with torch.no_grad():
        noise = torch.randn(num_images, NZ_VALUE, 1, 1, device=device)
        fake_images_tensor = model(noise).detach().cpu()
    return fake_images_tensor

# --- 픽셀화 함수 ---
def pixelate_image(image_tensor):
    grid = vutils.make_grid(image_tensor, padding=0, normalize=True)
    pil_img = transforms.ToPILImage()(grid)
    original_width, original_height = pil_img.size
    small_image = pil_img.resize((original_width // PIXELATION_SCALE_FACTOR, original_height // PIXELATION_SCALE_FACTOR), Image.Resampling.NEAREST)
    pixelated_image = small_image.resize((original_width, original_height), Image.Resampling.NEAREST)
    return pixelated_image

# ===============================================================
# 2. 사이드바 UI 구성
# ===============================================================
with st.sidebar:
    try:
        st.image("assets/logo.png")
    except FileNotFoundError:
        st.title("🗺️ Terraria Map Generator")
    
    st.markdown("---")
    st.header("1. 모델 선택")

    model_dir = "models/snow_biome"
    try:
        model_files = [f for f in os.listdir(model_dir) if f.startswith("netG_") and f.endswith(".pth")]
        model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
        selected_model_file = st.selectbox(
            "학습된 모델(에폭) 선택:", model_files,
            help="에폭이 높을수록 더 오래 학습되었지만, 과적합될 수 있습니다. 200~300 에폭을 추천합니다."
        )
        model_path = os.path.join(model_dir, selected_model_file)
    except FileNotFoundError:
        st.error(f"'{model_dir}' 폴더를 찾을 수 없습니다.")
        model_path = None

    st.markdown("---")
    st.header("2. 생성 옵션")
    num_images = st.selectbox("생성할 맵 조각 개수:", (1, 4, 9, 16, 64), index=3)
    
    st.markdown("---")
    st.header("3. 후처리 옵션")
    apply_pixelation = st.checkbox("픽셀화 효과 (권장)", value=True)

# ===============================================================
# 3. 메인 화면 UI 및 로직 실행
# ===============================================================
st.title("Terraria Map Generator")
st.write("좌측 사이드바에서 옵션을 선택하고 아래 버튼을 클릭하여 새로운 맵을 생성합니다.")

if st.sidebar.button("💎 새로운 맵 생성", type="primary"):
    if model_path:
        with st.spinner("월드 생성 중... ⛏️"):
            generated_tensor = generate_image(model_path, num_images)
            
            if apply_pixelation:
                final_image = pixelate_image(generated_tensor)
            else:
                grid = vutils.make_grid(generated_tensor, padding=2, normalize=True)
                final_image = transforms.ToPILImage()(grid)
            
        st.image(final_image, caption=f"생성된 지하 설원 (모델: {selected_model_file})", use_container_width=True)
        
        buf = io.BytesIO()
        final_image.save(buf, format="PNG")
        byte_im = buf.getvalue()
        download_filename = f"terraria_map_{selected_model_file.replace('.pth', '')}.png"
        st.download_button("💾 이미지 저장", byte_im, download_filename, "image/png")
    else:
        st.warning("모델 파일을 찾을 수 없습니다.")
else:
    st.info("좌측의 '새로운 맵 생성' 버튼을 클릭하여 시작하세요.")