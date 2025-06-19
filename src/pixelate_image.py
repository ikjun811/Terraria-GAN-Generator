# pixelate_image.py

import os
from PIL import Image
import argparse

def pixelate(image_path, output_path, scale_factor):
    """
    이미지를 낮은 해상도로 축소했다가 다시 확대하여 픽셀화 효과를 줍니다.
    """
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다 - {image_path}")
        return

    # 원본 이미지 크기
    original_width, original_height = image.size
    
    # 낮은 해상도로 축소
    # Image.Resampling.NEAREST는 가장 가까운 픽셀 값을 사용하는 방식으로, 픽셀 느낌을 살립니다.
    small_image = image.resize(
        (int(original_width / scale_factor), int(original_height / scale_factor)),
        Image.Resampling.NEAREST
    )
    
    # 다시 원본 크기로 확대
    pixelated_image = small_image.resize(
        (original_width, original_height),
        Image.Resampling.NEAREST
    )
    
    # 결과 저장
    pixelated_image.save(output_path)
    print(f"픽셀화된 이미지가 여기에 저장되었습니다: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Apply a pixelation effect to an image.")
    parser.add_argument("input_image", type=str, help="Path to the input image file.")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path to save the output image. (Defaults to 'pixelated_[input_name]')")
    parser.add_argument("--scale", "-s", type=int, default=4, help="Pixelation scale factor. Higher means more blocky. (e.g., 2, 4, 8)")
    
    args = parser.parse_args()

    if args.output is None:
        # 출력 파일 이름 자동 생성
        path, filename = os.path.split(args.input_image)
        name, ext = os.path.splitext(filename)
        args.output = os.path.join(path, f"{name}_pixelated{ext}")

    pixelate(args.input_image, args.output, args.scale)