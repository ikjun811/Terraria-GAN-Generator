# python src/pixelate_image.py generated_images/netG_epoch_399.png
# python src/pixelate_image.py generated_images/netG_epoch_350.png


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

    # 프로젝트 루트 폴더를 기준으로
    script_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_path)

    # 입력 경로가 상대 경로이면, 프로젝트 루트를 기준으로 절대 경로
    input_path_abs = args.input_image if os.path.isabs(args.input_image) else os.path.join(project_root, args.input_image)

    # 출력 경로 처리
    if args.output is None:
        path, filename = os.path.split(input_path_abs)
        name, ext = os.path.splitext(filename)
        output_path_abs = os.path.join(path, f"{name}_pixelated{ext}")
    else:
        output_path_abs = args.output if os.path.isabs(args.output) else os.path.join(project_root, args.output)

    pixelate(input_path_abs, output_path_abs, args.scale)