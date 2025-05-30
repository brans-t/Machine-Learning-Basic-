from os.path import join
import numpy as np
from PIL import Image

# 定义加粗文本的函数
def bold_text(text):
    return f"\033[1m{text}\033[0m"

def img2matrix(filename):
    """
    将图像文件转换为 32x32 的二进制矩阵
    :param filename: 图像文件的路径
    :return: 32x32 的二进制矩阵
    """
    try:
        print(bold_text(f"Opening image file: {filename}"))
        img = Image.open(filename)
        print(bold_text("Converting image to grayscale..."))
        img = img.convert('L')
        print(bold_text("Resizing image to 32x32..."))
        img = img.resize((32, 32))
        print(bold_text("Converting image to NumPy array..."))
        img_array = np.array(img)
        print(bold_text("Binarizing image..."))
        # 将像素值二值化：大于128的设为0，小于等于128的设为1
        img_matrix = np.where(img_array > 128, 0, 1)
        return img_matrix
    except Exception as e:
        print(bold_text(f"Error processing image file {filename}: {e}"))
        return None

def save_matrix_to_file(matrix, output_filename):
    """
    将32x32的二进制矩阵保存到文件中，数字之间不留空格
    :param matrix: 32x32 的二进制矩阵
    :param output_filename: 输出文件的名称
    """
    try:
        print(bold_text(f"Writing matrix to file: {output_filename}"))
        with open(output_filename, 'w') as f:
            for row in matrix:
                row_str = ''.join(map(str, row))  # 移除空格
                f.write(row_str + '\n')
    except Exception as e:
        print(bold_text(f"Error writing to file {output_filename}: {e}"))

def main():
    print(bold_text("Starting main function..."))
    input_filename = 'figure\\3_11.png'  # 替换为你的图像文件路径
    output_filename = 'trainingDigits\\3_0.txt'  # 输出文件名

    print(bold_text(f"Converting image {input_filename} to binary matrix..."))
    matrix = img2matrix(input_filename)
    if matrix is not None:
        print(bold_text("32x32二进制矩阵："))
        print(matrix)
        print(bold_text(f"Saving binary matrix to file {output_filename}..."))
        save_matrix_to_file(matrix, output_filename)
        print(bold_text(f"32x32二进制矩阵已保存到文件：{output_filename}"))
    else:
        print(bold_text("未能成功转换图像文件为32x32二进制矩阵。"))

if __name__ == "__main__":
    main()

