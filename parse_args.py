#!/usr/env/bin python3
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_img', type=int, default=1000, help="图片数量")

    parser.add_argument('--length', type=int, default=5, help='文字数量')
    parser.add_argument('--watermark_files',type=str, default='./data/watermaker', help='添加水印文件夹')

    parser.add_argument('--clip_max_chars', action='store_true', default=False, help='文字数量是否小于lay的数量')

    parser.add_argument('--img_height', type=int, default=32, help="图片高度")
    parser.add_argument('--img_width', type=int, default=100, help="如果0，图片长度随机")

    parser.add_argument('--chars_file', type=str, default='./data/chars/number.txt', help='字符库')

    parser.add_argument('--config_file', type=str, default='./configs/default.yaml', help='样式设置文件')

    parser.add_argument('--fonts_list', type=str, default='./data/fonts_list/chn.txt', help='字体列表配置文件')

    parser.add_argument('--bg_dir', type=str, default='./data/bg', help="背景图")

    parser.add_argument('--corpus_dir', type=str, default="./data/corpus",
                        help='当 corpus_mode is chn or eng，随机语料库文件夹下的所有文件')

    parser.add_argument('--corpus_mode', type=str, default='random', choices=['random', 'chn', 'eng', 'list'],
                        help='random: 随机抽取字符库文件 chn: 连续抽取语料库 eng: 连续抽取语料库，标签中包含空格')

    parser.add_argument('--output_dir', type=str, default='./output', help='输出文件夹')

    parser.add_argument('--tag', type=str, default='train', help='输出文件夹下子文件夹')

    parser.add_argument('--debug', action='store_true', default=False, help="调试模式")

    parser.add_argument('--viz', action='store_true', default=False)

    parser.add_argument('--strict', action='store_true', default=False, help="在生成图像时检查字体是否支持字符")

    parser.add_argument('--gpu', action='store_true', default=False, help="是否使用gpu")

    parser.add_argument('--num_processes', type=int, default=None, help="如果不填，使用所有cpu核来生产")

    # 处理输出文件夹路径，处理cpu核心数
    flags, _ = parser.parse_known_args()
    flags.save_dir = os.path.join(flags.output_dir, flags.tag)

    if os.path.exists(flags.bg_dir):
        num_bg = len(os.listdir(flags.bg_dir))
        flags.num_bg = num_bg

    if not os.path.exists(flags.save_dir):
        os.makedirs(flags.save_dir)

    if flags.num_processes == 1:
        parser.error("num_processes min value is 2")

    return flags


if __name__ == '__main__':
    args = parse_args()
    print(args.corpus_dir)
