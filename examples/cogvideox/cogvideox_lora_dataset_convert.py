import json
import argparse
import os


def process_files_to_jsonl(file1_path, file2_path, output_path):
    try:
        with open(file1_path, 'r', encoding='utf-8') as f1, \
                open(file2_path, 'r', encoding='utf-8') as f2, \
                open(output_path, 'w', encoding='utf-8') as out:

            for line1, line2 in zip(f1, f2):
                line1 = line1.strip()
                line2 = line2.strip()

                data_dict = {
                    "file": line1,
                    "captions": line2
                }

                json_line = json.dumps(data_dict, ensure_ascii=False)
                out.write(json_line + '\n')

        print(f"success! Output file saved to: {output_path}")

    except FileNotFoundError as e:
        print(f"Error: can not find file: {str(e)}")
    except Exception as e:
        print(f"Error: {str(e)}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="", help="video.txt file path")
    parser.add_argument("--prompt_path", type=str, default="", help="prompt.txt file path")
    parser.add_argument("--output_path", type=str, default="./", help="Layer numbers of video_dit")

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    process_files_to_jsonl(args.video_path, args.prompt_path, args.output_path)


if __name__ == "__main__":
    main()
