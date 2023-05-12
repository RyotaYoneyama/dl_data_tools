import cv2
import os
import argparse
import tqdm

def get_arguments():
    parser = argparse.ArgumentParser(description="Draw coco bounding boxes")  # 2. パーサを作る

    parser.add_argument("img_dir")
    parser.add_argument("--out", default="output.mp4")
    parser.add_argument("--fps", default=1)
    args = parser.parse_args()

    return args

def video_converter(img_dir, out_video_name, fps):

    image_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]
    image_files.sort()  # Sort the files to maintain the order

    frame_height, frame_width = cv2.imread(os.path.join(img_dir, image_files[0])).shape[:2]
    video_writer = cv2.VideoWriter(out_video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    for image_file in tqdm.tqdm(image_files):
        image_path = os.path.join(img_dir, image_file)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (frame_width, frame_height))
        video_writer.write(image)

    video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = get_arguments()
    video_converter(args.img_dir, args.out, args.fps)
