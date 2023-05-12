import argparse
import os
import yaml

import albumentations as A
import cv2
import numpy as np
import tqdm
from draw_bounding_box_coco import draw_bb_coco
from libs.coco_dumper import COCODumper
from matplotlib import pyplot as plt
from pycocotools.coco import COCO

BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White


def main():
    args = argparser()
    aug_transform = create_aug_transform(args.yaml)
    run_augmentation(args.coco_json, args.img_dir, args.out_dir, args.total_num_img, aug_transform, args.vis)


def run_augmentation(coco_file, img_dir, out_dir, total_num_img, aug_transform, flag_visualize=False):
    out_img_dir = os.path.join(out_dir, "images")
    out_debug_dir = os.path.join(out_dir, "debug")
    coco_out_file_path = os.path.join(out_dir, "instances.json")
    os.makedirs(out_img_dir, exist_ok=True)

    cocoGt = COCO(coco_file)
    categories = cocoGt.loadCats(cocoGt.getCatIds())
    category_id_to_name = {cat["id"]: cat["name"] for cat in categories}
    coco_dumper = COCODumper(
        img_dir,
        coco_out_file_path,
        category_id_to_name.values(),
    )

    img_ids = cocoGt.getImgIds()
    count = 0
    flg_break = False
    while True:
        for img_id in tqdm.tqdm(img_ids):
            img_info = cocoGt.loadImgs(ids=img_id)[0]

            img_cv2 = cv2.imread(os.path.join(img_dir, os.path.basename(img_info["file_name"])))
            img_h, img_w, _ = img_cv2.shape
            # img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
            id_gts = cocoGt.getAnnIds(imgIds=img_id, iscrowd=None)

            gts = cocoGt.loadAnns(id_gts)
            bboxes = [gt["bbox"] for gt in gts]
            for bbox in bboxes:
                bbox[0] = np.clip(bbox[0], 0, img_w - bbox[2])
                bbox[1] = np.clip(bbox[1], 0, img_h - bbox[3])

            category_ids = [gt["category_id"] for gt in gts]
            transformed = aug_transform(image=img_cv2, bboxes=bboxes, category_ids=category_ids)
            out_img_name = os.path.join(out_img_dir, str(count).zfill(8) + ".png")
            cv2.imwrite(out_img_name, transformed["image"])

            coco_dumper.add_one_image_and_add_annotations_per_image(
                os.path.basename(out_img_name), img_w, img_h, transformed["bboxes"]
            )
            count += 1
            if count > total_num_img:
                flg_break = True
                break
        if flg_break is True:
            break
    coco_dumper.dump_json()

    if flag_visualize is True:
        draw_bb_coco(coco_out_file_path, out_img_dir, out_debug_dir)


def create_aug_transform(yaml_file):
    with open(yaml_file) as file:
        cfgs = yaml.safe_load(file)
    aug_transform = A.Compose(
        [
            A.HorizontalFlip(p=cfgs["HorizontalFlip"]["p"]),
            A.MotionBlur(blur_limit=cfgs["MotionBlur"]["blur_limit"], p=cfgs["MotionBlur"]["p"]),
            A.GaussNoise(
                var_limit=(cfgs["GaussNoise"]["var_limit"][0], cfgs["GaussNoise"]["var_limit"][1]),
                p=cfgs["GaussNoise"]["p"],
            ),
            A.ISONoise(
                color_shift=(cfgs["ISONoise"]["color_shift"][0], cfgs["ISONoise"]["color_shift"][1]),
                intensity=(cfgs["ISONoise"]["intensity"][0], cfgs["ISONoise"]["intensity"][1]),
                p=cfgs["ISONoise"]["p"],
            ),
            A.Cutout(
                num_holes=cfgs["Cutout"]["num_holes"],
                max_h_size=cfgs["Cutout"]["max_h_size"],
                max_w_size=cfgs["Cutout"]["max_w_size"],
                p=cfgs["Cutout"]["p"],
            ),
            A.ColorJitter(
                brightness=cfgs["ColorJitter"]["brightness"],
                contrast=cfgs["ColorJitter"]["contrast"],
                saturation=cfgs["ColorJitter"]["saturation"],
                hue=cfgs["ColorJitter"]["hue"],
                p=cfgs["ColorJitter"]["p"],
            ),
            A.ShiftScaleRotate(
                shift_limit=cfgs["ShiftScaleRotate"]["shift_limit"],
                scale_limit=cfgs["ShiftScaleRotate"]["scale_limit"],
                rotate_limit=cfgs["ShiftScaleRotate"]["rotate_limit"],
                border_mode=cv2.BORDER_REPLICATE,
                p=cfgs["ShiftScaleRotate"]["p"],
            ),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["category_ids"]),
    )
    return aug_transform


def visualize(image, bboxes, category_ids, category_id_to_name, image_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    cv2.imwrite(image_name, img)


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(
        img,
        (x_min, y_min - int(1.3 * text_height)),
        (x_min + text_width, y_min),
        BOX_COLOR,
        -1,
    )
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def argparser():
    parser = argparse.ArgumentParser(description="coco_aug")

    parser.add_argument("coco_json", type=str)
    parser.add_argument("img_dir", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("yaml", type=str)
    parser.add_argument("total_num_img", type=int)
    parser.add_argument("--vis", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
