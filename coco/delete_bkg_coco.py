import argparse
import collections as cl
import json

from pycocotools.coco import COCO


def main():
    args = get_args()
    delete_images_with_bkg(args.coco_input_file, args.coco_output_file)


def delete_images_with_bkg(input_coco_file_name: str, output_coco_file_name: str):

    coco_data = COCO(input_coco_file_name)
    coco_out = coco_data.dataset
    coco_out_image_list = []

    for coco_image_info_dict in coco_out["images"]:
        id_img = coco_image_info_dict["id"]
        id_annos = coco_data.getAnnIds(imgIds=id_img, iscrowd=None)
        annos = coco_data.loadAnns(id_annos)
        if len(annos) > 0:
            coco_out_image_list.append(coco_image_info_dict)

    coco_out["images"] = coco_out_image_list
    fw = open(output_coco_file_name, "w")
    json.dump(coco_out, fw, indent=2)


def get_args():
    parser = argparse.ArgumentParser(
        description="Delete images that do not have labels."
    )
    parser.add_argument("coco_input_file", help="Path of coco file")
    parser.add_argument("coco_output_file", help="Path of output coco file, e.g., instances.json")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
