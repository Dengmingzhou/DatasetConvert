import os
import json
from tqdm import tqdm


def convert_labelme_json_to_yolo_txt(input_json_dir, out_txt_dir, classes):
    if not os.path.exists(input_json_dir):
        print("json输入文件夹不存在")
        return

    if not os.path.exists(out_txt_dir):
        os.mkdir(out_txt_dir)

    # 获取输入文件夹下所有json文件
    all_json_files = []
    for filename in os.listdir(input_json_dir):
        if filename.endswith(".json"):
            # file_path = os.path.join(input_json_dir, filename)
            # all_json_files.append(file_path)
            all_json_files.append(filename)

    # 遍历json文件 进行处理
    for idx, single_json_filename in tqdm(enumerate(all_json_files)):
        single_json_path = os.path.join(input_json_dir, single_json_filename)

        print("正在处理第{}个文件{}".format(idx, single_json_path))

        with open(single_json_path, "r") as single_json_load_file:
            json_data = json.load(single_json_load_file)

        image_width = json_data["imageWidth"]
        image_height = json_data["imageHeight"]
        shapes = json_data["shapes"]

        out_txt_file_path = os.path.join(
            out_txt_dir, single_json_filename.replace("json", "txt")
        )
        if os.path.exists(out_txt_file_path):
            try:
                os.remove(out_txt_file_path)
                print("{}已存在，删除文件".format(out_txt_file_path))
            except OSError as e:
                print("删除文件{}出错，错误：{}".format(out_txt_file_path, e))

        out_txt_file = open(out_txt_file_path, "w")

        for shape_data in shapes:
            label = shape_data["label"]
            labe_index = classes.index(label)
            shape_type = shape_data["shape_type"]
            points = shape_data["points"]

            x_min, y_min, x_max, y_max = get_polygon_box(points)

            # 将标注框按图像大小压缩
            x_center = (x_min + x_max) / 2 / image_width
            y_center = (y_min + y_max) / 2 / image_height

            bbox_w = (x_max - x_min) / image_width
            bbox_h = (y_max - y_min) / image_height

            bbox = (x_center, y_center, bbox_w, bbox_h)

            out_txt_file.writelines(
                str(labe_index) + " " + " ".join([str(a) for a in bbox]) + "\n"
            )


def get_polygon_box(points):
    point_x_list = []
    point_y_list = []

    for point in points:
        point_x = point[0]
        point_y = point[1]
        point_x_list.append(point_x)
        point_y_list.append(point_y)

    x_min = min(point_x_list)
    x_max = max(point_x_list)
    y_min = min(point_y_list)
    y_max = max(point_y_list)

    return x_min, y_min, x_max, y_max


# 创建yaml文件
def create_yaml_file(classes, yaml_path):
    with open(yaml_path, "w") as f:
        f.write("train: ./train\n")
        f.write("val: ./val\n")
        f.write("nc: {}\n".format(len(classes)))
        f.write("names: {}\n".format(classes))


if __name__ == "__main__":
    input_json_path = "/Volumes/Mobile/Home/Data/Layout/CDLA/CDLA_DATASET"
    output_txt_path = "/Volumes/Mobile/Home/Data/Layout/CDLA/CDLA_DATASET_YOLO"

    classes_list = [
        "Header",
        "Text",
        "Reference",
        "Figure caption",
        "Figure",
        "Table caption",
        "Table",
        "Title",
        "Footer",
        "Equation",
    ]

    yaml_path = os.path.join(output_txt_path, "layout.yaml")
    input_train_dir = os.path.join(input_json_path, "train")
    output_train_dir = os.path.join(output_txt_path, "train")

    input_val_dir = os.path.join(input_json_path, "val")
    output_val_dir = os.path.join(output_txt_path, "val")

    convert_labelme_json_to_yolo_txt(
        input_json_dir=input_train_dir,
        out_txt_dir=output_train_dir,
        classes=classes_list,
    )

    convert_labelme_json_to_yolo_txt(
        input_json_dir=input_val_dir,
        out_txt_dir=output_val_dir,
        classes=classes_list,
    )

    create_yaml_file(
        classes=classes_list,
        yaml_path=os.path.join(output_txt_path, "layout.yaml"),
    )
