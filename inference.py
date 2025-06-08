import argparse
import os
from functools import partial
from test import create_test_data_loader
from typing import Dict, List, Tuple

import accelerate
import cv2
import numpy as np
import torch
import torch.utils.data as data
from accelerate import Accelerator
from PIL import Image
from tqdm import tqdm

from util.lazy_load import Config
from util.logger import setup_logger
from util.utils import load_checkpoint, load_state_dict
from util.visualize import plot_bounding_boxes_on_image_cv2


def is_image(file_path):
    try:
        img = Image.open(file_path)
        img.close()
        return True
    except:
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="Inference a detector")

    # dataset parameters
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--workers", type=int, default=2)

    # model parameters
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)

    # visualization parameters
    parser.add_argument("--show-dir", type=str, default=None)
    parser.add_argument("--show-conf", type=float, default=0.5)

    # plot parameters
    parser.add_argument("--font-scale", type=float, default=1.0)
    parser.add_argument("--box-thick", type=int, default=1)
    parser.add_argument("--fill-alpha", type=float, default=0.2)
    parser.add_argument("--text-box-color", type=int, nargs="+", default=(255, 255, 255))
    parser.add_argument("--text-font-color", type=int, nargs="+", default=None)
    parser.add_argument("--text-alpha", type=float, default=1.0)

    # engine parameters
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    return args


class InferenceDataset(data.Dataset):
    def __init__(self, root):
        self.images = [os.path.join(root, img) for img in os.listdir(root)]
        self.images = [img for img in self.images if is_image(img)]
        assert len(self.images) > 0, "No images found"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        image = cv2.imdecode(np.fromfile(self.images[index], dtype=np.uint8), -1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        return torch.tensor(image)


# def inference():
#     args = parse_args()
#
#     # set fixed seed and deterministic_algorithms
#     accelerator = Accelerator()
#     accelerate.utils.set_seed(args.seed, device_specific=False)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#     # deterministic in low version pytorch leads to RuntimeError
#     # torch.use_deterministic_algorithms(True, warn_only=True)
#
#     # setup logger
#     for logger_name in ["py.warnings", "accelerate", os.path.basename(os.getcwd())]:
#         setup_logger(distributed_rank=accelerator.local_process_index, name=logger_name)
#
#     dataset = InferenceDataset(args.image_dir)
#     data_loader = create_test_data_loader(
#         dataset, accelerator=accelerator, batch_size=1, num_workers=args.workers
#     )
#
#     # get inference results from model output
#     model = Config(args.model_config).model.eval()
#     checkpoint = load_checkpoint(args.checkpoint)
#     if isinstance(checkpoint, Dict) and "model" in checkpoint:
#         checkpoint = checkpoint["model"]
#     load_state_dict(model, checkpoint)
#     model = accelerator.prepare_model(model)
#
#     with torch.inference_mode():
#         predictions = []
#         for index, images in enumerate(tqdm(data_loader)):
#             prediction = model(images)[0]
#
#             # change torch.Tensor to CPU
#             for key in prediction:
#                 prediction[key] = prediction[key].to("cpu", non_blocking=True)
#             image_name = data_loader.dataset.images[index]
#             image = images[0].to("cpu", non_blocking=True)
#             prediction = {"image_name": image_name, "image": image, "output": prediction}
#             predictions.append(prediction)
#
#     # save visualization results
#     if args.show_dir:
#         os.makedirs(args.show_dir, exist_ok=True)
#
#         # create a dummy dataset for visualization with multi-workers
#         data_loader = create_test_data_loader(
#             predictions, accelerator=accelerator, batch_size=1, num_workers=args.workers
#         )
#         data_loader.collate_fn = partial(_visualize_batch_for_infer, classes=model.CLASSES, **vars(args))
#         [None for _ in tqdm(data_loader)]
#
#
# def _visualize_batch_for_infer(
#     batch: Tuple[Dict],
#     classes: List[str],
#     show_conf: float = 0.0,
#     show_dir: str = None,
#     font_scale: float = 1.0,
#     box_thick: int = 3,
#     fill_alpha: float = 0.2,
#     text_box_color: Tuple[int] = (255, 255, 255),
#     text_font_color: Tuple[int] = None,
#     text_alpha: float = 0.5,
#     **kwargs,  # Not useful
# ):
#     image_name, image, output = batch[0].values()
#     # plot bounding boxes on image
#     image = image.numpy().transpose(1, 2, 0)
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     image = plot_bounding_boxes_on_image_cv2(
#         image=image,
#         boxes=output["boxes"],
#         labels=output["labels"],
#         scores=output.get("scores", None),
#         classes=classes,
#         show_conf=show_conf,
#         font_scale=font_scale,
#         box_thick=box_thick,
#         fill_alpha=fill_alpha,
#         text_box_color=text_box_color,
#         text_font_color=text_font_color,
#         text_alpha=text_alpha,
#     )
#     cv2.imwrite(os.path.join(show_dir, os.path.basename(image_name)), image)

# def inference():
#     args = parse_args()
#     accelerator = Accelerator()
#     accelerate.utils.set_seed(args.seed, device_specific=False)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#
#     for logger_name in ["py.warnings", "accelerate", os.path.basename(os.getcwd())]:
#         setup_logger(distributed_rank=accelerator.local_process_index, name=logger_name)
#
#     dataset = InferenceDataset(args.image_dir)
#     print(f"找到 {len(dataset.images)} 张图片在 {args.image_dir}")
#     data_loader = create_test_data_loader(
#         dataset, accelerator=accelerator, batch_size=1, num_workers=args.workers
#     )
#
#     model = Config(args.model_config).model.eval()
#     checkpoint_path = os.path.abspath(args.checkpoint)
#     print(f"加载权重: {checkpoint_path}")
#     checkpoint = load_checkpoint(checkpoint_path)
#     if checkpoint is None:
#         raise ValueError(f"无法加载权重: {checkpoint_path}")
#     if isinstance(checkpoint, Dict) and "model" in checkpoint:
#         checkpoint = checkpoint["model"]
#     load_state_dict(model, checkpoint)
#     model = accelerator.prepare_model(model)
#
#     with torch.inference_mode():
#         predictions = []
#         for index, images in enumerate(tqdm(data_loader, desc="推理进度")):
#             prediction = model(images)[0]
#             for key in prediction:
#                 prediction[key] = prediction[key].to("cpu", non_blocking=True)
#             image_name = data_loader.dataset.images[index]
#             image = images[0].to("cpu", non_blocking=True)
#             prediction = {"image_name": image_name, "image": image, "output": prediction}
#             predictions.append(prediction)
#
#             # 打印预测结果
#             print(f"\n图片: {os.path.basename(image_name)}")
#             boxes = prediction["output"]["boxes"].numpy()
#             labels = prediction["output"]["labels"].numpy()
#             scores = prediction["output"].get("scores", None)
#             num_detections = sum(1 for score in (scores.numpy() if scores is not None else [None]*len(labels)) if score is None or score >= args.show_conf)
#             print(f"  检测到 {num_detections} 个目标 (阈值: {args.show_conf})")
#             for i, (box, label, score) in enumerate(zip(boxes, labels, scores.numpy() if scores is not None else [None]*len(labels))):
#                 if score is None or score >= args.show_conf:
#                     print(f"    目标 {i+1}: 类别={model.CLASSES[label]}, 框={box.tolist()}, 置信度={score if score is not None else 'N/A'}")
#
#     if args.show_dir:
#         os.makedirs(args.show_dir, exist_ok=True)
#         data_loader = create_test_data_loader(
#             predictions, accelerator=accelerator, batch_size=1, num_workers=args.workers
#         )
#         data_loader.collate_fn = partial(_visualize_batch_for_infer, classes=model.CLASSES, **vars(args))
#         for _ in tqdm(data_loader, desc="可视化进度"):
#             pass
#
#
# def _visualize_batch_for_infer(
#         batch: Tuple[Dict],
#         classes: List[str],
#         show_conf: float = 0.0,
#         show_dir: str = None,
#         font_scale: float = 1.0,
#         box_thick: int = 3,
#         fill_alpha: float = 0.2,
#         text_box_color: Tuple[int] = (255, 255, 255),
#         text_font_color: Tuple[int] = None,
#         text_alpha: float = 0.5,
#         **kwargs,
# ):
#     image_name, image, output = batch[0].values()
#     image = image.numpy().transpose(1, 2, 0)
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     output_path = os.path.join(show_dir, os.path.basename(image_name))
#     num_boxes = len([score for score in (
#         output.get('scores', None).numpy() if output.get('scores') is not None else [None] * len(output['labels'])) if
#                      score is None or score >= show_conf])
#     print(f"可视化 {image_name}，包含 {num_boxes} 个检测框 (阈值: {show_conf})")
#
#     # 即使无检测框也保存原始图片
#     if num_boxes == 0:
#         print(f"  警告: {image_name} 无检测框，保存原始图片")
#         success = cv2.imwrite(output_path, image)
#         print(f"  保存原始图片到: {output_path}, 成功: {success}")
#         return
#
#     # 绘制检测框
#     image = plot_bounding_boxes_on_image_cv2(
#         image=image,
#         boxes=output["boxes"],
#         labels=output["labels"],
#         scores=output.get("scores", None),
#         classes=classes,
#         show_conf=show_conf,
#         font_scale=font_scale,
#         box_thick=box_thick,
#         fill_alpha=fill_alpha,
#         text_box_color=text_box_color,
#         text_font_color=text_font_color,
#         text_alpha=text_alpha,
#     )
#     success = cv2.imwrite(output_path, image)
#     print(f"  保存带 {num_boxes} 个检测框的图片到: {output_path}, 成功: {success}")
#     if not success:
#         print(f"  错误: 保存图片失败，检查目录权限或磁盘空间")

def inference():
    args = parse_args()
    accelerator = Accelerator()
    accelerate.utils.set_seed(args.seed, device_specific=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    for logger_name in ["py.warnings", "accelerate", os.path.basename(os.getcwd())]:
        setup_logger(distributed_rank=accelerator.local_process_index, name=logger_name)

    dataset = InferenceDataset(args.image_dir)
    print(f"找到 {len(dataset.images)} 张图片在 {args.image_dir}: {dataset.images}")
    data_loader = create_test_data_loader(
        dataset, accelerator=accelerator, batch_size=1, num_workers=args.workers
    )

    model = Config(args.model_config).model.eval()
    checkpoint_path = os.path.abspath(args.checkpoint)
    print(f"加载权重: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint is None:
        raise ValueError(f"无法加载权重: {checkpoint_path}")
    if isinstance(checkpoint, Dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    load_state_dict(model, checkpoint)
    model = accelerator.prepare_model(model)

    with torch.inference_mode():
        predictions = []
        for index, images in enumerate(tqdm(data_loader, desc="推理进度")):
            prediction = model(images)[0]
            for key in prediction:
                prediction[key] = prediction[key].to("cpu", non_blocking=True)
            image_name = data_loader.dataset.images[index]
            image = images[0].to("cpu", non_blocking=True)
            prediction = {"image_name": image_name, "image": image, "output": prediction}
            predictions.append(prediction)

            # 打印预测结果
            print(f"\n图片: {os.path.basename(image_name)}")
            boxes = prediction["output"]["boxes"].numpy()
            labels = prediction["output"]["labels"].numpy()
            scores = prediction["output"].get("scores", None)
            num_detections = sum(1 for score in (scores.numpy() if scores is not None else [None]*len(labels)) if score is None or score >= args.show_conf)
            print(f"  检测到 {num_detections} 个目标 (阈值: {args.show_conf})")
            for i, (box, label, score) in enumerate(zip(boxes, labels, scores.numpy() if scores is not None else [None]*len(labels))):
                if score is None or score >= args.show_conf:
                    print(f"    目标 {i+1}: 类别={model.CLASSES[label]}, 框={box.tolist()}, 置信度={score if score is not None else 'N/A'}")

    if args.show_dir:
        print(f"开始可视化，保存到 {args.show_dir}")
        os.makedirs(args.show_dir, exist_ok=True)
        try:
            print(f"预测数量: {len(predictions)}")
            for prediction in tqdm(predictions, desc="可视化进度"):
                print(f"处理预测: {prediction['image_name']}")
                _visualize_batch_for_infer(
                    batch=(prediction,),
                    classes=model.CLASSES,
                    show_conf=args.show_conf,
                    show_dir=args.show_dir,
                    font_scale=args.font_scale,
                    box_thick=args.box_thick,
                    fill_alpha=args.fill_alpha,
                    text_box_color=args.text_box_color,
                    text_font_color=args.text_font_color,
                    text_alpha=args.text_alpha
                )
        except Exception as e:
            print(f"可视化阶段出错: {e}")
    else:
        print("未指定 --show-dir，跳过可视化")

def _visualize_batch_for_infer(
    batch: Tuple[Dict],
    classes: List[str],
    show_conf: float = 0.0,
    show_dir: str = None,
    font_scale: float = 1.0,
    box_thick: int = 3,
    fill_alpha: float = 0.2,
    text_box_color: Tuple[int] = (255, 255, 255),
    text_font_color: Tuple[int] = None,
    text_alpha: float = 0.5,
    **kwargs,
):
    try:
        image_name, image, output = batch[0].values()
        print(f"进入 _visualize_batch_for_infer，处理图片: {image_name}")
        image = image.numpy().transpose(1, 2, 0)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        output_path = os.path.join(show_dir, os.path.basename(image_name))
        num_boxes = len([score for score in (output.get('scores', None).numpy() if output.get('scores') is not None else [None]*len(output['labels'])) if score is None or score >= show_conf])
        print(f"可视化 {image_name}，包含 {num_boxes} 个检测框 (阈值: {show_conf})")

        # 绘制检测框
        image = plot_bounding_boxes_on_image_cv2(
            image=image,
            boxes=output["boxes"],
            labels=output["labels"],
            scores=output.get("scores", None),
            classes=classes,
            show_conf=show_conf,
            font_scale=font_scale,
            box_thick=box_thick,
            fill_alpha=fill_alpha,
            text_box_color=text_box_color,
            text_font_color=text_font_color,
            text_alpha=text_alpha,
        )
        success = cv2.imwrite(output_path, image)
        print(f"  保存带 {num_boxes} 个检测框的图片到: {output_path}, 成功: {success}")
        if not success:
            print(f"  错误: 保存图片失败，检查目录权限或磁盘空间")
    except Exception as e:
        print(f"可视化 {image_name} 出错: {e}")


if __name__ == "__main__":
    inference()
