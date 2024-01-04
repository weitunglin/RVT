import math
from enum import Enum, auto
from typing import Any

import cv2
import torch
from einops import rearrange
from omegaconf import DictConfig
import numpy as np
from boxmot.trackers.bytetrack.byte_tracker import BYTETracker

from data.utils.types import ObjDetOutput
from loggers.wandb_logger import WandbLogger
from utils.evaluation.prophesee.visualize.vis_utils import LABELMAP_GEN1, LABELMAP_GEN4_SHORT, draw_bboxes
from .viz_base import VizCallbackBase


class DetectionVizEnum(Enum):
    EV_IMG = auto()
    LABEL_IMG_PROPH = auto()
    PRED_IMG_PROPH = auto()


class DetectionVizCallback(VizCallbackBase):
    def __init__(self, config: DictConfig):
        super().__init__(config=config, buffer_entries=DetectionVizEnum)

        dataset_name = config.dataset.name
        if dataset_name == 'gen1':
            self.label_map = LABELMAP_GEN1
        elif dataset_name == 'gen4':
            self.label_map = LABELMAP_GEN4_SHORT
        else:
            raise NotImplementedError

        self.batch_count = 0

        self.tracker = BYTETracker(
            track_thresh=0.7,
            match_thresh=0.8,
            track_buffer=30,
            frame_rate=30
        )

    def on_train_batch_end_custom(self,
                                  logger: WandbLogger,
                                  outputs: Any,
                                  batch: Any,
                                  log_n_samples: int,
                                  global_step: int) -> None:
        if outputs is None:
            # If we tried to skip the training step (not supported in DDP in PL, atm)
            return
        ev_tensors = outputs[ObjDetOutput.EV_REPR]
        num_samples = len(ev_tensors)
        assert num_samples > 0
        log_n_samples = min(num_samples, log_n_samples)

        merged_img = []
        captions = []
        start_idx = num_samples - 1
        end_idx = start_idx - log_n_samples
        # for sample_idx in range(log_n_samples):
        for sample_idx in range(start_idx, end_idx, -1):
            ev_img = self.ev_repr_to_img(ev_tensors[sample_idx].cpu().numpy())

            predictions_proph = outputs[ObjDetOutput.PRED_PROPH][sample_idx]
            prediction_img = ev_img.copy()
            draw_bboxes(prediction_img, predictions_proph, labelmap=self.label_map)

            labels_proph = outputs[ObjDetOutput.LABELS_PROPH][sample_idx]
            label_img = ev_img.copy()
            draw_bboxes(label_img, labels_proph, labelmap=self.label_map)

            merged_img.append(rearrange([prediction_img, label_img], 'pl H W C -> (pl H) W C', pl=2, C=3))
            captions.append(f'sample_{sample_idx}')

        logger.log_images(key='train/predictions',
                          images=merged_img,
                          caption=captions,
                          step=global_step)

    def on_validation_batch_end_custom(self, batch: Any, logger: Any, outputs: Any):
        if outputs[ObjDetOutput.SKIP_VIZ]:
            return
        ev_tensor = outputs[ObjDetOutput.EV_REPR]
        assert isinstance(ev_tensor, torch.Tensor)

        ev_img = self.ev_repr_to_img(ev_tensor.cpu().numpy())

        predictions_proph = outputs[ObjDetOutput.PRED_PROPH]
        prediction_img = ev_img.copy()
        draw_bboxes(prediction_img, predictions_proph, labelmap=self.label_map)
        self.add_to_buffer(DetectionVizEnum.PRED_IMG_PROPH, prediction_img)
        if 'select_sequence' in outputs:
            boxes = predictions_proph
            boxes_array = np.array([])
            for i in range(boxes.shape[0]):
                pt1 = (int(boxes['x'][i]), int(boxes['y'][i]))
                size = (int(boxes['w'][i]), int(boxes['h'][i]))
                pt2 = (pt1[0] + size[0], pt1[1] + size[1])
                score = boxes['class_confidence'][i]
                class_id = boxes['class_id'][i]
                boxes_array = np.append(boxes_array, [pt1[0], pt1[1], pt2[0], pt2[1], score, class_id])
            boxes_array = boxes_array.reshape(-1, 6)
            tracks = self.tracker.update(boxes_array, prediction_img)

            track_img = prediction_img.copy()
            # print bboxes with their associated id, cls and conf
            if tracks.shape[0] != 0:
                xyxys = tracks[:, 0:4].astype('int') # float64 to int
                ids = tracks[:, 4].astype('int') # float64 to int
                confs = tracks[:, 5]
                clss = tracks[:, 6].astype('int') # float64 to int
                inds = tracks[:, 7].astype('int') # float64 to int
                color = (0, 255, 255)  # BGR
                thickness = 2
                fontscale = 0.5
                for xyxy, id, conf, cls in zip(xyxys, ids, confs, clss):
                    cv2.putText(
                        track_img,
                        f'{id}',
                        (int(xyxy[0] + (int(xyxy[2] - xyxy[0]) / 2) - 5), int(xyxy[1] + (int(xyxy[3] - xyxy[1]) / 2) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontscale,
                        color,
                        thickness
                    )   
                    
            cv2.imwrite(f'/workspace/RVT/output_images/sample_prediction_tracking_{outputs["select_sequence"]}/{math.floor(self.batch_count/1):03}.jpg', track_img)
            cv2.imwrite(f'/workspace/RVT/output_images/sample_prediction_{outputs["select_sequence"]}/{math.floor(self.batch_count/1):03}.jpg', prediction_img)

        labels_proph = outputs[ObjDetOutput.LABELS_PROPH]
        label_img = ev_img.copy()
        draw_bboxes(label_img, labels_proph, labelmap=self.label_map)
        self.add_to_buffer(DetectionVizEnum.LABEL_IMG_PROPH, label_img)
        if 'select_sequence' in outputs:
            cv2.imwrite(f'/workspace/RVT/output_images/sample_label_{outputs["select_sequence"]}/{math.floor(self.batch_count/1):03}.jpg', label_img)

        self.batch_count = self.batch_count + 1

    def on_validation_epoch_end_custom(self, logger: WandbLogger, pl_module: Any):
        pred_imgs = self.get_from_buffer(DetectionVizEnum.PRED_IMG_PROPH)
        label_imgs = self.get_from_buffer(DetectionVizEnum.LABEL_IMG_PROPH)
        assert len(pred_imgs) == len(label_imgs)
        merged_img = []
        captions = []
        for idx, (pred_img, label_img) in enumerate(zip(pred_imgs, label_imgs)):
            merged_img.append(rearrange([pred_img, label_img], 'pl H W C -> (pl H) W C', pl=2, C=3))
            captions.append(f'sample_{idx}')
            if pl_module and isinstance(pl_module.full_config.custom.select_sequence, int):
                cv2.imwrite(f'/workspace/RVT/output_images/sample_{pl_module.full_config.custom.select_sequence}/{math.floor(idx/1):03}.jpg', merged_img[-1])

        # logger.log_images(key='val/predictions',
        #                   images=merged_img,
        #                   caption=captions)
