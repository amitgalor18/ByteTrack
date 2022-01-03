import matplotlib.pyplot as plt
from loguru import logger

import cv2
import numpy as np
import torch

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer


import argparse
import os
import time

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

GT_PREDICATIONS = False
GT_DETECTIONS = False
HEAD_DETECTIONS = False


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "--demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        #"--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
        "--path", default="./videos/palace.mp4", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video"
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    # parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--min-box-area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results_with_sigma(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},1,{sx},{sy},-1\n'

    filename = filename.replace(".txt", "_sig.txt")

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, sig_x, sig_y in results:
            for tlwh, track_id, sx, sy in zip(tlwhs, track_ids, sig_x, sig_y):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, sx=sx, sy=sy)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_net_detection_results(filename, results):

    filename = filename.replace(".txt", "_net_det.txt")
    save_format = '{frame},{x1},{y1},{w},{h},{obj_conf},{cls_conf},{cls}\n'

    with open(filename, 'w') as f:
        for dets in results:
            frame_id = dets[0]
            for det in dets[1]:

                x1, y1, x2, y2 = det[:4]
                w = x2 - x1
                h = y2 - y1

                obj_conf = det[4]
                if np.size(det) > 5:
                    cls_conf = det[5]
                    cls = det[6]
                else:
                    cls_conf = -1
                    cls = -1

                line = save_format.format(frame=frame_id, x1=x1, y1=y1, w=w, h=h, obj_conf=obj_conf, cls_conf=cls_conf, cls=cls)
                f.write(line)
    logger.info('save results to {}'.format(filename))

def write_detection_results(filename, results):

    filename = filename.replace(".txt", "_det.txt")
    save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            for tlwh, track_id in zip(tlwhs, track_ids):

                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=-1, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def generate_trajectories(file_path, groundTrues):
    f = open(file_path, 'r')

    lines = f.read().split('\n')
    values = []
    for l in lines:
        split = l.split(',')
        if len(split) < 2:
            break
        numbers = [float(i) for i in split]
        values.append(numbers)

    values = np.array(values, np.float_)

    if groundTrues:
        pass
        values = values[values[:, 6] == 1, :]  # Remove ignore objects
        values = values[values[:, 7] == 1, :]  # Pedestrian only

    ids = np.unique(values[:, 1])
    trajectories = []
    for id in ids:
        trajectory = values[id == values[:, 1], :].astype(float)
        trajectory[:, 2] += trajectory[:, 4] // 2
        trajectory[:, 3] += trajectory[:, 5] // 2
        trajectory[:, 4] = trajectory[:, 4] / trajectory[:, 5]

        trajectories.append(trajectory)

    return trajectories


def generate_head_detections(file_path):
    f = open(file_path, 'r')

    lines = f.read().split('\n')
    values = []
    for l in lines:
        split = l.split(',')
        if len(split) < 2:
            break
        numbers = [float(i) for i in split]
        values.append(numbers)

    values = np.array(values, np.float_)
    ids = np.unique(values[:, 0])
    trajectories = []
    for id in ids:
        trajectory = values[id == values[:, 0], :].astype(float)
        trajectories.append(trajectory[:, 2:7])

    return trajectories

def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked
    return boxes[pick], probs[pick]

def boxes_validation(main_boxes, main_probs, support_boxes, support_probs, overlapThresh=0.3):
    # if there are no boxes, return an empty list
    if len(main_boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if main_boxes.dtype.kind == "i":
        main_boxes = main_boxes.astype("float")

    if support_boxes.dtype.kind == "i":
        support_boxes = support_boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    mx1 = main_boxes[:, 0]
    my1 = main_boxes[:, 1]
    mx2 = main_boxes[:, 2]
    my2 = main_boxes[:, 3]

    sx1 = support_boxes[:, 0]
    sy1 = support_boxes[:, 1]
    sx2 = support_boxes[:, 2]
    sy2 = support_boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    main_area = (mx2 - mx1 + 1) * (my2 - my1 + 1)
    support_area = (sx2 - sx1 + 1) * (sy2 - sy1 + 1)

    # keep looping while some indexes still remain in the indexes list
    for i in range(np.size(main_boxes, 0)):
        for j in range(np.size(support_boxes, 0)):

            # find the largest (x, y) coordinates for the start of the bounding
            # box and the smallest (x, y) coordinates for the end of the bounding

            xx1 = np.maximum(mx1[i], sx1[j])
            yy1 = np.maximum(my1[i], sy1[j])
            xx2 = np.minimum(mx2[i], sx2[j])
            yy2 = np.minimum(my2[i], sy2[j])

            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            intersecation = (w * h)
            if intersecation == 0:
                continue

            iou = intersecation / (main_area[i] + support_area[j] - intersecation)

            if iou >= 0.6:
                main_probs[i] += support_probs[j]
                main_probs[i] = np.clip(float(main_probs[i]), 0, 1)

    # return only the bounding boxes that were picked
    return main_boxes, main_probs

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        # return img, img_info  # TODO: DEBUG ONLY

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            #logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info


def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    cov_results = []
    detection_results = []
    outputs_detections = []

    # gt_path = path[:-4] + 'gt/gt.txt'
    # gt_trajectories = generate_trajectories(gt_path, groundTrues=True)

    if HEAD_DETECTIONS:
        head_path = path[:-4] + 'DLA34 Head detections.txt'
        head_detections_all = generate_head_detections(head_path)

    for image_name in files:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # Get GT - DEVELOPMENT ONLY
        # current_gt_trajectories = []
        # current_gt_visibility = []
        # for trajectory in gt_trajectories:
            # frames = trajectory[:, 0] - 1
            # if frame_id in frames:
                # key = int(np.where(frame_id == frames)[0])
                # current_gt_trajectories.append(trajectory[key, 2:6])
                # current_gt_visibility.append(trajectory[key, 8])

        # current_gt_trajectories = np.array(current_gt_trajectories)
        # current_gt_visibility = np.expand_dims(current_gt_visibility, 1)

        # plt.figure()
        # plt.hist(current_gt_visibility, bins=50)
        # plt.show()

        outputs, img_info = predictor.inference(image_name, timer)
        online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)

        scale = min(exp.test_size[0] / float(img_info['height'],), exp.test_size[1] / float(img_info['width']))


        # if GT_PREDICATIONS:
            # online_targets, raw_detections = tracker.update(outputs[0], [img_info['height'], img_info['width']],
                                                            # exp.test_size, current_gt_trajectories)
        # elif GT_DETECTIONS:
            # current_gt_trajectories[:, 2] *= current_gt_trajectories[:, 3]
            # current_gt_trajectories[:, 0] -= current_gt_trajectories[:, 2] / 2
            # current_gt_trajectories[:, 1] -= current_gt_trajectories[:, 3] / 2
            # current_gt_trajectories[:, 2] += current_gt_trajectories[:, 0]
            # current_gt_trajectories[:, 3] += current_gt_trajectories[:, 1]

            # outputs = [np.hstack((current_gt_trajectories, current_gt_visibility))]
            # outputs = [np.hstack((current_gt_trajectories, current_gt_visibility * 0.0 + 0.99))]
            # outputs = torch.tensor(outputs)
            # img_info['height'] = exp.test_size[0]
            # img_info['width'] = exp.test_size[1]
            # online_targets, raw_detections = tracker.update(outputs[0], [img_info['height'], img_info['width']],
                                                        # exp.test_size)
        # elif HEAD_DETECTIONS:
            # body_detections = outputs[0].cpu().numpy()
            # body_detections[:, 4] *= body_detections[:, 5]
            # body_detections = body_detections[:, :5]

            # head_detections = head_detections_all[frame_id]
            # head_detections = head_detections[head_detections[:, 4] > 0.05, :]

            # x1b = head_detections[:, 0] - 0.75 * head_detections[:, 2]
            # y1b = head_detections[:, 1] - 0.0 * head_detections[:, 3]
            # x2b = head_detections[:, 0] + head_detections[:, 2] + 0.75 * head_detections[:, 2]
            # y2b = head_detections[:, 1] + head_detections[:, 3] + 5.0 * head_detections[:, 3]

            # head_detections[:, 0] = x1b * scale
            # head_detections[:, 1] = y1b * scale
            # head_detections[:, 2] = x2b * scale
            # head_detections[:, 3] = y2b * scale

            # Early union fusion
            # early_fuse_boxes, early_fuse_scores = boxes_validation(body_detections[:, :4],
            #                                                        body_detections[:, 4],
            #                                                        head_detections[:, :4],
            #                                                        head_detections[:, 4],
            #                                                         overlapThresh=predictor.nmsthre)
            #
            # body_detections[:, :4] = early_fuse_boxes
            # body_detections[:, 4] = early_fuse_scores

            # head_detections_high = head_detections[head_detections[:, 4] > 0.5, :]
            # early_fuse_detections = np.vstack((body_detections, head_detections_high))

            # early_fuse_boxes, early_fuse_scores = non_max_suppression(early_fuse_detections[:, :4],
                                                                    #   probs=early_fuse_detections[:, 4],
                                                                    #   overlapThresh=predictor.nmsthre)

            # early_fuse_scores = np.expand_dims(early_fuse_scores, 1)
            # outputs = [np.hstack((early_fuse_boxes, early_fuse_scores))]
            # outputs = [head_detections]
            # outputs = [body_detections]


            # outputs = torch.tensor(outputs)
            # online_targets, raw_detections = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)

        # else:
            # online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']],
                                                        # exp.test_size)


        online_tlwhs = []
        online_ids = []
        online_scores = []
        online_sig_x = []
        online_sig_y = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
                online_sig_x.append(np.sqrt(t.covariance[0, 0]))
                online_sig_y.append(np.sqrt(t.covariance[1, 1]))
        timer.toc()

        detection_tlwhs = []
        detection_ids = []
        # for d in raw_detections:
            # tlwh = d.tlwh
            # vertical = tlwh[2] / tlwh[3] > 1.6
            # if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                # detection_tlwhs.append(tlwh)
                # detection_ids.append(-1)

        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
        # cov_results.append((frame_id + 1, online_tlwhs, online_ids, online_sig_x, online_sig_y))
        # detection_results.append((frame_id + 1, detection_tlwhs, detection_ids))

        # output_numpy = outputs[0].cpu().numpy()
        # output_numpy[:, :4] = output_numpy[:, :4]  / scale

        # outputs_detections.append((frame_id, output_numpy))

        online_im = plot_tracking(img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1,
                                          fps=1. / timer.average_time)

        #result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            cv2.imwrite(save_file_name, online_im)

        if frame_id % 2000 == 0 and frame_id != 0:
            print("Writing results")
            result_filename = os.path.join(save_folder, 'result.txt')
            write_results(result_filename, results)
            # write_results_with_sigma(result_filename, cov_results)
            # write_detection_results(result_filename, detection_results)
            # write_net_detection_results(result_filename, outputs_detections)


        ch = cv2.waitKey(0)
        frame_id += 1
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

    # write_results(result_filename, results)
    result_filename = os.path.join(save_folder, 'result.txt')
    write_results(result_filename, results)
    # write_results_with_sigma(result_filename, cov_results)
    # write_detection_results(result_filename, detection_results)
    # write_net_detection_results(result_filename, outputs_detections)


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = os.path.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = os.path.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame, timer)
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
            timer.toc()
            results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
            online_im = plot_tracking(img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1,
                                      fps=1. / timer.average_time)
            if args.save_result:
                vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1
    result_filename = os.path.join(save_folder, 'result.txt')
    write_results(result_filename, results)


def main(exp, args):
    # torch.cuda.set_device('cuda')
    torch.cuda.set_device(0)
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    if args.save_result:
        vis_folder = os.path.join(file_name, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    # print(model)

    if args.device == "gpu":
        model.cuda()
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)
    
    if args.fp16:
            model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, args.path, current_time, args.save_result)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()

    # args.demo = 'image'
    # args.path = r'/workspace/ByteTrack/datasets/MOT20/train/MOT20-05/img1'
    # args.exp_file = r'/workspace/Thesis/ByteTrack/exps/example/mot/yolox_x_mix_mot20_ch.py'
    # args.ckpt = r'/workspace/Thesis/ByteTrack/pretrained/bytetrack_x_mot20.tar'

    # args.mot20 = True
    # args.save_result = True
    # args.fp16 = True
    # args.fuse = True
    # args.match_thresh = 0.7
    # args.batch_size = 1
    # args.devices = 1
    # args.track_buffer = 30
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
