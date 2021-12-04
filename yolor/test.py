
# This is borrowed from the wonderful work of Wong Kin-Yiu:
#   https://github.com/WongKinYiu/yolor


import argparse
from datetime import datetime
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from utils.datasets import create_dataloader
from utils.general import \
    coco80_to_coco91_class, check_dataset, check_file, check_img_size, box_iou, \
    non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, clip_coords, set_logging, increment_path
from utils.loss import compute_loss
from utils.metrics import ap_per_class
from utils.plots import plot_images, output_to_target
from utils.torch_utils import select_device, time_synchronized

from models.models import *

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

# Generate summary stats performance table
# @todo: a nicer way to achieve this?
def summary_stats(raw_stats):
    # these are based on what COCOeval() spits out
    ious = [
        "0.50:0.95","0.50","0.75","0.50:0.95","0.50:0.95","0.50:0.95",
        "0.50:0.95","0.50:0.95","0.50:0.95","0.50:0.95","0.50:0.95","0.50:0.95",
    ]
    areas = [
        "all","all","all","small","medium","large",
        "all","all","all","small","medium","large"
    ]

    max_dets = [
        100,100,100,100,100,100,
        1,10,100,100,100,100,
    ]

    metrics = [
        "ap","ap","ap","ap","ap","ap",
        "ar","ar","ar","ar","ar","ar",
    ]

    # columns: ["metric","IoU","area","maxDets","result"]
    return [[metric,iou,area,max_det,result] for (iou,area,max_det,metric,result) in zip(ious,areas,max_dets,metrics,raw_stats)]


def test(data,
         weights=None,
         batch_size=16,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_conf=False,
         plots=True,
         log_imgs=0,
         is_coco = False):  # number of logged images

    t_very_start = datetime.now()

    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)
        save_txt = opt.save_txt  # save *.txt labels

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        run_name = save_dir.parts[-1]

        # Load model
        model = dnmodel = Darknet(opt.cfg).to(device)
        dnmodel
        # load model
        try:
            ckpt = torch.load(weights[0], map_location=device)  # load checkpoint
            ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(ckpt['model'], strict=False)
        except:
            load_darknet_weights(model, weights[0])
        imgsz = check_img_size(imgsz, s=64)  # check img_size

    # Half
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()

    data_file = data
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict

    check_dataset(data)  # check

    
    nc = 1 if single_cls else int(data['nc'])  # number of classes

    # @todo: look into iouv, niou
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs = min(log_imgs, 100) # ceil

    # Setup run config for wandb
    run_config = {
        "data_file": opt.data,
        "names_file": opt.names,
        "is_coco": is_coco,
        "cfg": opt.cfg,
        "weights": weights,
        "nc": nc,
        "device": device,
        "single_cls": single_cls,
        "conf_thres": conf_thres,
        "iou_thres": iou_thres,
        "task": opt.task,
        "batch_size": batch_size,
        "image_size": imgsz,
        "training": training,
        "augment": opt.augment,
        "shm_size": os.getenv("SHM_SIZE"),
        "z": {
            "verbose": opt.verbose,
            "log_imgs": log_imgs,
            "save_txt": opt.save_txt,
            "save_conf": opt.save_conf,
            "save_json": opt.save_json,
            "project": opt.project,
            "name": opt.name,
            # @todo: find a more useful way to log model data
            # "model": model,
        },
    }


    # Get jetson_clocks
    jcfile = "/resources/yolor_edge_jetson_clocks.out"
    if os.path.isfile(jcfile):
        f = open(jcfile, 'r')
        jetson_clocks = f.read()
        f.close()
        run_config["z"]["jetson_clocks"] = jetson_clocks

    wandb = None 
    try:
        import wandb  # Weights & Biases

        # Set up wandb to log useful data
        tags = [opt.name, "yolor-edge", opt.task, Path(data_file).name]
        if not os.getenv("WANDB_TAGS") == None:
            tags = tags + str(os.getenv("WANDB_TAGS")).split(',')
        wandb.init(
            name = run_name,
            tags = tags,
            config = run_config
            )
    except ImportError:
        log_imgs = 0

    # Dataloader
    if not training:
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        path = data['test'] if opt.task == 'test' else data['val']  # path to val/test images
        dataloader = create_dataloader(path, imgsz, batch_size, 64, opt, pad=0.5, rect=True)[0]

    seen = 0
    # Work out names
    try:
        names = model.names if hasattr(model, 'names') else model.module.names
    except:
        names = load_classes(opt.names)
    # Needs to be a dict
    names_dict = dict(enumerate(names))

    if wandb:
        wandb.config.update({"z.class_count": len(names)})
        # Logging this seems pointless now but it can be turned back on
        # wandb.config.update({"z.class_names": names})
        pass

    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    mf1 = 0.
    loss = torch.zeros(3, device=device)
    run_loss = [loss]
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []

    if wandb:
        # @todo: turn this on and see how it goes
        wandb.watch(model, log_freq=100)
        pass


    t_sec_start = datetime.now()

    # Run inference
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        # Disable gradients
        with torch.no_grad():
            # Run model
            t = time_synchronized()
            inf_out, train_out = model(img, augment=augment)  # inference and training outputs
            t0 += time_synchronized() - t

            # Compute loss
            if training:  # if model has loss hyperparameters
                loss += compute_loss([x.float() for x in train_out], targets, model)[1][:3]  # box, obj, cls
                run_loss.append(loss)

            # Run NMS
            t = time_synchronized()
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres)
            t1 += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to text file
            path = Path(paths[si])
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                x = pred.clone()
                x[:, :4] = scale_coords(img[si].shape[1:], x[:, :4], shapes[si][0], shapes[si][1])  # to original
                for *xyxy, conf, cls in x:
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging                
            if len(wandb_images) < log_imgs:
                # pred.tolist() becomes:
                #   [239.5, 106.875, 257.75, 124.875, 0.490966796875, 14.0]
                box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                             "class_id": int(cls),
                             "box_caption": "%s %.3f" % (names_dict[int(cls)], conf),
                             "scores": {"class_score": conf},
                             "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                boxes = {"predictions": {"box_data": box_data, "class_labels": names_dict}}
                wandb_images.append(wandb.Image(img[si], boxes=boxes, caption=path.name))

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = pred[:, :4].clone()  # xyxy
                scale_coords(img[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices
                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f't  est_batch{batch_i}_labels.jpg'  # filename
            plot_images(img, targets, paths, f, names)  # labels
            f = save_dir / f'test_batch{batch_i}_pred.jpg'
            plot_images(img, output_to_target(output, width, height), paths, f, names)  # predictions

    t_sec_end = datetime.now()
    
    if wandb:
        inf_time = (t_sec_end - t_sec_start).total_seconds()
        wandb.log({"time.inference": inf_time})
    
    # Compute statistics
    t_sec_start = datetime.now()
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        # ap_per_class(): return p, r, ap, f1, unique_classes.astype('int32'), px, py
        p, r, ap, f1, ap_class, px, py = ap_per_class(*stats, plot=plots, fname=save_dir / 'precision-recall_curve.png')
        
        # @todo: Plot P and R
        if wandb:
            # table_data = [[x,y] for (x,y) in zip(px,py)]
            # table = wandb.Table(columns=["precision","recall"], data=table_data)
            # wandb.log({"precision_recall": wandb.plot.scatter(table, "Precision", "Recall", title="Precision vs Recall")})
            wandb.log({"stats1": {
                "p": p,
                "r": r,
                "ap": ap,
                "f1": f1,
                "ap_class": ap_class
            }})

        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        mf1 = f1.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class

        if wandb:
            wandb.log({"stats2": {
                "p": p,
                "r": r,
                "ap": ap,
                "ap50": ap50,
                "nt": nt
            }})

    else:
        nt = torch.zeros(1)

    t_sec_end = datetime.now()

    if wandb:
        stat_labels = ['mp','map50','mr', 'mf1', 'nt',   'seen', 'p', 'r', 'run_loss']
        stat_values = [mp,   map50,  mr,   mf1, nt.sum(), seen,  p,   r,   run_loss]
        wandb.log({"stats": dict(zip(stat_labels,stat_values))})
        wandb.log({"time.stats": (t_sec_end-t_sec_start).total_seconds()})

    # Print results
    pf = '%20s' + '%12.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if wandb:
        speeds = [round(x,5) for x in t[0:3]]
        wandb.log({"speed": {"inference": speeds[0], "nms": speeds[1], "total": speeds[2]} })
        # These are included in config but logging again incase changed
        wandb.log({"image_size": imgsz, "batch_size": batch_size})

    # Print speeds
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # W&B logging
    if plots and wandb:
        wandb.log({"images": wandb_images})
        wandb.log({"validation": [wandb.Image(str(x), caption=x.name) for x in sorted(save_dir.glob('test*.jpg'))]})
        # @todo: don't log pvr in wandb after started logging via data
        wandb.log({"precision_vs_recall": wandb.Image(str(save_dir.joinpath('precision-recall_curve.png')), caption="Precision Recall Curve")})

    t_sec_start = datetime.now()
    # Save JSON
    if len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_path = '../coco/annotations'

        # annotations location added to coco.yaml, read from there
        try:
            anno_path = data['annotations']
        except:
            anno_path = '../coco/annotations'

        anno_glob = f"{anno_path}/instances_val*.json"
        anno_glob = glob.glob(anno_glob)
        anno_json = ''
        if len(anno_glob) > 0:
            anno_json = anno_glob[0]
            pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
            if save_json:
                print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
                with open(pred_json, 'w') as f:
                    json.dump(jdict, f)


        # Updated pyocotools==2.0.2 so the float conversion bug is fixed
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        coco_eval_start = datetime.now()
        anno = COCO(anno_json)  # init annotations api
        pred = anno.loadRes(pred_json)  # init predictions api
        eval = COCOeval(anno, pred, 'bbox')
        if is_coco:
            eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
        eval.evaluate()
        eval.accumulate()
        eval.summarize()

        map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)

        # @todo: try this out
        # print(eval.params)

        coco_eval_time = datetime.now() - coco_eval_start

        if wandb:
            table = wandb.Table(columns=["metric","IoU","area","maxDets","result"], data=summary_stats(eval.stats))
            wandb.log({"performance_table": table})
            coco_eval_time = coco_eval_time.total_seconds()
            wandb.log({"eval": { "map": map, "map50": map50, "stats": eval.stats }})
            
    t_sec_end = datetime.now()
    if wandb:
        wandb.log({"time.eval": (t_sec_end-t_sec_start).total_seconds()})

    # Return results
    if not training:
        print('Results saved to %s' % save_dir)
    model.float()  # for training
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]

    if wandb:
        wandb.log({"maps": maps})
        t_run_time = datetime.now() - t_very_start
        t_run_time = t_run_time.total_seconds()
        wandb.log({"time.total" : t_run_time})

    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolor_p6.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='/yolor-edge/data/coco-2017/coco.yaml', help='*.data path')
    parser.add_argument('--names', type=str, default='/yolor-edge/data/coco-2017/coco.names', help='*.cfg path')
    parser.add_argument('--is-coco', action='store_true', help='Indicate using COCO data')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=128, help='inference size (pixels)')
    parser.add_argument('--log-images', type=int, default=32, help='log images')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help="'val', 'test', 'study'")
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='/resources/runs/yolor/test', help='save to project/name')
    parser.add_argument('--name', default='yolor_p6_test', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--cfg', type=str, default='/yolor-edge/yolor/cfg/yolor_p6.cfg', help='*.cfg path')

    opt = parser.parse_args()
    is_coco = opt.is_coco
    if not is_coco and opt.data.endswith('coco.yaml'):
        is_coco = True

    opt.save_json |= is_coco
    opt.data = check_file(opt.data)  # check file

    if opt.task in ['val', 'test']:  # run normally
        test(
            data = opt.data,
            weights = opt.weights,
            batch_size = opt.batch_size,
            imgsz = opt.img_size,
            conf_thres = opt.conf_thres,
            iou_thres = opt.iou_thres,
            save_json = opt.save_json,
            single_cls = opt.single_cls,
            augment = opt.augment,
            verbose = opt.verbose,
            save_txt = opt.save_txt,
            save_conf = opt.save_conf,
            log_imgs = opt.log_images,
            is_coco = is_coco
        )

    elif opt.task == 'study':  # run over a range of settings and save/plot
        for weights in ['yolor_p6.pt', 'yolor_w6.pt']:
            f = 'study_%s_%s.txt' % (Path(opt.data).stem, Path(weights).stem)  # filename to save to
            x = list(range(320, 800, 64))  # x axis
            y = []  # y axis
            for i in x:  # img-size
                print('\nRunning %s point %s...' % (f, i))
                r, _, t = test(opt.data, weights, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        # utils.general.plot_study_txt(f, x)  # plot
