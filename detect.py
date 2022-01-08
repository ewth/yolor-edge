import time
from pathlib import Path

import cv2
from numpy.lib.function_base import average
import torch
import torch.backends.cudnn as cudnn
from utils.fontscaling import write_heading

from yolor.utils.datasets import LoadStreams, LoadImages
from yolor.utils.general import (non_max_suppression, apply_classifier, scale_coords, xyxy2xywh)
from yolor.utils.plots import add_text, add_text_heading, calc_text_size, plot_one_box
from yolor.utils.torch_utils import select_device, load_classifier, time_synchronized

from yolor.models.models import *
from yolor.utils.datasets import *
from yolor.utils.general import *
import logging
from threading import Thread
from threading import Lock

class Detect:
    """
    Runs inference. Detects objects. The meaty parts.
    """

    # @todo: A whole lotta refactoring

    source_path: str
    output_path: str
    target_device: str
    inference_size: int
    run_name: str
    run_id: str

    webcam_source: bool

    model_weights: str
    model_name: str
    model_config: str

    class_names_file: str
    class_names: list
    classes_restrict: list

    confidence_threshold: float
    iou_threshold: float

    nms_is_agnostic: bool

    save_text: bool
    save_images: bool
    display_images: bool
    save_video_frames: bool
    capture_nth_frame: int
    save_nth_frame: int
    append_run_id_to_files: bool

    display_bounding_boxes: bool
    display_bounding_box_labels: bool
    display_bounding_box_confidence: bool
    
    display_stats: bool
    display_extra_stats: bool

    display_percent_decimal = True
    mode_verbose = False
    mode_augment = False
    mode_headless = False

    stats_top_prepend = []
    stats_top_append = []
    stats_bottom_prepend = []
    stats_bottom_append = []

    system_name = "edge.nv_jx_nx"

    #
    # Juicy bits
    #
    _model = None
    _half_mode = False

    # So modest
    _have_setup = False
    _device = None
    _have_init_device = False
    _model = None
    _model_name_loaded = None
    _have_loaded_model = False
    _class_bounding_box_colours = [[0, 255, 0], [255, 204, 0], [255, 173, 102], [148, 101, 59], [184, 75, 59], [219, 88, 114], [255, 61, 165], [148, 0, 79], [184, 29, 163], [193, 88, 219], [141, 41, 255], [69, 0, 148], [122, 102, 255], [71, 59, 148], [184, 159, 59], [184, 109, 44], [255, 87, 61], [148, 20, 0], [184, 0, 37], [255, 102, 184], [255, 20, 224], [148, 47, 134], [153, 29, 184], [173, 102, 255], [52, 20, 255], [63, 44, 184], [255, 130, 20], [148, 69, 0], [255, 122, 102], [255, 0, 51], [184, 44, 72], [184, 73, 132], [255, 102, 235], [208, 20, 255], [130, 59, 148], [109, 44, 184], [87, 61, 255], [20, 0, 148], [41, 52, 255], [0, 9, 184], [59, 64, 148], [53, 86, 219], [41, 116, 255], [29, 83, 184], [41, 148, 255], [29, 106, 184], [35, 92, 148], [0, 119, 184], [59, 117, 148], [0, 147, 184], [0, 242, 255], [41, 255, 234], [12, 148, 134], [102, 110, 255], [59, 65, 184], [41, 84, 255], [73, 95, 184], [82, 142, 255], [73, 112, 184], [82, 168, 255], [73, 129, 184], [20, 173, 255], [59, 140, 184], [20, 208, 255], [73, 162, 184], [102, 247, 255], [88, 219, 206], [53, 61, 219], [12, 19, 148], [102, 133, 255], [35, 58, 148], [0, 77, 219], [35, 75, 148], [0, 110, 219], [0, 74, 148], [82, 194, 255], [12, 100, 148], [102, 224, 255], [12, 121, 148], [59, 177, 184], [0, 184, 165]]
    _logger = None

    lock = None
    # Not reeeaalllly sure if "helping" with garbage collection is "Pythonic" so experimenting
    _help_garbage_man = False


    def __init__(
        # Lot of paramaters here...
        self,
        output_path: str,
        source_path: str,
        target_device: str,
        run_name: str,
        run_id: str,
        model_weights = "/resources/weights/yolor/yolor_p6.pt",
        model_config = "/yolor-edge/yolor/cfgyolor_p6.cfg",
        model_name = "",

        # Should be one or the other; list takes precedence
        class_names_file = "data/coco.names",
        class_names = [],
        # Empty = all classes; specificy numeric index in list to restrict (e.g. 0 = person)
        classes_restrict = [],

        inference_size = 1280,
        confidence_threshold = 0.4,
        iou_threshold = 0.5,
        nms_is_agnostic = False,

        save_text = False,
        save_images = False,
        save_video_frames = False,
        capture_nth_frame = 4,
        save_nth_frame = 20,
        append_run_id_to_files = False,

        display_images = False,
        display_bounding_boxes = False,
        display_bounding_box_labels = True,
        display_bounding_box_confidence = True,
        display_percent_decimal = True,
        display_stats = False,
        display_extra_stats = False,

        stats_top_append = [],
        stats_bottom_append = [],
        stats_top_prepend = [],
        stats_bottom_prepend = [],

        mode_headless = False,
        mode_augment = False,
        mode_verbose = False,

        system_name = "",

    ):
        print("[yolor.detect] Running Init")
        # Seens like a lot of double handling. Better way?

        # On 2nd thought:
        #   the one-liners look nice and all but it's hard to tell what has/hasn't been loaded.
        #   Or if things are lining up correctly with unpacking, etc.

        #
        # Paths; Device; Run name; stringy stuff
        #
        self.source_path = source_path
        self.output_path = output_path
        self.target_device = target_device
        self.run_name = run_name
        self.run_id = run_id

        # Model details
        self.model_weights = model_weights
        self.model_config = model_config
        self.model_name = model_name

        # Object class details
        self.class_names_file = class_names_file
        self.class_names = class_names
        self.classes_restrict = classes_restrict

        # Inference configuration
        self.inference_size = inference_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.nms_is_agnostic = nms_is_agnostic

        # Saving/output details
        self.save_text = save_text
        self.save_images = save_images
        self.save_video_frames = save_video_frames
        self.display_images = display_images
        self.capture_nth_frame = capture_nth_frame
        self.save_nth_frame = save_nth_frame
        self.append_run_id_to_files = append_run_id_to_files

        # Rendering image stuff
        self.display_stats = display_stats
        self.display_extra_stats = display_extra_stats
        self.display_bounding_boxes = display_bounding_boxes
        self.display_bounding_box_labels = display_bounding_box_labels
        self.display_bounding_box_confidence = display_bounding_box_confidence
        self.display_percent_decimal = display_percent_decimal

        # Stuff to tack on to stats
        self.stats_top_prepend = stats_top_prepend
        self.stats_top_append = stats_top_append
        self.stats_bottom_prepend = stats_bottom_prepend
        self.stats_bottom_append = stats_bottom_append

        # Mode flags
        self.mode_verbose = mode_verbose
        self.mode_augment = mode_augment
        self.mode_headless = mode_headless

        logging.basicConfig(level=logging.DEBUG)

        self.lock = Lock()

        if system_name:
            self.system_name = system_name

        # end __init__


    def setup(self):
        """
        Setup for a detection run
        """
        print("[yolor.detect] Running setup")
        self.lock.acquire()
        # This is probably unnecessary, but resetting flag at start of all setup methods in case it fails or something
        self._have_setup = False
        if not Path(self.output_path).exists:
            raise FileNotFoundError(self.output_path)

        source = self.source_path
        self.webcam_source = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

        # Load classes
        if self.class_names is None or len(self.class_names) < 1:
            with open(self.class_names_file, "r") as f:
                names = f.read().replace("\r\n","\n").split("\n")
            # Filter empties
            self.class_names = list(filter(None, names))
        self._have_setup = True
        self.lock.release()

    def display(self, text: str, ignore_verbose = False):
        """
        Display text only if in verbose mode
        """
        # @todo: attach to a logging instance or something better than this
        if not self.mode_verbose and not ignore_verbose:
            return
        print("[yolor.detect] " + text)


    def get_bounding_box_colours(self) -> list:
        """
        Return list of colours (key = class) for bounding boxes/labels
        """

        return self._class_bounding_box_colours.copy()

    def load_model(self):
        """
        Load a model (weights/config pair) into memory for use
        """
        print("[yolor.detect] Running load_model")

        self.unload_model()
        if not self._have_init_device:
            print("Init device first")
            return

        if len(self.model_weights) < 1 or not self.model_config:
            print("No weights/config specified")
            return
        # Load model
        logging.debug(f"Attempting to load model. Config: {self.model_config}, Weights: {self.model_weights}")
        model = Darknet(self.model_config, self.inference_size).cuda()
        model.load_state_dict(torch.load(self.model_weights, map_location=self._device)['model'])
        #model = attempt_load(weights, map_location=device)  # load FP32 model
        # @todo: might be worth enabling the check
        #inference_size = check_img_size(inference_size, s=model.stride.max())  # check img_size
        model.to(self._device).eval()
        if self._half_mode:
            model.half()  # to FP16
        
        self._model = model

        self._model_name_loaded = Path(self.model_weights).name.replace('.pt', '')
        if not self.model_name:
            self.model_name = self._model_name_loaded
        
        disp_string = f"Using model {self.model_name}"
        logging.debug(disp_string)
        self._have_loaded_model = True

    def unload_model(self):
        """
        Unload the model
        """
        print("[yolor.detect] Running unload_model")
        if self._help_garbage_man and self._model is not None:
            del self._model
        self._have_loaded_model, self._model, self._model_name_loaded = False, None, None

    def deinit_device(self):
        """
        De-initialise device
        """
        print("[yolor.detect] Running deinit_device")
        self.unload_model()
        self._have_init_device = False
        if self._help_garbage_man and self._device is not None:
            del self._device

    def init_device(self, no_half = False):
        """
        Initialise the device for usage
        """
        print("[yolor.detect] Running init_device")
        self.deinit_device()
        if not self._have_setup:
            print("Run setup first")
            return
        if not self.target_device:
            print("Target device not set")
            return

        self._device = select_device(self.target_device)
        self._half_mode = False

        if not no_half:
            self._half_mode = self._device.type != 'cpu'  # half precision only supported on CUDA

        self._have_init_device = True
        print(f"Device initialised: {self._device}")

    def run(self):
        """
        Run inference according to setup. The main event.
        """
        
        inference_size = self.inference_size
        print(f"[yolor.detect] Running inference at {inference_size}px")
        if not self._have_setup:
            print("Run setup first.")
            return

        if not self._have_init_device:
            self.init_device()

        if not self._device:
            print("Device not initialised")
            return

        if not self._have_loaded_model:
            self.load_model()

        if not self._model:
            print("Model not loaded.")
            return

        model = self._model


        # @todo: shallow copy lists etc rather than by ref
        source_path, output_path, classes_restrict = \
            self.source_path, self.output_path, self.classes_restrict

        capture_nth_frame, conf_thres, iou_thres = \
            self.capture_nth_frame, self.confidence_threshold, self.iou_threshold

        class_names = self.class_names.copy()

        augment, agnostic_nms,  display_stats, save_frames, save_txt = \
            self.mode_augment, self.nms_is_agnostic, self.display_stats, self.save_video_frames, self.save_text

        display_extra_stats = self.display_extra_stats

        display_bounding_boxes, display_bounding_box_labels = self.display_bounding_boxes, self.display_bounding_box_labels
        
        save_img, view_img = self.save_images, self.display_images

        webcam_source = self.webcam_source
        verbose = self.mode_verbose
        half = self._half_mode

        save_nth_frame = self.save_nth_frame

        if verbose:
            print("Running in verbose mode")
            
        if webcam_source:
            save_path = str(Path(output_path).joinpath('webcam_output.mp4'))
            if verbose:
                print(f"Using webcam as source")
        else:
            save_path = str(Path(output_path))

        if verbose:
            print(f"Saving to path: {save_path}")
        run_name = self.run_name
        run_id = self.run_id
        append_run_id_to_files = self.append_run_id_to_files
        
        device = self._device
        if verbose:
            print(f"Using device {device}")

        # Second-stage classifier
        # @todo: look into this
        classify = False
        if classify:
            logging.debug("Using second-stage classifier")
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
            modelc.to(device).eval()

        # Set Dataloader
        # @todo: refactor
        vid_path, vid_writer = None, None
        if webcam_source:
            view_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
            capture_nth_frame = capture_nth_frame if capture_nth_frame > 0 else 4
            dataset = LoadStreams(source_path, img_size=inference_size, nth_frame=capture_nth_frame)
        else:
            dataset = LoadImages(source_path, img_size=inference_size, auto_size=64, print_output=False)

        # colors = [[0, 255, 0]]
        colours = self.get_bounding_box_colours()

        stats_times = []
        stats_images = 0
        stats_detections = 0


        stats_top_base = ""
        stats_bottom_base = ""

        if display_stats:
            stats_top_base =  f"Run: {self.run_name}"

        
        if display_extra_stats:
            stats_bottom_base = [
                f"Algorithm: YOLOR; Model: '{self.model_name}'; Inf. Size {inference_size}px",
                f"Thresh: Conf {conf_thres:.3f}; IoU {iou_thres:.3f}",
                f"System: {self.system_name}; Device: {device}",
            ]

            if classes_restrict:
                restricted_classes = []
                for cls in classes_restrict:
                    # @todo: not sure why "if cls in names:"" doesn't work here?
                    if cls <= len(class_names):
                        restricted_classes.append(class_names[cls])
                    else:
                        restricted_classes.append(str(cls))

                if len(restricted_classes):
                    stats_bottom_base.append("Only Classes:" + "; ".join(restricted_classes))

            stats_bottom_base = "\n".join(list(filter(None,stats_bottom_base)))

        source_time_start = time.time()

        # @todo: cleanup this chaotic mess of variables.
        source_frames_count = 0
        detect_count = 0
        source_video_w = source_video_h = 0
        running_classes = []
        source_all_classes_names = []
        running_conf = []
        running_detect_count = source_frame_current = running_frame_count = 0
        source_frame_prev = 0
        iteration_start = time.time()
        last_frame_check = None
        source_path_name = None
        source_vid_writing = None
        frame_save_path = None
        source_number = 0
        last_frame_saved = None
        source_frame_current = 0
        source_fps_calculated = 0
        video_mode = False

        source_avg_conf = inst_avg_conf = 0

        font_scale_calculated = False
        font_scale = {}

        source_data_checked = False

        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, inference_size, inference_size), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        logging.debug("Beginning inference...")
        for path, img, im0s, vid_cap in dataset:

            # @todo: sus output_path what img, im0s, vid_cap contain
            # want to rescale if it's too wide
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            stats_images += 1

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=augment)[0]
            inference_time = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes_restrict, agnostic=agnostic_nms)
            t2 = time_synchronized()
            nms_time = t2 - inference_time
            inference_time = inference_time - t1

            stats_times.append(inference_time)
            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            inst_detected_classes = []
            source_frames_count += 1
            source_frame_current += 1
            running_frame_count += 1
            this_detect_count = 0

            if not webcam_source:
                source_frame_current = dataset.frame
                if not source_data_checked or source_frame_current == 0 or source_frame_current < source_frame_prev:
                    source_data_checked = True
                    source_frames_count = 0
                    frame_save_path = None
                    source_path_name = None
                    source_number += 1
                    self.display("New source, resetting stats")
                    source_frame_current = 0
                    running_classes = []
                    inst_detected_classes = []
                    inst_detected_conf = []
                    running_conf = []
                    font_scale = 0
                    video_mode = not dataset.mode == 'images'
                    source_frames_total = dataset.nframes
                    source_fps_calculated = 0
                    font_scale_calculated = False

                    # Added 0.5
                    source_all_classes_names = ['-']
                    video_src_fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    source_run_time = 0
                    source_time_start = time_synchronized()
                    source_detections = 0

            source_frame_prev = source_frame_current
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam_source:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                # s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                detections = det is not None and len(det)

                source_path_name = Path(p).name
                if append_run_id_to_files:
                    save_path = str(Path(output_path).joinpath(Path(p).stem + '-' + run_id + Path(p).suffix))
                else:
                    save_path = str(Path(output_path).joinpath(source_path_name))
                
                detect_count = len(det)
                inst_detected_conf = []
                inst_detected_classes = []


                if detections:

                    if (save_img or view_img) and not font_scale_calculated:
                        font_scale = calc_text_size(im0)
                        font_scale_calculated = True

                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    if verbose:
                        # self.display("%d detections in %.3fs" % (detect_count, inference_time))
                        print(f"[yolor.detect] {source_frame_current}/{source_frames_total} {detect_count:d} detections")
                    # Write results
                    for *xyxy, conf, cls in det:
                        this_detect_count += 1
                        stats_detections += 1
                        source_detections += 1
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            txt_path = str(Path(output_path) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                        if source_data_checked and (save_img or view_img):
                            cls_int = int(cls)
                            inst_detected_conf.append(float(conf))
                            inst_detected_classes.append(cls_int)
                            if display_bounding_boxes:  # Add bbox to image
                                # @todo: put in middle of bounding box
                                if display_bounding_box_labels:
                                    label = f"{class_names[cls_int].title()} {conf*100:.0f}%"
                                else:
                                    label = None
                                plot_one_box(xyxy, im0, label=label, color=colours[cls_int], line_thickness=2, text_color=[0,0,0], line_type=cv2.LINE_AA, font_scale=font_scale["text_font_scale"])
                            #if video_resize:
                                    # im0 = cv2.resize(im0, (video_resize_width, video_resize_height), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
                else:
                    if verbose:
                        print(f"[yolor.detect] {source_frame_current}/{source_frames_total} No detections")


                # @todo: refactor!
                # Print summary stats on image
                if (save_img or view_img) and display_stats and source_data_checked:
                    if not font_scale_calculated:
                        font_scale = calc_text_size(im0)
                        font_scale_calculated = True

                    running_detect_count += detect_count
                    if len(inst_detected_classes) > 0:
                        inst_detected_classes = list(set(inst_detected_classes))
                        inst_detected_classes.sort()
                        inst_detected_names = [class_names[x] for x in inst_detected_classes]
                    else:
                        inst_detected_names = ['-']

                    new_running_classes = running_classes + inst_detected_classes
                    new_running_classes = list(set(new_running_classes))
                    if len(new_running_classes) > len(running_classes):
                        running_classes = new_running_classes
                        running_classes.sort()
                        source_all_classes_names = [class_names[x] for x in running_classes]

                    inst_avg_conf = 0
                    if len(inst_detected_conf):
                        running_conf += inst_detected_conf
                        inst_avg_conf = np.average(inst_detected_conf)
                        source_avg_conf = np.average(running_conf)
                    
                    right_now = time_synchronized()
                    processing_time = (right_now - source_time_start)
                    if source_frame_current >= 5:
                        source_fps_calculated = source_frame_current / processing_time
                    source_run_time = source_frame_current / video_src_fps
                    
                    add_text_heading(im0, "yolor-edge / Ewan Thompson / 2021", font_scale)

                    # plot_text_with_border(img=im0, starting_row = 1, starting_column=1, label = 'yolor-edge / Ewan Thompson / 2021', font_scale = font_scale_h, font_face = 2)
                    stats_top = ""
                    stats_top += "Instantaneous:\n"
                    stats_top += f" Detections: {detect_count:d}\n"
                    stats_top += f" Objects: {', '.join(inst_detected_names)}\n"
                    stats_top += f" Avg. Confidence: {inst_avg_conf*100:.2f}%\n"
                    if source_fps_calculated > 0:
                        stats_top += f" FPS: {source_fps_calculated:.2f}\n"
                    else:
                        stats_top += f" FPS: ...\n"
                    stats_top += f" Inference Time: {(1E3 * inference_time):.3f}ms\n"
                    stats_top += f" NMS Time: {(1E3 * nms_time):.3f}ms\n"
                    # plot_text_with_border(img=im0, starting_row=4, starting_column=2, label = stats_top, font_scale = font_scale)
                    add_text(im0, stats_top, font_scale, 3, 1)

                    if display_extra_stats:
                        stats_bottom = f"{stats_top_base}\n"
                        stats_bottom += f"Source: '{source_path_name}'\n"

                        stats_bottom += f" Size: {source_video_w}x{source_video_h}\n"
                        stats_bottom += f" FPS: {video_src_fps:.3f}\n"
                        stats_bottom += f" Frame: {source_frame_current}/{source_frames_total-1}\n"
                        stats_bottom += f" Runtime: {source_run_time:.2f}s\n"

                        stats_bottom += f" Objects: {', '.join(source_all_classes_names)}\n"
                        # stats_bottom += f" Frames w/ Detections: {source_detections:d}\n"
                        stats_bottom += f" Avg. Confidence: {source_avg_conf*100:.2f}%\n"
                        stats_bottom += f" Processing Time: {processing_time:.2f}s\n"
                        stats_bottom += stats_bottom_base
                        # plot_text_with_border(img=im0, starting_row=2, starting_column=1, from_bottom = True, label = stats_bottom, font_scale = font_scale)
                        add_text(im0, stats_bottom, font_scale, 1, 1, True)

                # Display results
                if view_img:
                    cv2.imshow(self.run_name, im0)
                    # if cv2.waitKey(1) == ord('q'):  # q to quit
                        # raise StopIteration

                # @todo: save webcam frames
                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'images':
                        if webcam_source:
                            # @todo: save as a video instead, obviuosly indivd images = very slow
                            pass
                            # cv2.imwrite(f"{save_path}-webcam_frame{source_frame_count}.png", im0)
                        else:
                            cv2.imwrite(save_path, im0)
                    else:
                        if source_vid_writing != source_number:  # new video
                            source_vid_writing = source_number
                            vid_path = save_path
                            self.display(f"New vid path: {vid_path}")
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer

                            fourcc = 'mp4v'  # output video codec
                            video_src_fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            source_video_w = vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                            source_video_h = vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                            source_video_w = int(source_video_w)
                            source_video_h = int(source_video_h)
                            vid_writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*fourcc), video_src_fps, (source_video_w, source_video_h))
                        vid_writer.write(im0)

                if save_frames:
                    # Save the current frame to an image, if setup to do so
                    save_frame = False
                    if frame_save_path is None:
                        frame_save_path = vid_path + "-frames"
                        Path(frame_save_path).mkdir(parents=True, exist_ok=True)

                    if save_nth_frame == 1 or save_nth_frame < 0:
                        save_frame = True
                    else:
                        if save_nth_frame > 0 and (source_frame_current % save_nth_frame) == 0:
                            save_frame = True

                    if not save_frame and (last_frame_saved is None or source_frame_current < last_frame_saved):
                        save_frame = True

                    if save_frame and frame_save_path is not None:
                        self.display(f"Saving frame {source_frame_current}")
                        last_frame_saved = source_frame_current
                        frame_save_to = f"{frame_save_path}/frame-{source_frame_current:05d}.png"
                        cv2.imwrite(frame_save_to, im0)

        if isinstance(vid_writer, cv2.VideoWriter):
            vid_writer.release()

        cv2.destroyAllWindows()
        if save_txt or save_img:
            print('Results saved to %s' % Path(output_path))

        print("Summary:\n\tImages: {images}\n\tDetections: {detections}\n\tAvg. Inf. Time: {avgtime:.3f}".format(images=stats_images, detections = stats_detections, avgtime = average(stats_times)))
        print('Done. (%.3fs)' % (time.time() - t0))