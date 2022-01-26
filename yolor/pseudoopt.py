"""
Quickly chucking together class to replicate parser object
"""

from yolor.models.models import save_weights


class valopt:
    weights: list
    names: str
    batch_size: int
    img_size: int
    log_images: bool
    conf_thres: float
    iou_thres: float
    task: str
    device: str
    single_cls: bool
    augment: bool
    verbose: bool
    save_txt: bool
    save_json: bool
    project: str
    name: str
    exist_ok: bool
    cfg: str
    def __init__(
        self,
        weights,
        names,
        batch_size,
        img_size,
        log_images,
        conf_thres,
        iou_thres,
        task,
        device,
        single_cls,
        augment,
        verbose,
        save_txt,
        save_json,
        project,
        name,
        exist_ok,
        cfg
    ):
        self.weights = weights
        self.names = names
        self.batch_size = batch_size
        self.img_size = img_size
        self.log_images = log_images
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.task = task
        self.device = device
        self.single_cls = single_cls
        self.augment = augment
        self.verbose = verbose
        self.save_txt = save_txt
        self.save_json = save_json
        self.project = project
        self.name = name
        self.exist_ok = exist_ok
        self.cfg = cfg