
from pathlib import Path
from datetime import datetime
import wandb
import pandas as pd


datestr="%Y%m%d-%H%M%S"
out_file = Path(f"csv/yolor-edge-runs-{datetime.strftime(datetime.now(), datestr)}")

api = wandb.Api()

runs = api.runs(path="ewth/yolor-edge", filters={'state':'finished', 'config.image_size': {"$gte": 512}}, per_page=10000)
summary_list, config_list, name_list = [], [], []

labels = [
    'id',
    'state',
    'task',
    'i_size', # 'image_size',
    'b_size', # 'batch_size',
    'weight',
    'cfg',
    'nc',
    'i_time', # 'time.inference',
    'time.stats',
    'time.total',
    'seen'
]

perf_labels = ['ap','ap_50','ap_75','ap_S','ap_M','ap_L','ar_1','ar_10','ar','ar_S','ar_M','ar_L']
labels = labels+perf_labels

data_list = []
for run in runs: 
    eval_stats = run.summary['eval']['stats']
    performance = dict(zip(perf_labels, eval_stats))
    seen = -1
    if "class_stats" in run.summary:
        seen = run.summary["class_stats"][0]['stats']['seen']
    else:
        seen = run.summary['stats']['seen']

    data = {
        'id': run.id,
        'state': run.state,
        'task': run.config['task'],
        'image_size': run.config['image_size'],
        'batch_size': run.config['batch_size'],
        'weight': Path(run.config['weights'][0]).name.replace('.pt',''),
        'cfg': Path(run.config['cfg']).name,
        'nc': run.config['nc'],
        'time.inference': run.summary['time.inference'],
        'time.stats': run.summary['time.stats'],
        'time.total': run.summary['time.total'],
        'seen': seen
    }
    data = {**data, **performance}
    data_list.append(data)

runs_df = pd.DataFrame(data_list, columns=labels)

runs_df.to_csv(f"{out_file}-raw.csv")
