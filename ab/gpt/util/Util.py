import os.path
import re
import shutil
from pathlib import Path

from ab.gpt.util.Const import new_lemur_nn_dir, new_nn_file, new_lemur_stat_dir


# todo: Verify that the model's accuracy does not decrease by more than 10%, or increase at some epochs
def nn_accepted(nn_dir):
    accepted = True
    return accepted


# todo: Verify if model has implementation of all required methods, and use all mentioned hyperparameters, like 'lr', 'momentum'
# todo: Optimize code with library like 'deadcode' (after: pip install deadcode)
def verify_nn_code(nn_dir, nn_file):
    verified = True
    error_message = ''
    if not verified:
        with open(nn_dir / f"error_code_verification.txt", "w+") as error_file:
            error_file.write(f"Code verification failed: {error_message}")
    return verified


def exists(f):
    return f and os.path.exists(f)

def extract_str(s: str, start: str, end: str):
    try:
        return s[:s.rindex(end)].split(start)[-1].strip()
    except:
        return None


def extract_code(txt):
    return next(filter(None, map(lambda l: extract_str(txt, *l), (('<nn>', '</nn>'), ('```python', '```'), ('```', '```')))), '')


def extract_hyperparam(txt):
    return next(filter(None, map(lambda l: extract_str(txt, *l), (('<hp>', '</hp>'),))), '')


def copy_to_lemur(df, gen_nn_dir, name):
    Path(new_lemur_nn_dir).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(gen_nn_dir / new_nn_file, new_lemur_nn_dir / f'{name}.py')
    nn_model_dir = new_lemur_stat_dir / name
    if df is None:
        Path(nn_model_dir).mkdir(parents=True, exist_ok=True)
        for f_nm in [f for f in os.listdir(gen_nn_dir) if re.match(r'[0-9]+\.json', f)]:
            shutil.copyfile(gen_nn_dir / f_nm, nn_model_dir / f_nm)
    else:
        dr_nm = new_lemur_stat_dir / f"{df['task']}_{df['dataset']}_{df['metric']}_{name}"
        Path(dr_nm).mkdir(parents=True, exist_ok=True)
        for f_nm in [f for f in os.listdir(gen_nn_dir) if re.match(r'[0-9]+\.json', f)]:
            shutil.copyfile(gen_nn_dir / f_nm, dr_nm / f_nm)
