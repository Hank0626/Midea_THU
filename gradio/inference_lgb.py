import gradio as gr
from scipy.interpolate import interp1d
import numpy as np
import lightgbm as lgb


model_path = '../gradio_model/lgbm/lgm_params_2.txt'

def lgb_infer(trad, new):
    new = new[~np.isnan(new).any(axis=1), :]
    new = new[np.isfinite(new).all(axis=1), :]

    f = interp1d(new[:, 0], new[:, 1], kind="cubic", fill_value="extrapolate")

    y_new = f(trad[:, 0])
    
    new_align = np.c_[trad[:, 0], y_new]

    model = lgb.Booster(model_file = model_path)

    y = model.predict(new_align, num_iteration=model.best_iteration)
    
    return trad[:, 0], y_new, trad[:, 1], y, 0

