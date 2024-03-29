import math
from copy import deepcopy
import torch
from collections import OrderedDict

class ModelEMA:
    def __init__(self, model, num_gpus):
        # Create EMA
        self.ema = deepcopy(model).eval()  # FP32 EMA
        if num_gpus>1:
            self.ema = torch.nn.DataParallel(self.ema)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model, keep_rate=0.9996):
        # Update EMA parameters
        with torch.no_grad():
            new_teacher_dict = OrderedDict()
            student_model_dict = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if k in student_model_dict.keys():
                    new_teacher_dict[k] = (
                        student_model_dict[k] *
                        (1 - keep_rate) + v * keep_rate
                    )
        self.ema.load_state_dict(new_teacher_dict)
