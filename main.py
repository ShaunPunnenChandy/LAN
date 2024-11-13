import torch
from data import Dataset
from model import get_model
from metric import cal_batch_psnr_ssim
import pandas as pd
from tqdm import tqdm
import argparse
from adapt import zsn2n, nbr2nbr
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, required=True, choices=["finetune", "lan"])
parser.add_argument("--self_loss", type=str, required=True, choices=["nbr2nbr", "zsn2n"])
args = parser.parse_args()

if args.self_loss == "zsn2n":
    loss_func = zsn2n.loss_func
elif args.self_loss == "nbr2nbr":
    loss_func = nbr2nbr.loss_func
else:
    raise NotImplementedError

model_generator = get_model
model = model_generator()
for param in model.parameters():
    param.requires_grad = args.method == "finetune"
print("trainable model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

dataloader = torch.utils.data.DataLoader(Dataset("polyu/lq", "polyu/gt"), batch_size=1, shuffle=False)
lr = 5e-4 if args.method == "lan" else 5e-6
class Lan(torch.nn.Module):
    def __init__(self, shape):
        super(Lan, self).__init__()
        self.phi = torch.nn.parameter.Parameter(torch.zeros(shape), requires_grad=True)
    def forward(self, x):
        return x + torch.tanh(self.phi)

logs_key = ["psnr", "ssim"]
total_logs = {key: [] for key in logs_key}
inner_loop = 20
p_bar = tqdm(dataloader, ncols=100, desc=f"{args.method}_{args.self_loss}")
for lq, gt in p_bar:
    lq = lq.cuda()
    gt = gt.cuda()
    lan = Lan(lq.shape).cuda() if args.method == "lan" else torch.nn.Identity()
    tmp_batch_size = lq.shape[0]
    model = model_generator()
    for param in model.parameters():
        param.requires_grad = args.method == "finetune"

    params = list(lan.parameters()) if args.method == "lan" else list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    logs = {key: [] for key in logs_key}

    for i in range(inner_loop):
        optimizer.zero_grad()
        adapted_lq = lan(lq)
        with torch.no_grad():
            pred = model(adapted_lq).clip(0, 1)
        loss = loss_func(adapted_lq, model, i, inner_loop)
        loss.backward()
        optimizer.step()
        psnr, ssim = cal_batch_psnr_ssim(pred, gt)
        for key in logs_key:
            logs[key].append(locals()[key])
    else:
        with torch.no_grad():
            adapted_lq = lan(lq)
            pred = model(adapted_lq).clip(0, 1)
            psnr, ssim = cal_batch_psnr_ssim(pred, gt)
        for key in logs_key:
            logs[key].append(locals()[key])

    for key in logs_key:
        total_logs[key].extend(np.array(logs[key]).transpose())
    p_bar.set_postfix(
        PSNR=f"{np.array(total_logs['psnr']).mean(0)[0]:.2f}->{np.array(total_logs['psnr']).mean(0)[-1]:.2f}",
        SSIM=f"{np.array(total_logs['ssim']).mean(0)[0]:.3f}->{np.array(total_logs['ssim']).mean(0)[-1]:.3f}"
    )
    df_dict = {
        "idx": [i for i in range(len(total_logs['psnr'])) for _ in range(inner_loop+1)],
        "loop": [i for i in range(inner_loop+1)] * len(total_logs['psnr']),
    }
    for key in logs_key:
        df_dict[key] = [value for value_list in total_logs[key] for value in value_list]
    df = pd.DataFrame(df_dict)
    df.to_csv(f"result_{args.method}_{args.self_loss}.csv", index=False)
print(df.groupby('loop').mean()[['psnr', 'ssim']])