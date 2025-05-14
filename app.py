import torch
import numpy as np
from model import Transformer_ST, Model_all, ScoreMatch_module, SMASH
from torch.optim import AdamW
import argparse
from model.Dataset import get_dataloader
from model.Metric import get_calibration_score
import time
import datetime
import pickle
import os
from tqdm import tqdm
import random
import math


def setup_init(args):
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def model_name():
    TIME = int(time.time())
    TIME = time.localtime(TIME)
    return time.strftime("%Y-%m-%d %H:%M:%S", TIME)


def normalization(x, MAX, MIN):
    return (x - MIN) / (MAX - MIN)


def denormalization(x, MAX, MIN, log_normalization=False):
    if log_normalization:
        return torch.exp(x.detach().cpu() * (MAX - MIN) + MIN)
    else:
        return x.detach().cpu() * (MAX - MIN) + MIN


def get_args():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--seed', type=int, default=1234, help='')
    parser.add_argument('--model', type=str, default='SMASH', help='')
    parser.add_argument('--mode', type=str, default='train', help='')
    parser.add_argument('--total_epochs', type=int, default=1000, help='')
    parser.add_argument('--machine', type=str, default='none', help='')
    parser.add_argument('--dim', type=int, default=2, help='', choices=[1, 2, 3])
    parser.add_argument('--dataset',
                        type=str,
                        default='Earthquake',
                        choices=['Earthquake', 'crime', 'football'],
                        help='')
    parser.add_argument('--batch_size', type=int, default=64, help='')
    parser.add_argument('--samplingsteps', type=int, default=500, help='')
    parser.add_argument('--per_step', type=int, default=250, help='')
    parser.add_argument('--cuda_id', type=str, default='0', help='')
    parser.add_argument('--n_samples', type=int, default=100, help='')
    parser.add_argument('--log_normalization', type=int, default=1, help='')
    parser.add_argument('--weight_path',
                        type=str,
                        default='./ModelSave/dataset_Earthquake_model_SMSTPP/model_300.pkl',
                        help='')
    parser.add_argument('--save_path', type=str, help='')
    parser.add_argument('--cond_dim', type=int, default=64, help='')
    parser.add_argument('--sigma_time', type=float, default=0.05, help='')
    parser.add_argument('--sigma_loc', type=float, default=0.05, help='')
    parser.add_argument('--langevin_step', type=float, default=0.005, help='')
    parser.add_argument('--loss_lambda', type=float, default=0.5, help='')
    parser.add_argument('--loss_lambda2', type=float, default=1, help='')
    parser.add_argument('--smooth', type=float, default=0.0, help='')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print(args)
    return args


opt = get_args()
device = torch.device("cuda:{}".format(opt.cuda_id) if opt.cuda else "cpu")

if opt.dataset == 'HawkesGMM':
    opt.dim = 1

os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.cuda_id)


def data_loader(opt):

    f = open('dataset/{}/data_train.pkl'.format(opt.dataset), 'rb')
    train_data = pickle.load(f)
    train_data = [[list(i) for i in u] for u in train_data]
    f = open('dataset/{}/data_val.pkl'.format(opt.dataset), 'rb')
    val_data = pickle.load(f)
    val_data = [[list(i) for i in u] for u in val_data]
    f = open('dataset/{}/data_test.pkl'.format(opt.dataset), 'rb')
    test_data = pickle.load(f)
    test_data = [[list(i) for i in u] for u in test_data]

    if not opt.log_normalization:
        train_data = [[[i[0], i[0] - u[index - 1][0] if index > 0 else i[0], i[1] + 1] + i[2:]
                       for index, i in enumerate(u)] for u in train_data]
        val_data = [[[i[0], i[0] - u[index - 1][0] if index > 0 else i[0], i[1] + 1] + i[2:]
                     for index, i in enumerate(u)] for u in val_data]
        test_data = [[[i[0], i[0] - u[index - 1][0] if index > 0 else i[0], i[1] + 1] + i[2:]
                      for index, i in enumerate(u)] for u in test_data]
    else:
        train_data = [
            [[i[0],
              math.log(max(i[0] - u[index - 1][0], 1e-4)) if index > 0 else math.log(max(i[0], 1e-4)), i[1] + 1] + i[2:]
             for index, i in enumerate(u)] for u in train_data
        ]
        val_data = [
            [[i[0],
              math.log(max(i[0] - u[index - 1][0], 1e-4)) if index > 0 else math.log(max(i[0], 1e-4)), i[1] + 1] + i[2:]
             for index, i in enumerate(u)] for u in val_data
        ]
        test_data = [
            [[i[0],
              math.log(max(i[0] - u[index - 1][0], 1e-4)) if index > 0 else math.log(max(i[0], 1e-4)), i[1] + 1] + i[2:]
             for index, i in enumerate(u)] for u in test_data
        ]

    data_all = train_data + test_data + val_data

    Max, Min = [], []
    for m in range(opt.dim + 2):
        if m > 0:
            Max.append(max([i[m] for u in data_all for i in u]))
            Min.append(min([i[m] for u in data_all for i in u]))
        else:
            Max.append(1)
            Min.append(0)

    if opt.dim == 3:
        Max[2] = 1
        Min[2] = 0
        opt.num_types = int(max([i[2] for u in data_all for i in u]))
    else:
        opt.num_types = 1

    train_data = [[[normalization(i[j], Max[j], Min[j]) for j in range(len(i))] for i in u] for u in train_data]
    test_data = [[[normalization(i[j], Max[j], Min[j]) for j in range(len(i))] for i in u] for u in test_data]
    val_data = [[[normalization(i[j], Max[j], Min[j]) for j in range(len(i))] for i in u] for u in val_data]
    trainloader = get_dataloader(train_data, opt.batch_size, D=opt.dim, shuffle=True)
    testloader = get_dataloader(test_data, opt.batch_size, D=opt.dim, shuffle=False)
    valloader = get_dataloader(test_data, opt.batch_size, D=opt.dim, shuffle=False)
    print('Min & Max', (Max, Min), opt.num_types)
    # sys.exit()
    return trainloader, testloader, valloader, (Max, Min)


def Batch2toModel(batch, transformer):

    if opt.dim == 2:
        event_time_origin, event_time, lng, lat = map(lambda x: x.to(device), batch)
        event_loc = torch.cat((lng.unsqueeze(dim=2), lat.unsqueeze(dim=2)), dim=-1)

    if opt.dim == 3:
        event_time_origin, event_time, mark, lng, lat = map(lambda x: x.to(device), batch)

        event_loc = torch.cat((mark.unsqueeze(dim=2), lng.unsqueeze(dim=2), lat.unsqueeze(dim=2)), dim=-1)

    event_time = event_time.to(device)
    event_time_origin = event_time_origin.to(device)
    event_loc = event_loc.to(device)

    enc_out, mask = transformer(event_loc, event_time_origin)
    # print(event_time.size(),event_loc.size(), enc_out.size(),mask.size())

    enc_out_non_mask = []
    event_time_non_mask = []
    event_loc_non_mask = []
    for index in range(mask.shape[0]):
        length = int(sum(mask[index]).item())
        if length > 1:
            enc_out_non_mask += [i.unsqueeze(dim=0) for i in enc_out[index][:length - 1]]
            event_time_non_mask += [i.unsqueeze(dim=0) for i in event_time[index][1:length]]
            event_loc_non_mask += [i.unsqueeze(dim=0) for i in event_loc[index][1:length]]

    enc_out_non_mask = torch.cat(enc_out_non_mask, dim=0)
    event_time_non_mask = torch.cat(event_time_non_mask, dim=0)
    event_loc_non_mask = torch.cat(event_loc_non_mask, dim=0)

    event_time_non_mask = event_time_non_mask.reshape(-1, 1, 1)
    event_loc_non_mask = event_loc_non_mask.reshape(-1, 1, opt.dim)

    enc_out_non_mask = enc_out_non_mask.reshape(event_time_non_mask.shape[0], 1, -1)
    return event_time_non_mask, event_loc_non_mask, enc_out_non_mask


def LR_warmup(lr, epoch_num, epoch_current):
    return lr * (epoch_current + 1) / epoch_num


if __name__ == "__main__":

    setup_init(opt)

    print('dataset:{}'.format(opt.dataset))
    from datetime import datetime
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

    model_path = opt.save_path

    if not os.path.exists('./ModelSave'):
        os.mkdir('./ModelSave')

    if 'train' in opt.mode and not os.path.exists(model_path):
        os.mkdir(model_path)

    trainloader, testloader, valloader, (MAX, MIN) = data_loader(opt)

    model = ScoreMatch_module(dim=1 + opt.dim, condition=True, cond_dim=opt.cond_dim,
                              num_types=opt.num_types).to(device)

    decoder = SMASH(model,
                    sigma=(opt.sigma_time, opt.sigma_loc),
                    seq_length=1 + opt.dim,
                    sampling_timesteps=opt.samplingsteps,
                    n_samples=opt.n_samples,
                    langevin_step=opt.langevin_step,
                    num_types=opt.num_types,
                    loss_lambda=opt.loss_lambda,
                    loss_lambda2=opt.loss_lambda2,
                    smooth=opt.smooth).to(device)

    transformer = Transformer_ST(d_model=opt.cond_dim,
                                 d_rnn=opt.cond_dim * 4,
                                 d_inner=opt.cond_dim * 2,
                                 n_layers=4,
                                 n_head=4,
                                 d_k=16,
                                 d_v=16,
                                 dropout=0.1,
                                 device=device,
                                 loc_dim=opt.dim,
                                 CosSin=True,
                                 num_types=opt.num_types).to(device)

    Model = Model_all(transformer, decoder)
    if opt.mode == 'test':
        Model.load_state_dict(torch.load(opt.weight_path))
        print('Weight loaded!!')
    total_params = sum(p.numel() for p in Model.parameters())
    print(f"Number of parameters: {total_params}")

    warmup_steps = 5
    # training
    optimizer = AdamW(Model.parameters(), lr=1e-3, betas=(0.9, 0.99))
    step, early_stop = 0, 0
    min_loss_test = 1e20
    for itr in tqdm(range(opt.total_epochs)):

        print('epoch:{}'.format(itr))

        if (itr % 10 == 0) or opt.mode == 'test':
            print('Evaluate!')

            Model.eval()

            # testing set
            loss_test_all = 0.0
            for batch in testloader:
                event_time_non_mask, event_loc_non_mask, enc_out_non_mask = Batch2toModel(batch, Model.transformer)
                loss = Model.decoder(torch.cat((event_time_non_mask, event_loc_non_mask), dim=-1), enc_out_non_mask)
                loss_test_all += loss.item() * event_time_non_mask.shape[0]

            current_step = 0
            is_last = False
            last_sample = None
            while current_step < opt.samplingsteps:
                if (current_step + opt.per_step) >= opt.samplingsteps:
                    is_last = True
                cs_time_all = torch.zeros(5)
                cs_loc_all = torch.zeros(5)
                cs2_time_all = torch.zeros(5)
                cs2_loc_all = torch.zeros(5)
                acc_all = 0
                ece_all = 0
                correct_list_all = torch.zeros(10)
                num_list_all = torch.zeros(10)
                mae_temporal, mae_spatial, total_num = 0.0, 0.0, 0.0
                sampled_record_all = []
                gt_record_all = []
                for idx, batch in enumerate(testloader):
                    # First denormalize `time` & `loc` for further UQ/UC stage
                    event_time_non_mask, event_loc_non_mask, enc_out_non_mask = Batch2toModel(batch, Model.transformer)
                    real_time = denormalization(event_time_non_mask[:, 0, :], MAX[1], MIN[1], opt.log_normalization)
                    real_loc = event_loc_non_mask[:, 0, :]
                    real_loc = denormalization(real_loc, torch.tensor([MAX[2:]]), torch.tensor([MIN[2:]]))
                    total_num += real_loc.shape[0]
                    gt_record_all.append(torch.cat((real_time, real_loc), -1))


                    sampled_seq_all, sampled_seq_temporal_all, sampled_seq_spatial_all, sampled_seq_mark_all = [], [], [], []
                    for i in range(int(300 / opt.n_samples)):
                        sampled_seq, score_mark = Model.decoder.sample_from_last(
                            batch_size=event_time_non_mask.shape[0],
                            step=opt.per_step,
                            is_last=is_last,
                            cond=enc_out_non_mask,
                            last_sample=last_sample[idx][i] if last_sample is not None else None)
                        # print(sampled_seq, score_mark)
                        sampled_seq_all.append(
                            (sampled_seq.detach(), score_mark.detach() if score_mark is not None else None))
                        sampled_seq_temporal_all.append(
                            denormalization(sampled_seq[:, :, 0], MAX[1], MIN[1], opt.log_normalization))
                        sampled_seq_spatial_all.append(
                            denormalization(sampled_seq[:, :, -2:], torch.tensor([MAX[-2:]]), torch.tensor([MIN[-2:]])))
                        sampled_seq_mark_all.append(score_mark.detach().cpu())

                    sampled_record_all.append(sampled_seq_all)
                    gen_time = torch.cat(sampled_seq_temporal_all, 1).mean(1, keepdim=True)
                    assert real_time.shape == gen_time.shape
                    mae_temporal += torch.abs(real_time - gen_time).sum().item()

                    gen_loc = torch.cat(sampled_seq_spatial_all, 1).mean(1)
                    assert real_loc[:, -2:].shape == gen_loc.shape
                    mae_spatial += torch.sqrt(torch.sum((real_loc[:, -2:] - gen_loc)**2, dim=-1)).sum().item()

                    if score_mark is not None:
                        gen_mark = torch.mode(torch.max(torch.cat(sampled_seq_mark_all, 1), dim=-1)[1], 1)[0]
                        acc_all += torch.sum(gen_mark == (real_loc[:, 0] - 1))

                    if opt.mode == 'test':
                        calibration_score = get_calibration_score(sampled_seq_temporal_all, sampled_seq_spatial_all,
                                                                  sampled_seq_mark_all, real_time, real_loc)
                        cs_time_all += calibration_score[0]
                        cs_loc_all += calibration_score[1]
                        cs2_time_all += calibration_score[2]
                        cs2_loc_all += calibration_score[3]
                        if score_mark is not None:
                            ece_all += calibration_score[4]
                            correct_list_all += calibration_score[5]
                            num_list_all += calibration_score[6]

                last_sample = sampled_record_all
                current_step += opt.per_step

                if opt.mode == 'test':
                    cs_time_all /= total_num
                    cs_loc_all /= total_num
                    cs2_time_all /= total_num
                    cs2_loc_all /= total_num
                    ece_all /= total_num
                    correct_list_all /= num_list_all
                    print('Step: ', current_step)
                    print('Calibration Score Quantile: ', cs2_time_all, cs2_loc_all)
                    print('Calibration Score: ', cs_time_all.mean().item(), cs_loc_all.mean().item())
                    print('MAE: ', mae_temporal / total_num, mae_spatial / total_num)
                    if score_mark is not None:
                        print('Mark: ', acc_all / total_num, ece_all, correct_list_all)

                global_step = itr if opt.mode == 'train' else current_step

            if opt.mode == 'train':
                torch.save(Model.state_dict(), model_path + 'model_{}.pkl'.format(itr))
                print('Model Saved to {}'.format(model_path + 'model_{}.pkl').format(itr))
            if opt.mode == 'test':
                torch.save([sampled_record_all, gt_record_all],
                           './samples/test_{}_{}_sigma_{}_{}_steps_{}_log_{}.pkl'.format(
                               opt.dataset, opt.model, opt.sigma_time, opt.sigma_loc, opt.samplingsteps,
                               opt.log_normalization))
                break

            # validation set
            loss_test_all = 0.0
            total_num = 0.0
            for batch in valloader:
                event_time_non_mask, event_loc_non_mask, enc_out_non_mask = Batch2toModel(batch, Model.transformer)
                loss = Model.decoder(torch.cat((event_time_non_mask, event_loc_non_mask), dim=-1), enc_out_non_mask)
                loss_test_all += loss.item() * event_time_non_mask.shape[0]
                total_num += event_time_non_mask.shape[0]

            if loss_test_all < min_loss_test and opt.mode == 'train':
                early_stop += 1
                torch.save(Model.state_dict(), model_path + 'model_best.pkl')
                if early_stop >= 50:
                    break
            else:
                early_stop = 0
            min_loss_test = min(min_loss_test, loss_test_all)

        if itr < warmup_steps:
            for param_group in optimizer.param_groups:
                lr = LR_warmup(1e-3, warmup_steps, itr)
                param_group["lr"] = lr
        else:
            for param_group in optimizer.param_groups:
                lr = 1e-3 - (1e-3 - 5e-5) * (itr - warmup_steps) / opt.total_epochs
                param_group["lr"] = lr

        Model.train()
        loss_all, total_num = 0.0, 0.0
        for batch in trainloader:

            event_time_non_mask, event_loc_non_mask, enc_out_non_mask = Batch2toModel(batch, Model.transformer)
            loss = Model.decoder(torch.cat((event_time_non_mask, event_loc_non_mask), dim=-1), enc_out_non_mask)

            optimizer.zero_grad()
            loss.backward()
            loss_all += loss.item() * event_time_non_mask.shape[0]

            torch.nn.utils.clip_grad_norm_(Model.parameters(), 1.)
            optimizer.step()

            step += 1

            total_num += event_time_non_mask.shape[0]

        with torch.cuda.device("cuda:{}".format(opt.cuda_id)):
            torch.cuda.empty_cache()
        print('------- Training ---- Epoch: {} ;  Loss: {} --------'.format(itr, loss_all / total_num))
