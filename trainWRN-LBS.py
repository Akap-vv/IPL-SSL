import argparse
import logging
import math
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import random
import shutil
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset import MyDataset, MyDatasetSampler,UnlabelMyDataset2,UnlabelMyDataset
from dataset import data_reader
from models.agument import DataTransform
# from dataset.data_process import DATASET_GETTERS
from utils import AverageMeter, accuracy

logger = logging.getLogger(__name__)
best_acc = 0


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='6', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=15,
                        help='number of labeled data')
    parser.add_argument('--num-classes', type=int, default=5,
                        help='number of classes')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnext'],
                        help='dataset name')
    parser.add_argument('--early_stop', default=30, type=int,
                        help='number of max steps to eval')
    parser.add_argument('--total-steps', default=2**20, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=20, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=2, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=10, type=int,
                        help="random seed")
    parser.add_argument('--agu_level', default=3, type=int,
                        help="strong agument level")
    parser.add_argument('--num_branch', default=3, type=int,
                        help="use how many branches,2 or 3")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    parser.add_argument('--alpha', default=0.75, type=float,
                        help="to make a beta")
    parser.add_argument('--weight', default=0.5, type=float,
                       help="to balance the loss")

    args = parser.parse_args()
    global best_acc

    def create_model(args):
        if args.arch == 'wideresnet':
            import models.wideresnet as models
            model = models.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes)
        elif args.arch == 'resnext':
            import models.resnext as models
            model = models.build_resnext(cardinality=args.model_cardinality,
                                         depth=args.model_depth,
                                         width=args.model_width,
                                         num_classes=args.num_classes)
        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters())/1e6))
        return model

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}",)

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)

    # if args.dataset == 'cifar10':
    # args.num_classes = 5
    if args.arch == 'wideresnet':
        args.model_depth = 28
        args.model_width = 2
    elif args.arch == 'resnext':
        args.model_cardinality = 4
        args.model_depth = 28
        args.model_width = 4
    # train_path = './data/MQTT/train.npy'
    # test_path = './data/MQTT/test.npy'
    train_X = np.load('./data/CICIDS2017/LBS/train_x_lbs33_choose1.npy')
    train_Y = np.load('./data/CICIDS2017/LBS/train_y_lbs33_choose1.npy')
    test_X = np.load('./data/CICIDS2017/LBS/val_x_lbs33_choose4.npy')
    test_Y = np.load('./data/CICIDS2017/LBS/val_y_lbs33_choose4.npy')
    # ind1, time1 = np.unique(train_Y, return_counts=True)
    # ind2, time2 = np.unique(test_Y, return_counts=True)
    # print(ind1, time1)
    # print(ind2, time2)
    # import pdb;
    # pdb.set_trace()
    # train_X, train_Y = data_reader(train_path)
    # test_X, test_Y = data_reader(test_path)
    # train_X = np.load('./data/ISCX2012/LBS/train_x_lbs33_clean2.npy')
    # train_Y = np.load('./data/ISCX2012/LBS/train_y_lbs33_clean2.npy')
    # test_X = np.load('./data/ISCX2012/LBS/val_x_lbs33_clean2.npy')
    # test_Y = np.load('./data/ISCX2012/LBS/val_y_lbs33_clean2.npy')
    # train_X = np.load('./data/MQTT/LBS/train_x_lbs33_clean1.npy')
    # train_Y = np.load('./data/MQTT/LBS/train_y_lbs33_clean1.npy')
    # test_X = np.load('./data/MQTT/LBS/val_x_lbs33_clean1.npy')
    # test_Y = np.load('./data/MQTT/LBS/val_y_lbs33_clean1.npy')
    # train_X = np.load('./data/vpn2016/train_x_s30.npy')
    # train_Y = np.load('./data/vpn2016/train_y_s30.npy')
    # test_X = np.load('./data/vpn2016/test_x_s30_choose.npy')
    # test_Y = np.load('./data/vpn2016/test_y_s30_choose.npy')

    # import pdb;pdb.set_trace()
    # def x_u_split(num_labeled, num_classes, labels):
    #     label_per_class = num_labeled // num_classes
    #     labels = np.array(labels)
    #     labeled_idx = []
    #     # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    #     unlabeled_idx = np.array(range(len(labels)))
    #     for i in range(num_classes):
    #         idx = np.where(labels == i)[0]
    #         idx = np.random.choice(idx, label_per_class, False)
    #         labeled_idx.extend(idx)
    #     labeled_idx = np.array(labeled_idx)
    #     assert len(labeled_idx) == num_labeled
    #     np.random.shuffle(labeled_idx)
    #     return labeled_idx, unlabeled_idx

    # train_labeled_idxs, train_unlabeled_idxs = x_u_split(args.num_labeled, args.num_classes, train_Y)
    # train_unlabeled_X = train_X[train_unlabeled_idxs]
    train_unlabeled_X = train_X
    # train_unlabeled_X_w, train_unlabeled_X_s = DataTransform(train_unlabeled_X)
    train_unlabeled_dataset = UnlabelMyDataset2(train_unlabeled_X)
    unlabeled_trainloader = DataLoader(train_unlabeled_dataset, batch_size=args.mu * args.batch_size,
                                       sampler=RandomSampler(train_unlabeled_dataset), drop_last=True)
    args.eval_step = len(unlabeled_trainloader)
    # train_labeled_x = np.load('./data/ISCX2012/LBS/train_labeled_5per0.npy')[:, :-1]
    # train_labeled_y = np.load('./data/ISCX2012/LBS/train_labeled_5per0.npy')[:, -1]
    # train_labeled_X = np.load('./data/CICIDS2017/LBS/train_labeled_5per1.npy')[:, :-1]
    # train_labeled_Y = np.load('./data/CICIDS2017/LBS/train_labeled_5per1.npy')[:, -1]
    # labeled_idx = np.arange(train_labeled_X.shape[0])
    # if args.expand_labels or args.num_labeled < args.batch_size:
    #     num_expand_x = math.ceil(
    #             args.batch_size * args.eval_step / args.num_labeled)
    #     labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    #     np.random.shuffle(labeled_idx)
    # train_labeled_X = train_labeled_X[labeled_idx]
    # train_labeled_Y = train_labeled_Y[labeled_idx]

    # import pdb;pdb.set_trace()
    # test_X = test_X / 255
    def x_u_split(labels):
        labeled_idx = np.arange(labels.shape[0])
        if args.expand_labels or args.num_labeled < args.batch_size:
            num_expand_x = math.ceil(
                args.batch_size * args.eval_step / labels.shape[0])
            labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
        np.random.shuffle(labeled_idx)
        return labeled_idx

    # labeled_idx = x_u_split(train_labeled_y)
    # train_labeled_X = train_labeled_x[labeled_idx]
    # train_labeled_Y = train_labeled_y[labeled_idx]
    train_labeled_X = train_X
    train_labeled_Y = train_Y
    test_dataset = MyDataset(test_X, test_Y)
    test_loader = DataLoader(test_dataset, batch_size=args.mu * args.batch_size)

    train_labeled_dataset = MyDataset(train_labeled_X, train_labeled_Y)
    labeled_trainloader = DataLoader(train_labeled_dataset, batch_size=args.batch_size,
                                     sampler=RandomSampler(train_labeled_dataset), drop_last=True)



    # if args.local_rank not in [-1, 0]:
    #     torch.distributed.barrier()

    # labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
    #     args, './data')

    # if args.local_rank == 0:
    #     torch.distributed.barrier()

    # train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    #
    # labeled_trainloader = DataLoader(
    #     labeled_dataset,
    #     sampler=train_sampler(labeled_dataset),
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     drop_last=True)
    #
    # unlabeled_trainloader = DataLoader(
    #     unlabeled_dataset,
    #     sampler=train_sampler(unlabeled_dataset),
    #     batch_size=args.batch_size*args.mu,
    #     num_workers=args.num_workers,
    #     drop_last=True)
    #
    # test_loader = DataLoader(
    #     test_dataset,
    #     sampler=SequentialSampler(test_dataset),
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers)

    # if args.local_rank not in [-1, 0]:
    #     torch.distributed.barrier()

    model = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    args.epochs = 300
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size*args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()
    if args.num_branch == 2:
        train_agu(args, labeled_trainloader, unlabeled_trainloader, test_loader,
              model, optimizer, ema_model, scheduler)
    elif args.num_branch == 3:
        train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
                  model, optimizer, ema_model, scheduler)
    else:
        train_sup(args, labeled_trainloader, test_loader,
                  model, optimizer, ema_model, scheduler)

def train_sup(args, labeled_trainloader, test_loader,
              model, optimizer, ema_model,scheduler):
    best_acc = 0
    test_accs = []
    Loss = []
    best_accs = []
    # batch_time = AverageMeter()
    # data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    labeled_iter = iter(labeled_trainloader)

    model.train()

    for epoch in range(args.start_epoch, args.epochs):
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])
        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x = labeled_iter.next()
            except:
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = labeled_iter.next()

            # data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            inputs_x = inputs_x.float().to(args.device)

            logits_x, _, _ = model(inputs_x,isLBS=True)
            # CE loss for labeled data
            targets_x = torch.tensor(targets_x, dtype=torch.long).to(args.device)
            # import pdb;  pdb.set_trace()
            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            loss = Lx
            # import pdb; pdb.set_trace()
            loss.backward()
            losses.update(loss.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time = time.time() - end
            # end = time.time()
            if not args.no_progress:
                # print(losses.avg,losses_x.avg,losses_m.avg)
                p_bar.set_description(
                    "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Time:{bt:.3f}. Loss: {loss:.4f}. ".format(
                        epoch=epoch + 1,
                        epochs=args.epochs,
                        batch=batch_idx + 1,
                        iter=args.eval_step,
                        lr=scheduler.get_lr()[0],
                        # data=data_time.avg,
                        bt=batch_time,
                        loss=losses.avg))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model  = model
        if args.local_rank in [-1, 0]:
            test_loss, test_acc = test(args, test_loader, test_model, epoch)

            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)
            Loss.append(losses.avg)
            test_accs.append(test_acc)
            is_best = test_acc > best_acc
            if is_best:
                counter = 0
                best_acc = test_acc
            else:
                counter += 1
                if counter > args.early_stop:
                    print('EarlyStopping!!!   Saving Best_acc is {}'.format(best_acc))
                    break
            best_accs.append(best_acc)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))

    if args.local_rank in [-1, 0]:
        args.writer.close()
    Loss0 = torch.tensor(Loss).cpu().numpy()
    BestACC = torch.tensor(best_accs).cpu().numpy()
    # np.savez('./loss/IDS2012_5shot_1branch', loss=Loss0, bestACC=BestACC)
    return best_acc

def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler):
    global best_acc
    test_accs = []
    best_accs = []
    Loss = []
    Loss_x = []
    Loss_m = []
    Loss_u =[]
    # batch_time = AverageMeter()
    # data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    losses_m = AverageMeter()
    mask_probs = AverageMeter()
    end = time.time()

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])
        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x = labeled_iter.next()
            except:
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = labeled_iter.next()
            try:
                inputs_u = unlabeled_iter.next()
                inputs_u1 = unlabeled_iter.next()
                # inputs_u_w, inputs_u_s = unlabeled_iter.next()
                # inputs_u_w1, inputs_u_s1 = unlabeled_iter.next()
            except:
                unlabeled_iter = iter(unlabeled_trainloader)
                inputs_u = unlabeled_iter.next()
                inputs_u1 = unlabeled_iter.next()
                # inputs_u_w, inputs_u_s = unlabeled_iter.next()
                # inputs_u_w1, inputs_u_s1 = unlabeled_iter.next()

            inputs_u_w, inputs_u_s =DataTransform(inputs_u,args.agu_level)
            inputs_u_w1, inputs_u_s1 =DataTransform(inputs_u1,args.agu_level)

            # try:
            #     (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
            #     (inputs_u_w1, inputs_u_s1), _ = unlabeled_iter.next()
            # except:
            #     unlabeled_iter = iter(unlabeled_trainloader)
            #     (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
            #     (inputs_u_w1, inputs_u_s1), _ = unlabeled_iter.next()

            # import pdb;pdb.set_trace()
            # data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]


            lam = np.random.beta(args.alpha, args.alpha)
            lam = max(lam, 1-lam)

            mix_uniput_w = lam * inputs_u_w + (1 - lam) * inputs_u_w1
            mix_uniput_s = lam * inputs_u_s + (1 - lam) * inputs_u_s1


            inputs = interleave(
                torch.cat((inputs_x, mix_uniput_w, mix_uniput_s)), 2*args.mu+1).to(args.device)
            targets_x = targets_x.to(args.device)
            logits, features, outs = model(inputs,isLBS=True)
            logits = de_interleave(logits, 2*args.mu+1)
            logits_x = logits[:batch_size]
            del logits
            outs = de_interleave(outs, 2*args.mu+1)
            outs_w, outs_s = outs[batch_size:].chunk(2)


            input1 = interleave(torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu+1).to(args.device)
            logits1, features1, outs1 = model(input1,isLBS=True)
            logits1 = de_interleave(logits1,2*args.mu+1)
            logits1_u_w, logits1_u_s = logits1[batch_size:].chunk(2)
            del logits1
            outs1 = de_interleave(outs1,2*args.mu+1)
            outs1_w, outs1_s = outs1[batch_size:].chunk(2)
            del outs1

            input2 = interleave(torch.cat((inputs_x, inputs_u_w1, inputs_u_s1)), 2*args.mu+1).to(args.device)
            logits2, features2, outs2 = model(input2,isLBS=True)
            del logits2
            outs2 = de_interleave(outs2, 2*args.mu+1)
            outs2_w, outs2_s = outs2[batch_size:].chunk(2)
            del outs2


            # positive sample pairs construction
            mix_out_w = outs1_w * lam + (1 - lam) * outs2_w
            mix_out_s = outs1_s * lam + (1 - lam) * outs2_s
            out_pos = torch.exp(torch.sum(outs_w * mix_out_w, dim=-1) / args.T)
            out_pos1 = torch.exp(torch.sum(outs_s * mix_out_s, dim=-1) / args.T)
            out_pos2 = torch.cat([out_pos, out_pos1], dim=0)
            out_neg = torch.cat([outs1_w, outs2_s], dim=0)
            sim_marix = torch.exp(torch.mm(out_neg, out_neg.t().contiguous() / args.T))
            mask = (torch.ones_like(sim_marix) - torch.eye(2 * args.mu* batch_size, device=sim_marix.device)).bool()
            sim_marix = sim_marix.masked_select(mask).view(2 * args.mu*  batch_size, -1)
            Lm = (- torch.log(out_pos2 / sim_marix.sum(dim=-1))).mean()


            # CE loss for labeled data
            targets_x = torch.tensor(targets_x, dtype=torch.long).to(args.device)
            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            # CE loss for unlabeled data
            pseudo_label = torch.softmax(logits1_u_w.detach()/args.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()
            Lu = (F.cross_entropy(logits1_u_s, targets_u,
                                  reduction='none') * mask).mean()

            #total loss
            loss = Lx + args.lambda_u * Lu + args.weight * Lm
            loss.backward()
            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            losses_m.update(Lm.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time=time.time() - end
            # end = time.time()
            mask_probs.update(mask.mean().item())
            if not args.no_progress:
                p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Time:{bt:.3f}. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Loss_m: {loss_m:.4f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_step,
                    lr=scheduler.get_lr()[0],
                    # data=data_time.avg,
                    bt=batch_time,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    loss_m=losses_m.avg,
                    # mask=mask_probs.avg
                ))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model
        # if args.local_rank in [-1, 0]:
        test_loss, test_acc = test(args, test_loader, test_model, epoch)

        args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
        args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
        args.writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
        args.writer.add_scalar('train/4.mask', mask_probs.avg, epoch)
        args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
        args.writer.add_scalar('test/2.test_loss', test_loss, epoch)
        Loss.append(losses.avg)
        Loss_x.append(losses_x.avg)
        Loss_m.append(losses_m.avg)
        Loss_u.append(losses_u.avg)
        test_accs.append(test_acc)

        is_best = test_acc > best_acc
        # best_acc = max(test_acc, best_acc)
        if is_best:
            counter = 0
            best_acc = test_acc
            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.out)
        else:
            counter += 1
            if counter > args.early_stop:
                print('EarlyStopping!!!   Saving Best_acc is {}'.format(best_acc))
                # return best_acc
                break
        best_accs.append(best_acc)
        logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
        logger.info('Mean top-1 acc: {:.2f}\n'.format(np.mean(test_accs[-20:])))
    if args.local_rank in [-1, 0]:
        args.writer.close()
    Loss0 = torch.tensor(Loss).cpu().numpy()
    Loss1 = torch.tensor(Loss_x).cpu().numpy()
    Loss2 = torch.tensor(Loss_m).cpu().numpy()
    Loss3 = torch.tensor(Loss_u).cpu().numpy()
    BestACC = torch.tensor(best_accs).cpu().numpy()
    # import pdb;pdb.set_trace()
    # os.makedirs(args.out + '/loss', exist_ok=True)
    # np.savez('./loss/IDS2012_5shot_3branch', loss=Loss0, lx=Loss1, lm=Loss2,lu=Loss3, bestACC=BestACC)

def train_agu(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler):
    global best_acc
    test_accs = []
    best_accs = []
    Loss = []
    Loss_x = []
    Loss_u = []
    # batch_time = AverageMeter()
    # data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    mask_probs = AverageMeter()
    end = time.time()

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])
        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x = labeled_iter.next()
            except:
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = labeled_iter.next()
            try:
                inputs_u = unlabeled_iter.next()
            except:
                unlabeled_iter = iter(unlabeled_trainloader)
                inputs_u = unlabeled_iter.next()

            inputs_u_w, inputs_u_s =DataTransform(inputs_u,args.agu_level)
            # inputs_u_w1, inputs_u_s1 =DataTransform(inputs_u1,args.agu_level)

            # data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]


            input1 = interleave(torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu+1).to(args.device)
            logits1, features1, outs1 = model(input1,isLBS=True)
            logits1 = de_interleave(logits1,2*args.mu+1)
            logits_x = logits1[:batch_size]
            logits1_u_w, logits1_u_s = logits1[batch_size:].chunk(2)
            del logits1
            # outs1 = de_interleave(outs1,2*args.mu+1)
            # outs1_w, outs1_s = outs1[batch_size:].chunk(2)
            del outs1

            # CE loss for labeled data
            targets_x = torch.tensor(targets_x, dtype=torch.long).to(args.device)
            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            # CE loss for unlabeled data
            pseudo_label = torch.softmax(logits1_u_w.detach()/args.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()
            Lu = (F.cross_entropy(logits1_u_s, targets_u,
                                  reduction='none') * mask).mean()

            #total loss
            loss = Lx + args.lambda_u * Lu
            loss.backward()
            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time=time.time() - end
            # end = time.time()
            mask_probs.update(mask.mean().item())
            if not args.no_progress:
                p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Time:{bt:.3f}. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_step,
                    lr=scheduler.get_lr()[0],
                    # data=data_time.avg,
                    bt=batch_time,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    # loss_m=losses_m.avg,
                    # mask=mask_probs.avg
                ))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model
        if args.local_rank in [-1, 0]:
            test_loss, test_acc = test(args, test_loader, test_model, epoch)

            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
            args.writer.add_scalar('train/4.mask', mask_probs.avg, epoch)
            args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)
            Loss.append(losses.avg)
            Loss_x.append(losses_x.avg)
            Loss_u.append(losses_u.avg)
            test_accs.append(test_acc)

            is_best = test_acc > best_acc
            # best_acc = max(test_acc, best_acc)
            if is_best:
                counter = 0
                best_acc = test_acc
                model_to_save = model.module if hasattr(model, "module") else model
                if args.use_ema:
                    ema_to_save = ema_model.ema.module if hasattr(
                        ema_model.ema, "module") else ema_model.ema
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model_to_save.state_dict(),
                    'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, is_best, args.out)
            else:
                counter += 1
                if counter > args.early_stop:
                    print('EarlyStopping!!!   Saving Best_acc is {}'.format(best_acc))
                    break

            best_accs.append(best_acc)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))

    if args.local_rank in [-1, 0]:
        args.writer.close()
    Loss0 = torch.tensor(Loss).cpu().numpy()
    Loss1 = torch.tensor(Loss_x).cpu().numpy()
    Loss3 = torch.tensor(Loss_u).cpu().numpy()
    BestACC = torch.tensor(best_accs).cpu().numpy()
    # np.savez('./loss/IDS2012_5shot_2branch', loss=Loss0, lx=Loss1, lu=Loss3, bestACC=BestACC)

def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.float().to(args.device)
            targets = targets.long().to(args.device)
            outputs, fea, outs = model(inputs,isLBS=True)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 3))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    # top5=top5.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    # logger.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg


if __name__ == '__main__':
    main()
