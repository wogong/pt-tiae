import os
import argparse
import json
from datetime import datetime
import numpy as np
from utils import (save_roc_pr_curve_data,
                   get_class_name_from_index,
                   get_channels_axis,
                   save_model,
                   init_weights,
                   denormalize_minus1_1,
                   load_cifar10,
                   load_mnist)
from outlier_datasets import (load_cifar10_with_outliers,
                              load_cifar100_with_outliers,
                              load_fashion_mnist_with_outliers,
                              load_mnist_with_outliers,
                              load_svhn_with_outliers)
from transformations import Transform
from models.tiae import TIAE
import torchvision.transforms as transforms
import torchvision
from keras2pytorch_dataset import trainset_pytorch_tiae
import torch.utils.data as data
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from misc import AverageMeter

from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter
from mailgun import send_mailgun

RESULTS_DIR = '/home/wogong/models/pt-tiae/work' + datetime.now().strftime('%Y-%m-%d-%H%M%S')
MODEL_DIR = RESULTS_DIR + '/tensorboard'
logger = SummaryWriter(MODEL_DIR)

cifar_mean = [0.49139968, 0.48215827, 0.44653124]
cifar_std = [0.24703233, 0.24348505, 0.26158768]
cifar_mean_gray = [0.4753234924]
cifar_std_gray = [0.2475349119]

transform_train = transforms.Compose([transforms.ToTensor(), ])
transform_test = transforms.Compose([transforms.ToTensor(), ])


def calc_weight(losses, device, process, spp):
    """
    calculate sample weight from epoch loss and sample id.
    Args:
        losses: torch.Tensor(sample_size), sample loss of last epoch
        device: cpu or gpu
        process: current process of training

    Returns:
        sample_weight: torch.Tensor(sample_size), newly calculated sample weight
    """
    mean = torch.mean(losses)
    std = torch.std(losses)
    lambda_ = mean - (spp - process) * std # large spp means few selected examples
    #lambda_ = np.percentile(losses.cpu(), 1 + 29*process)

    sample_weight = torch.where(losses < lambda_, torch.tensor(1.0).to(device), torch.tensor(0.0).to(device))
    selected_number = torch.sum(sample_weight).item()

    print('mean: {:.2f}, std: {:.2f}, lambda: {:.2f}, selected: {:.0f}'.format(mean, std, lambda_, selected_number))
    return sample_weight


def train(trainloader, model, criterion, class_name, testloader, y_train, device, args):
    """
    model train function.
    Args:
        trainloader:
        model:
        criterion:
        class_name:
        testloader:
        y_train: numpy array, sample normal/abnormal labels, [1 1 1 1 0 0] like, original sample size.
        device: cpu or gpu:0/1/...

    Returns:
        None
    """
    losses = AverageMeter()
    global_step = 0
    sample_weight = torch.ones(np.size(y_train)).to(device)

    for epoch in range(args.epochs):
        model.train()
        losses_epoch = torch.zeros(np.size(y_train)).to(device)
        losses_epoch_sum = torch.zeros(np.size(y_train)).to(device)
        losses_epoch_transform = torch.zeros(np.size(y_train) * 4).to(device)
        lr = 0.1 / pow(2, np.floor(epoch / args.lr_schedule))
        optimizer = optim.SGD(model.parameters(), lr=lr)
        # optimizer = optim.Adam(model.parameters(), eps=1e-7, weight_decay=0.0005)
        logger.add_scalar(class_name + "/lr", lr, global_step)
        for batch_idx, (inputs, inputs_transformed, labels, ids, tids) in enumerate(trainloader):

            inputs = inputs.to(device)
            inputs_transformed = inputs_transformed.to(device)
            ids = ids.to(device)
            tids = tids.to(device)

            outputs = model(inputs_transformed)

            loss = criterion(inputs, outputs)
            # criterion_validate = nn.MSELoss()
            # loss_validate = criterion_validate(inputs, outputs)

            loss_reduce = torch.mean(loss, (1, 2, 3))  # (batch_size, )

            indices = (ids, )
            indices_transform = (ids * 4 + tids, )
            value = loss_reduce.data.detach()
            losses_epoch.index_put_(indices, value, accumulate=True)
            losses_epoch_sum = losses_epoch_sum + losses_epoch
            losses_epoch_transform.index_put_(indices_transform, value, accumulate=True)

            weight_batch = torch.index_select(sample_weight, 0, ids)
            selected_num_batch = torch.sum(weight_batch)
            if selected_num_batch == 0:
                # no selected sample in this batch
                pass
            else:
                weighted_loss = torch.dot(loss_reduce, weight_batch) / selected_num_batch
                losses.update(weighted_loss.item(), selected_num_batch)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                weighted_loss.backward()
                optimizer.step()

            global_step = global_step + 1

            if (batch_idx + 1) % 200 == 0:
                print('Epoch: [{} | {}], batch: {}, loss: {:.4f}'.format(epoch + 1, args.epochs, batch_idx + 1, losses.avg))
                logger.add_scalar(class_name + '/training loss', losses.avg, global_step)

        # calculate weight
        if args.weight_flag == 1 and epoch > args.pretrain - 1:
            sample_weight = calc_weight(losses_epoch_sum, device, epoch / args.epochs, args.spp)

        logger.add_scalar(class_name + '/weight_sum', torch.sum(sample_weight).item(), epoch)

        # log images
        # if (epoch + 1) % 2 == 0:
        #     inputs_img = torchvision.utils.make_grid(inputs)
        #     inputs_transformed_img = torchvision.utils.make_grid(inputs_transformed)
        #     outputs_img = torchvision.utils.make_grid(outputs)
        #     merge_img = torchvision.utils.make_grid(
        #         [inputs_img, inputs_transformed_img.expand(3, -1, -1), outputs_img],
        #         nrow=1)
        #     merge_img_de = denormalize_minus1_1(merge_img.cpu().detach().numpy())
        #     plt.imshow(np.transpose(merge_img_de, (1, 2, 0)), interpolation='nearest')
        #     plt.axis('off')
        #     os.makedirs(os.path.join(MODEL_DIR, class_name), exist_ok=True)
        #     plt.savefig(os.path.join(MODEL_DIR, class_name, 'train_epoch_' + str(epoch) + '.png'))
        #     # logger.add_image('train/origin_trans_recon', merge_img_de, epoch)

        # testing while training
        if (epoch + 1) % 2 == 0:
            losses_copied = test(testloader, model, class_name, device, epoch)
            dataset_length = len(losses_copied)

            loss_grouped = np.array(np.split(losses_copied, dataset_length / 4))
            loss_grouped_mean = np.mean(loss_grouped, 0)
            loss_grouped_normalized = loss_grouped / loss_grouped_mean

            losses_result = np.array([np.mean(x, 0) for x in loss_grouped_normalized])

            losses_result = losses_result - losses_result.min()
            losses_result = losses_result / (1e-8 + losses_result.max())
            scores = 1 - losses_result

            auc_roc = roc_auc_score(y_train, scores)
            print('Epoch: [{} | {}], auc_roc: {:.4f}'.format(epoch + 1, args.epochs, auc_roc))
            logger.add_scalar(class_name + '/auc_roc', auc_roc, epoch)


def test(testloader, model, class_name, device, epoch=999):
    model.eval()
    losses = []
    for batch_idx, (inputs, inputs_transformed, labels, ids, tids) in enumerate(testloader):

        inputs = inputs.to(device)
        inputs_transformed = inputs_transformed.to(device)

        outputs = model(inputs_transformed)

        loss = outputs.sub(inputs).abs().view(outputs.size(0), -1)
        loss = loss.sum(dim=1, keepdim=False)
        losses.append(loss.data.cpu())

        # log images
        # if (batch_idx + 1) % 50 == 0 and epoch == 999:
        #     inputs_img = torchvision.utils.make_grid(inputs[:32, :, :, :])
        #     inputs_transformed_img = torchvision.utils.make_grid(inputs_transformed[:32, :, :, :])
        #     outputs_img = torchvision.utils.make_grid(outputs[:32, :, :, :])
        #     merge_img = torchvision.utils.make_grid(
        #         [inputs_img, inputs_transformed_img.expand(3, -1, -1), outputs_img],
        #         nrow=1)
        #     merge_img_de = denormalize_minus1_1(merge_img.cpu().detach().numpy())
        #     plt.imshow(np.transpose(merge_img_de, (1, 2, 0)), interpolation='nearest')
        #     plt.axis('off')
        #     os.makedirs(os.path.join(MODEL_DIR, class_name), exist_ok=True)
        #     plt.savefig(os.path.join(MODEL_DIR, class_name,
        #                              'test_epoch_' + str(epoch) + '_batch_' + str(batch_idx) + '.png'))

    losses = torch.cat(losses, dim=0)
    return losses.numpy()


def tiae_experiment(x_train, y_train, dataset_name, single_class_ind, restore, args):
    device = torch.device("cuda:" + args.gpu_id if torch.cuda.is_available() else "cpu")
    if dataset_name == 'mnist' or dataset_name == 'fashion-mnist':
        print("Not using data augmentation")
        data_aug = False
    else:
        print("Using data augmentation")
        data_aug = True

    n_channels = x_train.shape[get_channels_axis()]

    transformer = Transform()
    transform_num = transformer.n_transforms
    print("transform number is {}".format(transformer.n_transforms))

    class_name = get_class_name_from_index(single_class_ind, dataset_name)

    x_train_task = x_train
    y_train_task = y_train
    transformations_inds = np.tile(np.arange(transformer.n_transforms), len(x_train_task))
    x_train_task_transformed = transformer.transform_batch(np.repeat(x_train_task, transformer.n_transforms, axis=0),
                                                           transformations_inds)
    x_train_task_copied = np.repeat(x_train_task, transformer.n_transforms, axis=0)
    y_train_task_copied = np.repeat(y_train_task, transformer.n_transforms, axis=0)

    model = TIAE(n_channels=n_channels)
    model = model.to(device)
    init_weights(model, init_type='xavier', init_gain=0.02)

    trainset = trainset_pytorch_tiae(train_data=x_train_task_copied,
                                     train_data_transformed=x_train_task_transformed,
                                     train_labels=y_train_task_copied,
                                     data_aug=data_aug,
                                     transform=transform_train)
    testset = trainset_pytorch_tiae(train_data=x_train_task_copied,
                                    train_data_transformed=x_train_task_transformed,
                                    train_labels=y_train_task_copied,
                                    transform=transform_train)

    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    cudnn.benchmark = True
    criterion = nn.MSELoss(reduction='none')

    if args.epochs == 0:
        args.epochs = int(np.ceil(800 / transform_num))
    if args.lr_schedule ==0:
        args.lr_schedule = np.floor(50 / transform_num)

    # #########################Training########################
    if not restore:
        train(trainloader, model, criterion, class_name, testloader, y_train, device, args)
        model_file_name = '{}_tiae-{}_{}_{}.model.npz'.format(dataset_name, args.ratio,
                                                              get_class_name_from_index(single_class_ind, dataset_name),
                                                              datetime.now().strftime('%Y-%m-%d-%H%M'))
        model_path = os.path.join(RESULTS_DIR, dataset_name)
        save_model(model, model_path, model_file_name)
    else:
        print("restore model from: {}".format(restore))
        model.load_state_dict(torch.load(restore))

    # testing

    losses_copied = test(testloader, model, class_name, device)
    dataset_length = len(losses_copied)
    loss_grouped = np.array(np.split(losses_copied, dataset_length / transformer.n_transforms))

    # loss_grouped_mean = np.mean(loss_grouped, 0)
    # loss_grouped_normalized = loss_grouped / loss_grouped_mean

    # losses = np.array([np.mean(x, 0) for x in loss_grouped_normalized])
    losses = np.array([np.mean(x, 0) for x in loss_grouped])

    losses = losses - losses.min()
    losses = losses / (1e-8 + losses.max())
    scores = 1 - losses

    res_file_name = '{}_tiae-{}_{}_{}.npz'.format(dataset_name, args.ratio,
                                                  get_class_name_from_index(single_class_ind, dataset_name),
                                                  datetime.now().strftime('%Y-%m-%d-%H%M'))
    res_file_path = os.path.join(RESULTS_DIR, dataset_name, res_file_name)
    os.makedirs(os.path.join(RESULTS_DIR, dataset_name), exist_ok=True)
    save_roc_pr_curve_data(scores, y_train, res_file_path)


def run_experiments(args, load_dataset_fn, class_num, run_idx):
    max_sample_num = 12000
    os.makedirs(os.path.join(RESULTS_DIR, dataset_name), exist_ok=True)

    for c in range(class_num):
        # for c in range(0, 1):
        np.random.seed(run_idx)
        x_train, y_train = load_dataset_fn(c, args.ratio)

        # random sampling if the number of data is too large
        if x_train.shape[0] > max_sample_num:
            selected = np.random.choice(x_train.shape[0], max_sample_num, replace=False)
            x_train = x_train[selected, :]
            y_train = y_train[selected]

        restore = None
        tiae_experiment(x_train, y_train, dataset_name, c, restore, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TIAE experiment parameters.')
    parser.add_argument('--run_times', type=int, default=1, help='how many run times, default 1 time.')
    parser.add_argument('--gpu_id', type=str, default='0', help='which gpu to use, default id 0.')
    parser.add_argument('--dataset', type=str, required=True, help='which dataset used.')
    parser.add_argument('--weight_flag', type=int, default=1, help='turn self-paced learning on/off, default on.')
    parser.add_argument('--pretrain', type=int, default=0, help='how many pretrain steps before self-paced learning.')
    parser.add_argument('--ratio', type=float, required=True, help='outlier ratio used.')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size.')
    parser.add_argument('--spp', type=float, default=1.0, help='self-paced learning parameter.')
    parser.add_argument('--epochs', type=int, default=0, help='training epochs, default 800/K.')
    parser.add_argument('--lr_schedule', type=float, default=0, help='learning rate schedule parameter, default 80/K.')

    args = parser.parse_args()
    print(args)

    experiments_dict = {
        'mnist': (load_mnist_with_outliers, 'mnist', 10),
        'fashion-mnist': (load_fashion_mnist_with_outliers, 'fashion-mnist', 10),
        'cifar10': (load_cifar10_with_outliers, 'cifar10', 10),
        'cifar100': (load_cifar100_with_outliers, 'cifar100', 20),
        'svhn': (load_svhn_with_outliers, 'svhn', 10)
    }

    with open(RESULTS_DIR+'/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    n_run = args.run_times
    data_load_fn, dataset_name, n_classes = experiments_dict[args.dataset]

    for i in range(n_run):
        run_experiments(args, data_load_fn, n_classes, i)
    #send_mailgun()
