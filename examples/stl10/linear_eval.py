import time
import argparse

import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

from util import AverageMeter
from encoder import SmallAlexNet


def parse_option():
    parser = argparse.ArgumentParser('STL-10 Representation Learning with Alignment and Uniformity Losses')

    parser.add_argument('encoder_checkpoint', type=str, help='Encoder checkpoint to evaluate')
    parser.add_argument('--feat_dim', type=int, default=128, help='Encoder feature dimensionality')
    parser.add_argument('--layer_index', type=int, default=-2, help='Evaluation layer')

    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='Learning rate decay rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,80', help='When to decay learning rate')

    parser.add_argument('--num_workers', type=int, default=6, help='Number of data loader workers to use')
    parser.add_argument('--log_interval', type=int, default=40, help='Number of iterations between logs')
    parser.add_argument('--gpu', type=int, default='0', help='One GPU to use')

    parser.add_argument('--data_folder', type=str, default='./data', help='Path to data')

    opt = parser.parse_args()

    if opt.lr is None:
        opt.lr = 0.12 * (opt.batch_size / 256)

    opt.gpu = torch.device('cuda', opt.gpu)
    opt.lr_decay_epochs = list(map(int, opt.lr_decay_epochs.split(',')))

    return opt


def get_data_loaders(opt):
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(64, scale=(0.08, 1)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.44087801806139126, 0.42790631331699347, 0.3867879370752931),
            (0.26826768628079806, 0.2610450402318512, 0.26866836876860795),
        ),
    ])
    val_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(70),
        torchvision.transforms.CenterCrop(64),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.44087801806139126, 0.42790631331699347, 0.3867879370752931),
            (0.26826768628079806, 0.2610450402318512, 0.26866836876860795),
        ),
    ])
    train_dataset = torchvision.datasets.STL10(opt.data_folder, 'train', download=True, transform=train_transform)
    val_dataset = torchvision.datasets.STL10(opt.data_folder, 'test', transform=val_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size,
                                               num_workers=opt.num_workers, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size,
                                             num_workers=opt.num_workers, pin_memory=True)
    return train_loader, val_loader


def validate(opt, encoder, classifier, val_loader):
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            pred = classifier(encoder(images.to(opt.gpu), layer_index=opt.layer_index).flatten(1)).argmax(dim=1)
            correct += (pred.cpu() == labels).sum().item()
    return correct / len(val_loader.dataset)


def main():
    opt = parse_option()

    torch.cuda.set_device(opt.gpu)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    encoder = SmallAlexNet(feat_dim=opt.feat_dim).to(opt.gpu)
    encoder.eval()
    train_loader, val_loader = get_data_loaders(opt)

    with torch.no_grad():
        sample, _ = train_loader.dataset[0]
        eval_numel = encoder(sample.unsqueeze(0).to(opt.gpu), layer_index=opt.layer_index).numel()
    print(f'Feature dimension: {eval_numel}')

    encoder.load_state_dict(torch.load(opt.encoder_checkpoint, map_location=opt.gpu))
    print(f'Loaded checkpoint from {opt.encoder_checkpoint}')

    classifier = nn.Linear(eval_numel, 10).to(opt.gpu)

    optim = torch.optim.Adam(classifier.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, gamma=opt.lr_decay_rate,
                                                     milestones=opt.lr_decay_epochs)

    loss_meter = AverageMeter('loss')
    it_time_meter = AverageMeter('iter_time')
    for epoch in range(opt.epochs):
        loss_meter.reset()
        it_time_meter.reset()
        t0 = time.time()
        for ii, (images, labels) in enumerate(train_loader):
            optim.zero_grad()
            with torch.no_grad():
                feats = encoder(images.to(opt.gpu), layer_index=opt.layer_index).flatten(1)
            logits = classifier(feats)
            loss = F.cross_entropy(logits, labels.to(opt.gpu))
            loss_meter.update(loss, images.shape[0])
            loss.backward()
            optim.step()
            it_time_meter.update(time.time() - t0)
            if ii % opt.log_interval == 0:
                print(f"Epoch {epoch}/{opt.epochs}\tIt {ii}/{len(train_loader)}\t{loss_meter}\t{it_time_meter}")
            t0 = time.time()
        scheduler.step()
        val_acc = validate(opt, encoder, classifier, val_loader)
        print(f"Epoch {epoch}/{opt.epochs}\tval_acc {val_acc*100:.4g}%")


if __name__ == '__main__':
    main()
