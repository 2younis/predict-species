import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.parallel.data_parallel import DataParallel
import torch.optim as optim
from datetime import timedelta
from time import time
import torch.nn.functional as F
from utils.parameters import *
from utils.meter import AverageMeter, ProgressMeter, accuracy
import model.resnet as network
import hrb_input

device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

TIME = time()


def train(model, optimizer, loss_func, train_loader, epoch):
    losses = AverageMeter('Loss', ':4.4f')
    top1 = AverageMeter('Acc@1', ':4.2f')
    top5 = AverageMeter('Acc@5', ':4.2f')
    progress = ProgressMeter(len(train_loader), [losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch + 1))

    model.train()

    for i, data in enumerate(train_loader, 1):

        imgs = data['image'].to(device)
        label = data['label'].to(device)

        out = model(imgs)
        loss = loss_func(out, label)

        acc1, acc5 = accuracy(out, label, topk=(1, 5))
        losses.update(loss.item())
        top1.update(acc1[0])
        top5.update(acc5[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % PRINT_FREQ == 0:
            progress.display(i)

    timer = time() - TIME
    print("Total time Elapsed (H:m:s):", timedelta(seconds=timer))


def validate(model, loss_func, val_loader):
    losses = AverageMeter('Loss', ':4.4f')
    top1 = AverageMeter('Acc@1', ':4.2f')
    top5 = AverageMeter('Acc@5', ':4.2f')
    progress = ProgressMeter(len(val_loader), [losses, top1, top5],
                             prefix='Validation: ')

    model.eval()

    with torch.no_grad():
        for i, data in enumerate(val_loader, 1):
            imgs = data['image'].to(device)
            label = data['label'].to(device)

            out = model(imgs)
            loss = loss_func(out, label)

            acc1, acc5 = accuracy(out, label, topk=(1, 5))
            losses.update(loss.item())
            top1.update(acc1[0])
            top5.update(acc5[0])

        progress.display(i)

    return losses.avg


def predict(test_loader):
    losses = AverageMeter('Loss', ':4.4f')
    top1 = AverageMeter('Acc@1', ':4.2f')
    top5 = AverageMeter('Acc@5', ':4.2f')
    progress = ProgressMeter(len(test_loader), [losses, top1, top5],
                             prefix='Test: ')

    model.eval()

    with torch.no_grad():
        for i, data in enumerate(test_loader, 1):
            imgs = data['image'].to(device)
            label = data['label'].to(device)

            out = model(imgs)
            loss = loss_func(out, label)

            acc1, acc5 = accuracy(out, label, topk=(1, 5))
            losses.update(loss.item())
            top1.update(acc1[0])
            top5.update(acc5[0])

        progress.display(i)


def main():
    train_dataset = hrb_input.TrainDataset()
    val_dataset = hrb_input.ValidationDataset()
    test_dataset = hrb_input.TestDataset()

    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=8,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=8,
                            pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=BATCH_SIZE,
                             num_workers=8,
                             pin_memory=True)

    model = network.resnet50(num_classes=1000)
    model = DataParallel(model, device_ids=[0, 1, 2])
    model.to(device)

    loss_func = F.cross_entropy
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=LR_STEPS,
                                               gamma=LR_GAMMA)

    best_loss = 100.0
    patience_counter = 0

    for epoch in range(EPOCHS):

        train(model, optimizer, loss_func, train_loader, epoch)
        val_loss = validate(model, loss_func, val_loader)

        if val_loss < best_loss:
            torch.save(model, MODEL_PATH)
            print('Saving improved model')
            print()
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print('Epoch(s) since best model: ', patience_counter)
            print()
        if patience_counter >= EARLY_STOPPING_EPOCHS:
            print('Early Stopping ...')
            print()
            break
        scheduler.step()

    print('Predicting labels from best trained model')
    predict(test_loader)


if __name__ == '__main__':
    main()
