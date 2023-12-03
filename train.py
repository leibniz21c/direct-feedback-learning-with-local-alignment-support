import os
import argparse
import signal

import torch
import torch.nn as nn
from torch import optim

import models
from utils import get_loaders, get_logger, get_device, logging_vars, get_lr_decays


# Global constants
_trainable = True

# SIGINT(_trainable) signal handler
def sigint_handler(signum, frame) -> None:
    global _trainable
    _trainable = False


def main(args):
    # GLobal constants
    global _trainable

    # Arguments
    result_path = args.result_path
    num_workers = args.num_workers
    seed = args.seed
    dataset = args.dataset
    batch_size = args.batch_size
    optimizer = args.optimizer
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    weight_decay = args.weight_decay
    learning = args.learning
    architecture = args.architecture
    
    # Logging
    logger = get_logger(result_path, exist_ok=False)

    # Devices
    device = get_device(seed)

    # Model, TODO : refactoring
    model = getattr(
        models, 
        f"{learning.upper()}_{architecture.upper()}_{dataset}"
    )().to(device)

    # Losses
    if learning == 'olas':
        local_criteria = nn.MSELoss()
    elif learning == 'llas':
        local_criteria = nn.CrossEntropyLoss()
    else:
        local_criteria = None
    criteria = nn.CrossEntropyLoss()

    # Optimizers
    optimizer = getattr(optim, optimizer)(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # learning rate scheduler
    lr_decays = get_lr_decays(num_epochs)

    # loaders
    train_loader, test_loader = get_loaders(
        dataset=dataset,
        root=f"{os.path.expanduser('~')}/data/image-classification", 
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )

    # Logging
    logging_vars(
        logger,
        result_path=result_path,
        num_workers=num_workers,
        seed=seed,
        dataset=dataset,
        batch_size=batch_size,
        optimizer=optimizer,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        weight_decay=weight_decay,
        learning=learning,
        architecture=architecture,
    )

    # ***************************** Experiment ***************************** #
    model.eval()
    losses, predicts, labels = [], [], []
    with torch.no_grad():
        for data, label in test_loader:
            if architecture == 'fc':
                data, label = data.to(device).flatten(start_dim=1), label.to(device)
            else:
                data, label = data.to(device), label.to(device)

            # Get loss
            output = model(data)
            loss = criteria(output, label)

            # log
            losses.append(loss.item())
            predicts.append(torch.argmax(output, dim=1))
            labels.append(label)

    # Signal setting
    signal.signal(signal.SIGINT, sigint_handler)

    # Logging
    pilot_test_losses = torch.tensor(losses).mean().item()
    pilot_test_acc = (sum(torch.cat(predicts) == torch.cat(labels)) / len(test_loader.dataset)).item()

    logger.info(f"[INIT_TEST] test_loss = {pilot_test_losses:.4}, test_acc = {pilot_test_acc:.4}")
    
    best_train_acc, best_test_acc = 0.0, 0.0
    train_losses, train_acc, test_losses, test_acc = [], [], [], [] 


    for epoch in range(args.num_epochs):
        # lr schedules
        if epoch in lr_decays:
            optimizer.param_groups[0]["lr"] *= 0.25
    
        model.train()
        global_losses, global_predicts, labels = [], [], []
        # Training loop
        for data, label in train_loader:
            if architecture == 'fc':
                data, label = data.to(device).flatten(start_dim=1), label.to(device)
            else:
                data, label = data.to(device), label.to(device)
                
            if 'las' in learning:
                result = model.fit(data, label, (local_criteria, criteria), optimizer)
            else:
                result = model.fit(data, label, criteria, optimizer)
        
            global_losses.append(result['global_loss'].item())
            global_predicts.append(torch.argmax(result['output'], dim=1))
            labels.append(label)

        # Logging
        train_loss = torch.tensor(global_losses).mean()
        train_losses.append(train_loss.item())
        print()

        train_acc.append((sum(torch.cat(global_predicts) == torch.cat(labels)) / len(train_loader.dataset)).item())

        # Test loop
        model.eval()
        global_losses, global_predicts, labels = [], [], []
        with torch.no_grad():
            for data, label in test_loader:
                if architecture == 'fc':
                    data, label = data.to(device).flatten(start_dim=1), label.to(device)
                else:
                    data, label = data.to(device), label.to(device)

                output = model(data)

                global_loss = criteria(output, label)
                global_predict = torch.argmax(output, dim=1)

                # log
                global_losses.append(global_loss.item())
                global_predicts.append(global_predict)
                labels.append(label)

        # Logging
        test_loss = torch.tensor(global_losses).mean()
        test_losses.append(test_loss.item())
        test_acc.append((sum(torch.cat(global_predicts) == torch.cat(labels)) / len(test_loader.dataset)).item())
        
        logger.info(
            f"[EPOCH {epoch + 1:02}] : train_acc = {train_acc[-1]:.4}, train_loss = {train_losses[-1]:.4}, test_acc = {test_acc[-1]:.4}, test_loss = {test_losses[-1]:.4}"
        )
        
        if test_acc[-1] > best_test_acc:
            best_test_acc = test_acc[-1]
            best_train_acc = train_acc[-1]

        # is it trainable model?
        if epoch == 5 and best_test_acc < pilot_test_acc*1.15: _trainable = False
        if torch.isnan(train_loss) or torch.isnan(test_loss): _trainable = False

        if not _trainable: break

    # end of epochs
    logger.info(f"Experiment end with best test accuracy : {best_test_acc:.4}.")
    torch.save(torch.tensor(train_acc), f"{result_path}/train_acc.pt")
    torch.save(torch.tensor(train_losses), f"{result_path}/train_losses.pt")
    torch.save(torch.tensor(test_acc), f"{result_path}/test_acc.pt")
    torch.save(torch.tensor(test_losses), f"{result_path}/test_losses.pt")
    torch.save(torch.tensor(best_train_acc), f"{result_path}/best_train_acc.pt")
    torch.save(torch.tensor(best_test_acc), f"{result_path}/best_test_acc.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)

    # Shell configs
    parser.add_argument("--result_path", type=str, default="result")

    # Device configs
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)

    # Training configs
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--optimizer", type=str, default='SGD')
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.)

    # Learning method configs
    parser.add_argument("--learning", type=str, default='bp')
    parser.add_argument("--architecture", type=str, default='fc')

    args = parser.parse_args()
    main(args)
