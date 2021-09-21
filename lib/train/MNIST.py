import time

import torch


def train(data_loader, model, cfg):
    # avg_meters =
    model.train()
    end = time.time()
    for iter, batch in enumerate(data_loader):
        data_time = time.time() - end
        iter += 1

        outputs = model(batch)


if __name__=='__main__':
    
    train()