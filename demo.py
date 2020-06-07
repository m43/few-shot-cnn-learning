import random

from data_loader.data_loaders import OmniglotDataLoaderCreator
from model.model import CnnKoch2015
from trainer import *
from utils.util import top_k_acc, accuracy, accuracy_oneshot

if __name__ == '__main__':
    random.seed(72)
    np.random.seed(72)
    torch.manual_seed(72)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # export CUDA_VISIBLE_DEVICES=1

    print(torch.cuda.device_count())
    device = torch.device("cuda") if torch.cuda.device_count() else torch.device("cpu")
    device_ids = list(range(torch.cuda.device_count()))
    print(device_ids)
    print("Using device", device)

    omniglot_dataloader_creator = OmniglotDataLoaderCreator("./data/", 30000, 10000, 320, 400, False, 8)
    train_loader = omniglot_dataloader_creator.load_train(128, True)
    val_loader = omniglot_dataloader_creator.load_validation(False, 128)
    val_oneshot_loader = omniglot_dataloader_creator.load_validation(True, 2)
    test_oneshot_loader = omniglot_dataloader_creator.load_test(2)

    # train_loader = omniglot_dataloader_creator.load_train(16, False)
    # val_loader = omniglot_dataloader_creator.load_validation(False, 128, False)
    # val_oneshot_loader = omniglot_dataloader_creator.load_validation(True, 2, False)
    # test_oneshot_loader = omniglot_dataloader_creator.load_test(2, False)

    koch2015 = CnnKoch2015()
    learning_rate = 0.00006
    epochs = 500
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.Adam(koch2015.parameters(), lr=learning_rate)
    metric_ftns = [accuracy]
    metric_ftns_oneshot = [accuracy_oneshot, top_k_acc]
    save_folder = "./saved/"

    trainer = OmniglotTrainer(koch2015, criterion, metric_ftns, metric_ftns_oneshot, optimizer, device, [], epochs,
                              save_folder, "max val accuracy_oneshot", train_loader, val_loader, val_oneshot_loader,
                              test_oneshot_loader)

    koch2015.summary(device.type)
    trainer.train()
