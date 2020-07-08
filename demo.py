import random

from data_loader.data_loaders import OmniglotDataLoaderCreator
from model import accuracy, accuracy_oneshot, top_k_acc
from model.model import CnnKoch2015, CnnKoch2015BatchNorm
from trainer import *

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

    ## DATASET/DATALOADER PARAMETERS
    train_samples = 150000
    train_batch_size = 128
    train_shuffle = True
    validation_samples = 10000
    do_affine_transformations = True

    num_workers = 12
    cache_path = "/mnt/sdb1/datasets/omniglot"
    # cache_path = "./temp"
    already_cached = True

    omniglot_dataloader_creator = OmniglotDataLoaderCreator(
        "./data/", train_samples, validation_samples, 320, 400, do_affine_transformations, 8)

    train_loader = omniglot_dataloader_creator.load_train(train_batch_size, train_shuffle, num_workers, cache_path, already_cached)
    train_oneshot_loader = omniglot_dataloader_creator._load_train_oneshot()
    val_loader = omniglot_dataloader_creator.load_validation(False, 128, False, num_workers)
    val_oneshot_loader = omniglot_dataloader_creator.load_validation(True, 2, False, num_workers)
    test_oneshot_loader = omniglot_dataloader_creator.load_test(2, False, num_workers)

    # train_loader = omniglot_dataloader_creator.load_train(16, False)
    # val_loader = omniglot_dataloader_creator.load_validation(False, 128, False)
    # val_oneshot_loader = omniglot_dataloader_creator.load_validation(True, 2, False)
    # test_oneshot_loader = omniglot_dataloader_creator.load_test(2, False)

    ## TRAIN PARAMETERS
    koch2015 = CnnKoch2015()
    # koch2015 = CnnKoch2015BatchNorm()
    learning_rate = 1e-4  # 0.00006
    weight_decay = 1e-3
    epochs = 500
    early_stopping = 60
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.Adam(koch2015.parameters(), lr=learning_rate, weight_decay=weight_decay)
    metric_ftns = [accuracy]
    metric_ftns_oneshot = [accuracy_oneshot, top_k_acc]
    save_folder = "./saved/"

    run_name = f"{train_samples // 1000}k{'+af' if do_affine_transformations else ''} lr:{learning_rate} wd:{weight_decay} norm img"
    trainer = OmniglotTrainer(run_name, koch2015, criterion, metric_ftns,
                              metric_ftns_oneshot, optimizer, device, [], epochs,
                              save_folder, "max val accuracy_oneshot", early_stopping, train_loader, val_loader,
                              val_oneshot_loader, test_oneshot_loader, train_oneshot_loader, 1)

    koch2015.summary(device.type)
    trainer.train()
