from utils.file_io.dataloading.abstract_dataloaders import DataLoaderNoConfig
from utils.file_io.dataloading.middlebury_dataloader import MiddleburyDataloader
from utils.file_io.dataloading.sintel_dataloader import SintelDataloader

def get_dataloader(args,ignore_config_settings = False):
    if ignore_config_settings:
        dl = DataLoaderNoConfig(args)
    else:
        if args.dataset == 'Middlebury':
            dl = MiddleburyDataloader(args)
        elif args.dataset == 'Sintel':
            dl  = SintelDataloader(args)
    return dl


