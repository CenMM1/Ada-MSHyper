from data_provider.data_loader import Dataset_Multimodal_Classification, Dataset_PKL_Multimodal_Classification
from torch.utils.data import DataLoader

def data_provider(args, flag):
    data_format = getattr(args, 'data_format', 'pt')
    if data_format == 'pkl':
        Data = Dataset_PKL_Multimodal_Classification
    else:
        Data = Dataset_Multimodal_Classification

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    # 多模态数据加载
    data_set = Data(
        root_path=args.root_path,
        flag=flag,
        args=args
    )

    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        pin_memory=True)
    return data_set, data_loader
