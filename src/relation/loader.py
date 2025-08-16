from torch.utils.data import DataLoader

from itmo_pirsii_2023_diploma.src.relation.dataset import RelationDataset, RelationDatasetBert, RawDataset


def get_loaders(train_path: str = None, val_path: str = None, test_path: str = None, tokenizer=None, batch_size=32,
                max_length=350, num_workers=7, debug_size=10, use_bert=False, output_dim=2, raw_dataset=False):
    assert tokenizer or raw_dataset
    assert not (use_bert and raw_dataset)

    if use_bert:
        dataset_cls = RelationDatasetBert
    elif raw_dataset:
        dataset_cls = RawDataset
    else:
        dataset_cls = RelationDataset
    train_loader, val_loader, test_loader = None, None, None
    if train_path:
        train_dataset = dataset_cls.from_csv(train_path, tokenizer, max_length, sep="\t", output_dim=output_dim)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  persistent_workers=True)  # , pin_memory=True
    if val_path:
        val_dataset = dataset_cls.from_csv(val_path, tokenizer, max_length, sep="\t", output_dim=output_dim)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                persistent_workers=True)
    if test_path:
        test_dataset = dataset_cls.from_csv(test_path, tokenizer, max_length, sep="\t", output_dim=output_dim)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 persistent_workers=True)
    if debug_size > 0:
        debug_dataset = dataset_cls.from_csv(test_path, tokenizer, max_length, sep="\t")
        debug_dataset.cut(stop=debug_size)
        debug_loader = DataLoader(debug_dataset, batch_size=1, shuffle=False, num_workers=1,
                                  persistent_workers=True)
        return train_loader, val_loader, test_loader, debug_loader

    return train_loader, val_loader, test_loader, None
