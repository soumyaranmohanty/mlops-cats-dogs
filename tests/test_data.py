from src.data.preprocess import get_data_loaders

def test_data_loader():
    train_loader, val_loader, test_loader = get_data_loaders("data/raw", batch_size=4)

    # Check loaders exist
    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None

    # Check dataset not empty
    assert len(train_loader.dataset) > 0
    assert len(val_loader.dataset) > 0
    assert len(test_loader.dataset) > 0