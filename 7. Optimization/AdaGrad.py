import DIDLutils
import torch.optim as optim
import LoadAirfoilNoiseDataset

features, labels = LoadAirfoilNoiseDataset.get_data()

DIDLutils.train_pytorch(optim.Adagrad, {"lr": 0.1}, features, labels, num_epochs=10 )

DIDLutils.train_pytorch(optim.Adam, {"lr": 0.1}, features, labels, num_epochs=10 )