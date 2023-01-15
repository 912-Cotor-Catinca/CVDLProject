import torch
import torch.utils.data
import torchvision.transforms as T
import config
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import utils
from decoder_model import ConvDecoder
from encoder_model import ConvEncoder
from engine import train_step, val_step, create_embedding
from folder_dataset import FolderDataset

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print("Setting Seed for the run, seed = {}".format(config.SEED))

    utils.seed_everything(config.SEED)

    transforms = T.Compose([T.Resize((512, 512)), T.ToTensor()])

    full_dataset = FolderDataset(config.IMG_PATH, transforms)
    train_size = int(config.TRAIN_RATIO * len(full_dataset))
    validation_size = len(full_dataset) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(full_dataset, [train_size, validation_size])

    train_loader = torch.utils.data.DataLoader(
                            train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True, drop_last=True
                    )
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=config.TEST_BATCH_SIZE)
    full_loader = torch.utils.data.DataLoader(full_dataset, batch_size=config.FULL_BATCH_SIZE)

    loss_fn = nn.MSELoss()
    encoder = ConvEncoder()
    decoder = ConvDecoder()

    encoder.to(device)
    decoder.to(device)

    autoencoder_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.AdamW(autoencoder_params, lr=config.LEARNING_RATE)

    max_loss = 9999

    for epoch in tqdm(range(config.EPOCHS)):
        train_loss = train_step(encoder, decoder, train_loader, loss_fn, optimizer, device=device)
        print(f"Epochs = {epoch}, Training Loss : {train_loss}")
        validation_loss = val_step(encoder, decoder, validation_loader, loss_fn, device=device)
        if validation_loss < max_loss:
            print("Validation Loss decreased, saving new best model")
            torch.save(encoder.state_dict(), config.ENCODER_MODEL_PATH)
            torch.save(decoder.state_dict(), config.DECODER_MODEL_PATH)

        print(f"Epochs = {epoch}, Validation Loss : {validation_loss}")

    print("Training done")

    embedding = create_embedding(encoder, full_loader, config.EMBEDDING_SHAPE, device)

    numpy_embedding = embedding.cpu().detach().numpy()
    num_images = numpy_embedding.shape[0]

    flattened_embedding = numpy_embedding.reshape((num_images, -1))
    np.save(config.EMBEDDING_PATH, flattened_embedding)
