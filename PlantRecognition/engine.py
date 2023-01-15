import torch
import torch.nn as nn


def train_step(encoder, decoder, train_loader, loss_fn, optimizer, device):
    """
    Performs a single training step
    :param encoder: A convolutional encoder
    :param decoder: A convolutional decoder
    :param train_loader: dataloader containing(images, images)
    :param loss_fn: loss function computed between 2 images
    :param optimizer: pytorch optimizer
    :param device: cuda or cpu
    :return: Train loss
    """
    encoder.train()
    decoder.train()

    for batch_idx, (train_img, target_img) in enumerate(train_loader):
        train_img = train_img.to(device)
        target_img = target_img.to(device)

        optimizer.zero_grad()
        encoder_output = encoder(train_img)
        decoder_output = decoder(encoder_output)

        loss = loss_fn(decoder_output, target_img)
        loss.backward()
        optimizer.step()
    return loss.item()


def val_step(encoder, decoder, val_loader, loss_fn, device):
    """
    Perform a single training step
    :param encoder: A convolutional encoder
    :param decoder: A convolutional decoder
    :param val_loader: dataloader containing(images, images)
    :param loss_fn: loss function computed between 2 images
    :param device: cpu or cuda
    :return: Validation loss
    """

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        for batch_idx, (train_img, target_img) in enumerate(val_loader):
            train_img = train_img.to(device)
            target_img = target_img.to(device)

            encoder_output = encoder(train_img)
            decoder_output = decoder(encoder_output)

            loss = loss_fn(decoder_output, target_img)

    return loss.item()


def create_embedding(encoder, full_loader, embedding_dim, device):
    """
    Creates embedding using encoder from dataloader.
    encoder: A convolutional Encoder. E.g. torch_model ConvEncoder
    full_loader: PyTorch dataloader, containing (images, images) over entire dataset.
    embedding_dim: Tuple (c, h, w) Dimension of embedding = output of encoder dimesntions.
    device: "cuda" or "cpu"
    Returns: Embedding of size (num_images_in_loader + 1, c, h, w)
    """
    # Set encoder to eval mode.
    encoder.eval()
    # Just a place holder for our 0th image embedding.
    embedding = torch.randn(embedding_dim)

    # Again we do not compute loss here so. No gradients.
    with torch.no_grad():
        for batch_idx, (train_img, target_img) in enumerate(full_loader):
            train_img = train_img.to(device)

            # Get encoder outputs and move outputs to cpu
            enc_output = encoder(train_img).cpu()
            # Keep adding these outputs to embeddings.
            # print(enc_output.shape)
            embedding = torch.cat((embedding, enc_output), 0)

    # Return the embeddings
    print(embedding.shape)
    return embedding
