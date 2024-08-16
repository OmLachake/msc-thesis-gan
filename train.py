# Basile Van Hoorick, Jan 2020
# Om Lachake, Aug 2024

if __name__ == '__main__':

    import torch
    from outpainting import *
    from loss import HueLoss, PerceptualLoss

    print("PyTorch version: ", torch.__version__)
    print("Torchvision version: ", torchvision.__version__)

    # Define paths

    
    dataset_path = './dataset_name'

    model_save_path = './temp/outpaint_models'
    html_save_path = './temp/outpaint_html'
    train_dir = f'{dataset_path}/train'
    val_dir = f'{dataset_path}/val'
    test_dir = f'{dataset_path}/test'

    # Define datasets & transforms
    my_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
    
    batch_size = 16
    train_data = CEImageDataset(train_dir, my_tf, output_size, input_size, outpaint=True)
    val_data = CEImageDataset(val_dir, my_tf, output_size, input_size, outpaint=True)
    test_data = CEImageDataset(test_dir, my_tf, output_size, input_size, outpaint=True)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=1)
    print('train:', len(train_data), 'val:', len(val_data), 'test:', len(test_data))

    device = torch.device('cuda:0')
    G_net = CEGenerator(extra_upsample=True)
    D_net = CEDiscriminator()
    G_net.apply(weights_init_normal)
    D_net.apply(weights_init_normal)
    G_net.to(device)
    D_net.to(device)
    print('device:', device)

    # Define losses
    PixelLoss = nn.L1Loss()
    DiscriminatorLoss = nn.BCEWithLogitsLoss()
    Hue_Loss = HueLoss()
    Perception_Loss = PerceptualLoss()

    optimizer_G = optim.Adam(G_net.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D_net.parameters(), lr=1e-4, betas=(0.5, 0.999))

    PixelLoss.to(device)
    DiscriminatorLoss.to(device)
    Hue_Loss.to(device)
    Perception_Loss.to(device)

    # Start training
    data_loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader} 
    n_epochs = 100
    adv_weight = [0.001, 0.005, 0.015, 0.040, 0.1]

    start = time.time()
    hist_loss = train_CE(G_net, D_net, device, PixelLoss, DiscriminatorLoss, Hue_Loss, Perception_Loss, optimizer_G, optimizer_D,
                         data_loaders, model_save_path, html_save_path, n_epochs=n_epochs, adv_weight=adv_weight)

    # Save loss history and final generator
    pickle.dump(hist_loss, open('./temp/hist_loss.p', 'wb'))
    torch.save(G_net.state_dict(), './temp/generator_final.pt')
    end = time.time()

    print('SAVED')
    print(f'\n\nTime for {dataset_path} - {n_epochs} - random mask - p over hue = {end - start:.4f} seconds')
