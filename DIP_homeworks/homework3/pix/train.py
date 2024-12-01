import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from FCN_network import FullyConvNetwork
from torch.optim.lr_scheduler import StepLR

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.main(x)

def tensor_to_image(tensor):
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = (image + 1) / 2
    image = (image * 255).astype(np.uint8)
    return image

def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    for i in range(num_images):
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        comparison = np.hstack((input_img_np, target_img_np, output_img_np))

        cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison[..., ::-1])  # Save as BGR for OpenCV

def adversarial_loss(input, target_is_real, real_label=1.0, fake_label=0.0):
    label = torch.full_like(input, real_label if target_is_real else fake_label, device=input.device)
    criterion = nn.BCEWithLogitsLoss()
    return criterion(input, label)

def train_one_epoch(generator, discriminator, dataloader, g_optimizer, d_optimizer, device, epoch, num_epochs):
    generator.train()
    discriminator.train()

    g_running_loss = 0.0
    d_running_loss = 0.0

    for i, (image_rgb, image_semantic) in enumerate(dataloader):
        image_rgb = image_rgb.to(device)
        image_semantic = image_semantic.to(device)

        # Generate fake images
        fake_images = generator(image_rgb)

        # Train Discriminator
        d_optimizer.zero_grad()

        # Real images should be classified as real
        real_outputs = discriminator(torch.cat([image_rgb, image_semantic], dim=1))
        d_real_loss = adversarial_loss(real_outputs, target_is_real=True)

        # Fake images should be classified as fake
        fake_outputs = discriminator(torch.cat([image_rgb, fake_images.detach()], dim=1))
        d_fake_loss = adversarial_loss(fake_outputs, target_is_real=False)

        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        g_optimizer.zero_grad()

        # Adversarial Loss: fool the discriminator
        gen_outputs = discriminator(torch.cat([image_rgb, fake_images], dim=1))
        g_adv_loss = adversarial_loss(gen_outputs, target_is_real=True)

        # L1 Loss: make sure the generated image is close to the real image
        l1_criterion = nn.L1Loss()
        l1_loss = l1_criterion(fake_images, image_semantic)

        # Total Generator Loss
        g_loss = g_adv_loss + 100 * l1_loss
        g_loss.backward()
        g_optimizer.step()

        g_running_loss += g_loss.item()
        d_running_loss += d_loss.item()

        if epoch % 5 == 0 and i == 0:
            save_images(image_rgb, image_semantic, fake_images, 'train_results', epoch)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], G_Loss: {g_loss.item():.4f}, D_Loss: {d_loss.item():.4f}')

def validate(generator, discriminator, dataloader, device, epoch, num_epochs):
    generator.eval()
    discriminator.eval()

    g_val_loss = 0.0
    d_val_loss = 0.0

    with torch.no_grad():
        for i, (image_rgb, image_semantic) in enumerate(dataloader):
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)

            fake_images = generator(image_rgb)

            real_outputs = discriminator(torch.cat([image_rgb, image_semantic], dim=1))
            d_real_loss = adversarial_loss(real_outputs, target_is_real=True)

            fake_outputs = discriminator(torch.cat([image_rgb, fake_images], dim=1))
            d_fake_loss = adversarial_loss(fake_outputs, target_is_real=False)

            d_loss = (d_real_loss + d_fake_loss) / 2

            gen_outputs = discriminator(torch.cat([image_rgb, fake_images], dim=1))
            g_adv_loss = adversarial_loss(gen_outputs, target_is_real=True)

            l1_criterion = nn.L1Loss()
            l1_loss = l1_criterion(fake_images, image_semantic)

            g_loss = g_adv_loss + 100 * l1_loss

            g_val_loss += g_loss.item()
            d_val_loss += d_loss.item()

            if epoch % 5 == 0 and i == 0:
                save_images(image_rgb, image_semantic, fake_images, 'val_results', epoch)

    avg_g_val_loss = g_val_loss / len(dataloader)
    avg_d_val_loss = d_val_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation G Loss: {avg_g_val_loss:.4f}, Validation D Loss: {avg_d_val_loss:.4f}')

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_dataset = FacadesDataset(list_file='train_list.txt')
    val_dataset = FacadesDataset(list_file='val_list.txt')

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=4)

    generator = FullyConvNetwork().to(device)
    discriminator = Discriminator().to(device)

    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    scheduler_g = StepLR(g_optimizer, step_size=200, gamma=0.2)
    scheduler_d = StepLR(d_optimizer, step_size=200, gamma=0.2)

    num_epochs = 20
    for epoch in range(num_epochs):
        train_one_epoch(generator, discriminator, train_loader, g_optimizer, d_optimizer, device, epoch, num_epochs)
        validate(generator, discriminator, val_loader, device, epoch, num_epochs)

        scheduler_g.step()
        scheduler_d.step()

        if (epoch + 1) % 20 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(generator.state_dict(), f'checkpoints/generator_model_epoch_{epoch + 1}.pth')
            torch.save(discriminator.state_dict(), f'checkpoints/discriminator_model_epoch_{epoch + 1}.pth')

if __name__ == '__main__':
    main()



