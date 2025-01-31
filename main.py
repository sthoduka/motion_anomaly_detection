import pytorch_lightning as pl
from argparse import ArgumentParser
from models import unet_trainer
from pytorch_lightning.callbacks import ModelCheckpoint


def main():
    parser = ArgumentParser()

    parser.add_argument('--video_root', default='', type=str, help='Root path of input videos')
    parser.add_argument('--val_video_root', default='', type=str, help='Root path of validation videos')
    parser.add_argument('--test_video_root', default='', type=str, help='Root path of test videos')
    parser.add_argument('--resize_size', default=80, type=int, help='resize to this size before cropping')
    parser.add_argument('--sample_size', default=64, type=int, help='final image size to crop to')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch Size')
    parser.add_argument('--n_threads', default=4, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')

    parser.add_argument('--flow_type', type=str, default='normal', help='normal, normal_masked, registered, registered_masked')
    parser.add_argument('--latent_dim', type=int, default=6, help='dimension of latent space')
    parser.add_argument('--beta', type=float, default=10.0, help='weighting of KL-div in loss function')
    parser.add_argument('--prediction_offset', type=int, default=9, help='Input should be a maximum of how many frames in the past?')
    parser.add_argument('--prediction_offset_start', type=int, default=5, help='Input should be a minimum of how many frames in the past?')
    parser.add_argument('--max_epochs', default=50, type=int, help='maximum epochs')
    parser.add_argument('--accumulate_grad_batches', default=1, type=int, help='gradient accumulation steps')
    parser.add_argument('--log_dir', default='', type=str, help='directory to store logs')

    args = parser.parse_args()

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        monitor='val_loss',
        every_n_epochs=1,
        mode='min',
    )
    trainer_obj = pl.Trainer(accelerator="gpu", devices="1", log_every_n_steps=1, default_root_dir=args.log_dir, max_epochs=args.max_epochs, accumulate_grad_batches=args.accumulate_grad_batches, callbacks=[checkpoint_callback])
    model = unet_trainer.UnetVAETrainer(args)

    trainer_obj.fit(model)


if __name__ == '__main__':
    main()
