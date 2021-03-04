import pytorch_lightning as pl
from argparse import ArgumentParser
from models import unet_trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profiler import AdvancedProfiler


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

    parser = unet_trainer.UnetVAETrainer.add_model_specific_args(parser)

    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        verbose=True,
        monitor='loss',
        mode='min',
        prefix=''
    )
    trainer = pl.Trainer.from_argparse_args(args, checkpoint_callback=checkpoint_callback)
    model = unet_trainer.UnetVAETrainer(args)

    trainer.fit(model)


if __name__ == '__main__':
    main()
