import os
from data import DIV2K
from model import generator, discriminator
from train import SrganTrainer, SrganGeneratorTrainer

# -----------------------------------------------------------
#  Download and prepare dataset
# -----------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--prepare", action="store_true", default=False)
    parser.add_argument("--train_generator", action="store_true", default=False)
    parser.add_argument("--train_gans", action="store_true", default=False)

    args = parser.parse_args()

    if args.prepare: 
        print('Preparing train dataset...')
        div2k_train = DIV2K(scale=4, subset='train', downgrade='bicubic')
        train_ds = div2k_train.dataset(batch_size=16, random_transform=True)

        print('Preparing valisation dataset')
        div2k_valid = DIV2K(scale=4, subset='valid', downgrade='bicubic')
        valid_ds = div2k_valid.dataset(batch_size=16, random_transform=True, repeat_count=1)

# -----------------------------------------------------------
#  Training Generator
# -----------------------------------------------------------
    if args.train_generator:
        pre_trainer = SrganGeneratorTrainer(model=generator,
                                            checkpoint_dir='./log/ckpt/pre_generator')
        pre_trainer.train(train.ds,
                        valid_ds.take(10),
                        steps=1000,
                        evaluate_every=50,
                        save_best_only=False)

        pre_trainer.model.save_weights(weights_file('pre_generator.h5'))

# -----------------------------------------------------------
#  Training GANS
# -----------------------------------------------------------
    if args.train_gans:
        gan_generator = generator()
        gan_generator.load_weights(weights_file('pre_generator.h5'))

        gan_trainer = SrganTrainer(generator=gan_generator, discriminator=discriminator())
        gan_trainer.train(train_ds, steps=200000)

        gan_trainer.generator.save_weights(weights_file('gan_generator.h5'))
        gan_trainer.discriminator.save_weights(weights_file('gan_discriminator.h5'))

