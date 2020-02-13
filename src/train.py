import argparse
import time
import tensorflow as tf

from data import DIV2K
from utils import evaluate
import model as srgan

from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay


# -----------------------------------------------------------
# Build base trainer
# -----------------------------------------------------------
class Trainer: 
    
    def __init__(self, 
                 model, 
                 loss, 
                 learning_rate, 
                 checkpoint_dir='../log/ckpt'):

        self.now = None
        self.loss = loss
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                              psnr=tf.Variable(-1.0),
                                              optimizer=Adam(learning_rate),
                                              model=model)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                             directory=checkpoint_dir, 
                                                             max_to_keep=3)
        self.restore()

    @property
    def model(self):
        return self.checkpoint.model

    def train(self, train_ds, valid_ds, steps, evaluate_every=1000, save_best_only=False):
        loss_mean = Mean()

        ckpt_mgr = self.checkpoint_manager
        ckpt = self.checkpoint

        self.now = time.perf_counter()

        for lr, hr in train_ds.take(steps - ckpt.step.numpy()):
            ckpt.step.assign_add(1)
            step = ckpt.step.numpy()

            loss = self.train_step(lr, hr)
            loss_mean(loss)

            if step % evaluate_every == 0:
                # Record loss value
                loss_value = loss_mean.result()
                loss_mean.reset_states()
                
                # Comput PSNR on validation set
                psnr_value = self.evaluate(valid_ds)
                
                # Calculate time consumed
                duration = time.perf_counter() - self.now
                print('{}/{}: loss = {:.3f}, PSNR = {:.3f} ({:.2f}s)'.format(step, steps, loss_value.numpy(), psnr_value.numpy(), duration))

                # Skip checkpoint if PSNR does not improve
                if save_best_only and psnr_value <= ckpt.psnr:
                    self.now = time.perf_counter()
                    continue
                
                # Save checkpoint
                ckpt.psnr = psnr_value
                ckpt_mgr.save()

                self.now = time.perf_counter()

    @tf.function
    def train_step(self, lr, hr):
        with tf.GradientTape() as tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            sr = self.checkpoint.model(lr, training=True)
            loss_value = self.loss(hr, sr)
        
        gradients = tape.gradient(loss_value, self.checkpoint.model.trainable_variables)
        self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_variables))

        return loss_value
    
    def evaluate(self, ds):
        ''' Evaluate training process by PSNR metrics
        '''
        return evaluate(self.checkpoint.model, ds)
    
    def restore(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print('Model restored from checkpoint at step {}.'.format(self.checkpoint.step.numpy()))


# -----------------------------------------------------------
# Build GANS trainer which inherits base trainer
# -----------------------------------------------------------

class SrganGeneratorTrainer(Trainer):
    
    def __init__(self,
                 model,
                 checkpoint_dir,
                 learning_rate=1e-4):
        super().__init__(model, loss=MeanSquaredError(), learning_rate=learning_rate, checkpoint_dir=checkpoint_dir)
        # Using super() to refer back to the base class Trainer
        
class SrganTrainer:

    def __init__(self,
                 generator,
                 discriminator,
                 content_loss='VGG54',
                 learning_rate=PiecewiseConstantDecay(boundaries=[100000], values=[1e-4, 1e-5]),
                 log_dir='../log'):
        
        if content_loss == 'VGG22':
            self.vgg = srgan.vgg_22()
        elif content_loss == 'VGG54':
            self.vgg = srgan.vgg_54()
        else: 
            raise ValueError("content_loss must be either 'VGG22' or 'VGG54'")

        self.content_loss = content_loss
        self.log_dir = log_dir
        
        self.generator = generator
        self.discriminator = discriminator 
        self.generator_optimizer = Adam(learning_rate=learning_rate)
        self.discriminator_optimizer = Adam(learning_rate=learning_rate)

        self.binary_cross_entropy = BinaryCrossentropy(from_logits=False)
        self.mean_squared_error = MeanSquaredError()

    def train(self, train_ds, steps=200000):
        pls_metric = Mean()
        dls_metric = Mean()
        step = 0

        log_file = open(log_dir + 'losses.txt' , 'w+')
        log_file.close()

        for lr, hr in train_ds.take(steps):
            step += 1

            pl, dl = self.train_step(lr, hr)
            pls_metric(pl)
            dls_metric(dl)

            if step % 50 == 0:
                print('{}/{}, perceptual loss = {:.4f}, discriminator loss = {:.4f}'.format(step, steps, pls_metric.result(), dls_mtric.result()))
                
                # Update log file
                log_file = open(log_dir + 'losses.txt' , 'a')
                log_file.write('{}/{}, perceptual loss = {:.4f}, discriminator loss = {:.4f}\n'.format(step, steps, pls_metric.result(), dls_mtric.result()))
                log_file.close()

                # Restart metrics
                pls.metric.reset_states()
                dls.metric.reset_states()
    
    @tf.function
    def train_step(self, lr, hr):
        ''' Calculate training loss including
            Perceptual loss 
            Discriminator loss 
        '''
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            sr = self.generator(lr, training=True)

            hr_output = self.discriminator(hr, training=True)
            sr_output = self.discriminator(sr, training=True)

            con_loss = self._content_loss(hr, sr)
            gen_loss = self._generator_loss(sr_output)
            perc_loss = con_loss + 0.001 * gen_loss
            disc_loss = self._discriminator_loss(hr_output, sr_output)
        
        gradients_of_generator = gen_tape.gradient(perc_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return perc_loss, disc_loss

    @tf.function
    def _content_loss(self, hr, sr):
        sr = preprocess_input(sr)
        hr = preprocess_input(hr)
        sr_features = self.vgg(sr) / 12.75
        hr_features = self.vgg(hr) / 12.75
        return self.mean_squared_error(hr_features, sr_features)

    def _generator_loss(self, sr_out):
        return self.binary_cross_entropy(tf.ones_like(sr_out), sr_out)
    
    def _discriminator_loss(self, hr_out, sr_out):
        hr_loss = self.binary_cross_entropy(tf.ones_like(hr_out), hr_out)
        sr_loss = self.binary_cross_entropy(tf.zeros_like(hr_out), hr_out)
        return hr_loss + sr_loss

# -----------------------------------------------------------
#  Command
# -----------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, required=True)
    parser.add_argument("--step", type=str, required=True)
    parser.add_argument("--evaluate", type=str, required=True)
    args = parser.parse_args()

    # TRAIN GENERATOR
    if args.type == "generator": 
        pre_trainer = SrganGeneratorTrainer(model=srgan.generator,
                                            checkpoint_dir='../log/ckpt')
        pre_trainer.train(train.ds,
                        valid_ds.take(10),
                        steps=args.step,
                        evaluate_every=args.evaluate,
                        save_best_only=False)

        pre_trainer.model.save_weights(weights_file('pre_generator.h5'))

    # TRAIN GANS
    elif args.type == "gans":
        gan_generator = srgan.generator()
        gan_generator.load_weights(weights_file('pre_generator.h5'))

        gan_trainer = SrganTrainer(generator=gan_generator, discriminator=srgan.discriminator())
        gan_trainer.train(train_ds, steps=args.step, evaluate_every=args.evaluate)

        gan_trainer.generator.save_weights(weights_file('gan_generator.h5'))
        gan_trainer.discriminator.save_weights(weights_file('gan_discriminator.h5'))