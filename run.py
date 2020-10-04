import time
from options.train_options import TrainOptions
from data import create_dataset 
from models import create_model
from util.tensorboard_visualizer import Visualizer
from util.get_data import GetData

# Polyaxon
from polyaxon_client.tracking import Experiment, get_data_paths

if __name__ == '__main__':

    opt = TrainOptions().parse()   # get training options

    # Polyaxon
    if opt.local == False:
        experiment = Experiment()

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        
        for i, data in enumerate(dataset):  # inner loop within one epoch

            # update the total number of iterations and the iterations in current epoch
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            # preprocess data and update weights
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

        # Logging
        # log images to tensorboard  
        model.compute_visuals()
        visualizer.log_images(model.get_current_visuals(), epoch)

        # log losses to tensorboard  
        losses = model.get_current_losses()
        visualizer.log_losses(epoch, losses)

        # Save the model at the end of each epoch
        model.save_networks('latest')
        model.save_networks(epoch)
        print('Epoch: {} | Model Saved'.format(epoch))
        for key, value in losses.items():
            print('  {} : {:.3f}'.format(key, value))
