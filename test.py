from options import test_options
#from dataloader import data_loader
from dataloader import data_loader
from model import create_model
from util import visualizer
from itertools import islice

import torch
import numpy as np
import random

#fix seed
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__=='__main__':
    # get testing options
    opt = test_options.TestOptions().parse()

    # set seed
    if not opt.seed is None:
        fix_seed(opt.seed)

    # creat a dataset
    dataset = data_loader.dataloader(opt)
    dataset_size = len(dataset) * opt.batchSize
    print('testing images = %d' % dataset_size)
    
    # create a model
    model = create_model(opt)
    model.eval()
    
    # create a visualizer
    visualizer = visualizer.Visualizer(opt)

    for i, data in enumerate(islice(dataset, opt.how_many)):
        model.set_input(data)
        s_set = model.test()
    
        # header for the file of estimated s
        if i == 0:
            fout = open("{}/s_estimate.csv".format(opt.results_dir), mode='w')
            if opt.num_sym == 4:
                fout.write("90r_s,180r_s,90f_s,180f_s,90r_m,180r_m,90f_m,180f_m,90r_sd,180r_sd,90f_sd,180f_sd,\n")
            if opt.num_sym == 5:
                fout.write("90r_s,180r_s,270r_s,90f_s,180f_s,90r_m,180r_m,270r_m,90f_m,180f_m,90r_sd,180r_sd,270r_sd,90f_sd,180f_sd,\n")
            else:
                for i in range(opt.num_sym*3):
                    fout.write(str(i)+',')
                fout.write('\n')

        # output estimated s
        for row in s_set:
            for elem in row:
                fout.write("{},".format(elem))
            fout.write("\n")
