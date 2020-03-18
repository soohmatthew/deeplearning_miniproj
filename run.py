import pandas as pd
import os
import numpy as np
import torch
import datetime as dt
import time
import logging

from torch.utils import data
from torchvision import models, transforms
from PIL import Image
import GPUtil
from sklearn.metrics import average_precision_score
import torch.nn.functional as F
import torch.nn as nn
import copy
import matplotlib.pyplot as plt

class PascalVOC:
    """
    Handle Pascal VOC dataset
    """
    def __init__(self, root_dir):
        """
        Summary: 
            Init the class with root dir
        Args:
            root_dir (string): path to your voc dataset
        """
        self.root_dir = root_dir
        self.img_dir =  os.path.join(root_dir, 'JPEGImages/')
        self.ann_dir = os.path.join(root_dir, 'Annotations')
        self.set_dir = os.path.join(root_dir, 'ImageSets', 'Main')
        self.cache_dir = os.path.join(root_dir, 'csvs')
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def list_image_sets(self):
        """
        Summary: 
            List all the image sets from Pascal VOC. Don't bother computing
            this on the fly, just remember it. It's faster.
        """
        return [
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train',
            'tvmonitor']

    def _imgs_from_category(self, cat_name, dataset):
        """
        Summary: 
        Args:
            cat_name (string): Category name as a string (from list_image_sets())
            dataset (string): "train", "val", "train_val", or "test" (if available)
        Returns:
            pandas dataframe: pandas DataFrame of all filenames from that category
        """
        filename = os.path.join(self.set_dir, cat_name + "_" + dataset + ".txt")
        df = pd.read_csv(
            filename,
            delim_whitespace=True,
            header=None,
            names=['filename', 'true'])
        return df

    def imgs_from_category_as_list(self, cat_name, dataset):
        """
        Summary: 
            Get a list of filenames for images in a particular category
            as a list rather than a pandas dataframe.
        Args:
            cat_name (string): Category name as a string (from list_image_sets())
            dataset (string): "train", "val", "train_val", or "test" (if available)
        Returns:
            list of srings: all filenames from that category
        """
        df = self._imgs_from_category(cat_name, dataset)
        df = df[df['true'] == 1]
        return df['filename'].values

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, pic_filename, labels, transforms = None):
        self.labels = labels
        self.pic_filename = pic_filename
        self.transforms = transforms

    def __len__(self):
        return len(self.pic_filename)
    def __getitem__(self, index):
        ID = self.pic_filename[index]
        # Load data and get label
        X = Image.open(ID).convert('RGB')
        if self.transforms is not None:
            X = self.transforms(X)
        if self.labels is not None:
            y = torch.from_numpy(np.array(self.labels[index]))
            return X, y, ID
        else:
            return X, ID

def get_data(dataset, filepath, transforms = None):
    pv=PascalVOC(filepath)
    label_dict = {}
    for cat_name in pv.list_image_sets():
        ls = pv.imgs_from_category_as_list(cat_name, dataset)
        for pic in ls:
            if pic in label_dict:
                label_dict[pic].append(cat_name)
            else:
                label_dict[pic] = [cat_name]

    X = [pv.img_dir + '/' + i + '.jpg' for i in sorted([*label_dict])]
    y = [[0.0] * len(pv.list_image_sets()) for _ in range(len(X))]

    for index in range(len(X)):
        for label in label_dict[sorted([*label_dict])[index]]:
            y[index][pv.list_image_sets().index(label)] = 1.0
    
    return Dataset(pic_filename = X, labels = y, transforms = transforms)

def get_test_data(filepath, transforms):
    X = [filepath + '/' + x for x in os.listdir(filepath)]
    return Dataset(pic_filename = X, labels = None, transforms = transforms)

def prepare_data(batch_size, trainval_filepath, test_filepath, train_transforms = None, test_transforms = None):
    train_data = get_data('train', trainval_filepath, train_transforms)
    valid_data = get_data('val', trainval_filepath, test_transforms)
    test_data = get_test_data(test_filepath, test_transforms)

    train_dl = data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
    valid_dl = data.DataLoader(valid_data, batch_size = batch_size, shuffle = False)
    test_dl = data.DataLoader(test_data, batch_size = batch_size, shuffle = False)
    return train_dl, valid_dl, test_dl

def avg_precision(outputs, labels, threshold = 0.5):
    outputs = outputs.cpu().detach()
    outputs = np.where(outputs >= threshold, 1, 0) # Shape: (nsamples, nclass)
    classwise_ps = np.array([average_precision_score(labels[:,i], outputs[:,i]) for i in range(labels.shape[1])])
    # average precision score unable to handle all 0s, we choose to ignore the nans.
    avg_precision_score = np.nanmean(classwise_ps)
    return avg_precision_score, classwise_ps

def train_epoch(model,  trainloader,  criterion, device, optimizer, threshold = 0.5, print_batch_results = False, print_classwise_results = False):
    model.train()
 
    running_precision = [] 
    running_classwise_ps = []
    losses = []
    
    for batch_idx, data in enumerate(trainloader):

        inputs=data[0].to(device)
        labels=data[1].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward Feed
        outputs = model(inputs).cpu().float()
        labels = labels.cpu().float()
        # Computer Loss
        loss = criterion(outputs, labels)
        # Backpropagate
        loss.backward()
        # Update parameters
        optimizer.step()
        
        losses.append(loss.detach().cpu())

        # Track training precision score
        avg_precision_score, classwise_ps = avg_precision(outputs, labels, threshold)
        running_precision.append(avg_precision_score)
        running_classwise_ps.append(classwise_ps)
        if print_batch_results:
            print(f'Batch {batch_idx}. Loss: {loss.item():.2f}, Precision Score: {avg_precision_score}')

    # Prints the classwise precision scores in the train set
    if print_classwise_results:
        labels = [
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train',
            'tvmonitor']
        
        running_classwise_ps = np.array(running_classwise_ps)
        running_classwise_ps = np.nanmean(running_classwise_ps, axis = 0)        
        for label in range(len(labels)):
            print(f'Class: {labels[label]}, Precision Score: {running_classwise_ps[label]}')

    precision = np.mean(running_precision)
    return np.array(losses), precision

def evaluate(model, dataloader, criterion, device, threshold, last_epoch, validation_results_fp):

    model.eval()

    running_classwise_ps = []
    running_precision = [] 
    losses = []
    all_outputs = torch.tensor([], device=device)
    all_filepaths = [] 

    with torch.no_grad():
        for _, data in enumerate(dataloader):

            #print ('epoch at',len(dataloader.dataset), ctr)
            inputs = data[0].to(device)
            outputs = model(inputs).float()

            # If it is the last epoch, we want to save all the output results
            if last_epoch:
                all_outputs = torch.cat((all_outputs, outputs), 0)
            labels = data[1]
            labels = labels.cpu().float()

            filepaths = data[2]
            all_filepaths.append(list(filepaths))
            losses.append(criterion(outputs, labels.to(device)).detach().cpu().numpy())

            avg_precision_score, classwise_ps = avg_precision(outputs, labels, threshold)
            running_precision.append(avg_precision_score)
            running_classwise_ps.append(classwise_ps)
    
    if last_epoch:
        cwd = os.getcwd()
        if not os.path.exists(cwd+'/predictions/'):
            os.makedirs(cwd+'/predictions/')
        all_outputs = all_outputs.cpu().numpy()
        all_filepaths = np.array(all_filepaths)
        np.savez(validation_results_fp, all_outputs, all_filepaths)

    # Prints the classwise precision scores in the train set
    labels = [
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']
    
    running_classwise_ps = np.array(running_classwise_ps)
    running_classwise_ps = np.nanmean(running_classwise_ps, axis = 0)    
    for label in range(len(labels)):
        logging.info(f'Class: {labels[label]}, Precision Score: {running_classwise_ps[label]}')

    precision_score = np.mean(running_precision)

    return np.array(losses), precision_score

def trainModel(train_dataloader,
                validation_dataloader,
                model,
                criterion,
                optimizer,
                num_epochs,
                validation_results_fp,
                threshold = 0.5,
                scheduler = None,
                plot = True,
                device= torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                print_batch_results = False,
                print_classwise_results = False):
    if torch.cuda.is_available():
        model.cuda()

    best_measure = 0
    best_epoch =-1
    losses_epoch_training = []
    losses_epoch_val = []

    measure_epoch_training = []
    measure_epoch_val = []

    # Log results
    cwd = os.getcwd()
    if not os.path.exists(cwd+'/logs/'):
        os.makedirs(cwd+'/logs/')
    logging.basicConfig(filename="logs/" + model.__class__.__name__ + "_nn_training_" + str(time.ctime()).replace(':','').replace('  ',' ').replace(' ','_') + ".log",
                        format='%(message)s',
                        level=logging.INFO)

    logging.info(f"Training Log for {model.__class__.__name__} on {str(time.ctime()).replace(':','').replace('  ',' ').replace(' ','_')}")

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-' * 10)

        if epoch >= 5:
            for idx, param in enumerate(model.parameters()):
                param.requires_grad = True

        model.train(True)
        GPUtil.showUtilization()
        training_losses, training_measure = train_epoch(model, train_dataloader, criterion, device, optimizer, threshold, print_batch_results = print_batch_results, print_classwise_results = print_classwise_results)
        
        losses_epoch_training.append(training_losses)
        measure_epoch_training.append(training_measure)

        logging.info(f'Average Training Loss for epoch {epoch}: {np.mean(training_losses):.3f}')
        print(f'Average Training Loss for epoch {epoch}: {np.mean(training_losses):.3f}')
        logging.info(f'Average Training Performance Measure for epoch {epoch}: {training_measure:.3f}')
        print(f'Average Training Performance Measure for epoch {epoch}: {training_measure:.3f}')
        
        if scheduler is not None:
            scheduler.step()

        model.train(False)
        logging.info(f'Classwise Precision Scores for epoch {epoch}')
        if epoch == num_epochs - 1: # If last epoch
            last_epoch = True
        else:
            last_epoch = False
        validation_losses, validation_measure = evaluate(model, validation_dataloader, criterion, device, threshold, last_epoch, validation_results_fp)
        losses_epoch_val.append(validation_losses)
        measure_epoch_val.append(validation_measure)

        # Record info in log
        logging.info(f'Average Validation Loss for epoch {epoch}: {np.mean(validation_losses):.3f}')
        print(f'Average Validation Loss for epoch {epoch}: {np.mean(validation_losses):.3f}')
        logging.info(f'Validation Performance Measure for epoch {epoch}: {validation_measure:.3f}')
        print(f'Validation Performance Measure for epoch {epoch}: {validation_measure:.3f}')

        if validation_measure > best_measure: #Higher measure better as higher measure is higher precision score
            bestweights = copy.deepcopy(model.state_dict())
            best_measure = validation_measure
            best_epoch = epoch
            print(f'Current best measure {validation_measure:.3f} at epoch {best_epoch}')
        print('')

    losses_epoch_training = np.array(losses_epoch_training).mean(1)
    measure_epoch_training = np.array(measure_epoch_training)

    losses_epoch_val = np.array(losses_epoch_val).mean(1)
    measure_epoch_val = np.array(measure_epoch_val)

    if plot:
        fig, ax = plt.subplots(1,2, figsize = (10,5))

        fig.suptitle(f"Learning Rate: {optimizer.state_dict()['param_groups'][0]['lr']}, Max Epochs: {num_epochs}")

        ax[0].set_title('Loss Value')
        ax[0].plot(losses_epoch_training, color = 'skyblue', label="Training Loss")
        ax[0].plot(losses_epoch_val, color = 'orange', label = "Validation Loss")
        ax[0].legend()
        ax[0].set_xticks(range(num_epochs))

        ax[1].set_title('Measure Value')
        ax[1].plot(measure_epoch_training, color = 'skyblue', label="Training Measure")
        ax[1].plot(measure_epoch_val, color = 'orange', label="Validation Measure")
        ax[1].legend()
        ax[1].set_xticks(range(num_epochs))

        cwd = os.getcwd()
        if not os.path.exists(cwd+'/plots/'):
            os.makedirs(cwd+'/plots/')
        plt.savefig("plots/nn_training_" + str(time.ctime()).replace(':','').replace('  ',' ').replace(' ','_') + ".png")

    return best_epoch, best_measure, bestweights

def predict(model, test_dl, save_weights_fp, device):
    model.load_state_dict(torch.load(save_weights_fp))
    model.eval()

    all_outputs = torch.tensor([], device=device)

    with torch.no_grad():
        for _, data in enumerate(test_dl):
            inputs = data[0].to(device)
            outputs = model(inputs)
            all_outputs = torch.cat((all_outputs, outputs), 0)

    all_outputs = all_outputs.cpu().numpy()
    return all_outputs

def train_test_model(train_dataloader,
                    validation_dataloader,
                    test_dataloader,
                    model,
                    criterion,
                    optimizer,
                    threshold = 0.5,
                    scheduler = None,
                    learning_rate = 0.01,
                    batch_size = 32,
                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                    num_epochs = 12,
                    **kwargs):

    if torch.cuda.is_available():
        print('Using GPU')
    else:
        print('Using CPU')
    
    #############
    # Train model
    if params['train_model']:
        start_time = dt.datetime.now()
        if params['verbose']:
            print('Training model...')
            print_batch_results = True
            print_classwise_results = True
        else:
            print_batch_results = False
        best_epoch, best_perfmeasure, bestweights = trainModel(train_dataloader = train_dataloader, validation_dataloader = validation_dataloader, model = model_ft, criterion = criterion , optimizer = optimizer, num_epochs = num_epochs, threshold = threshold, scheduler = scheduler, print_batch_results = print_batch_results, print_classwise_results = print_classwise_results, validation_results_fp = params['validation_results_fp'])
        print(f'Best epoch: {best_epoch} Best Performance Measure: {best_perfmeasure:.5f}')
        
        if params['verbose']:
            print('Saving weights...')
        cwd = os.getcwd()
        if not os.path.exists(cwd+'/model_weights/'):
            os.makedirs(cwd+'/model_weights/')
        torch.save(bestweights, save_weights_fp)
        print(f'Time Taken to train: {dt.datetime.now()-start_time}')

    ################
    # For prediction on test
    if params['predict_on_test']:
        if params['verbose']:
            print('Predicting on test set...')
        try:
            if torch.cuda.is_available():
                model.cuda()
            predictions = predict(model, test_dataloader, params['save_weights_fp'], device)
            if params['verbose']:
                print('Saving predictions...')
            np.save(params['predictions_fp'], predictions)
        except:
            raise ImportError('Model weights do not exist')

def get_max_min_results(label):
    
    labels = [
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']

    index = labels.index(label)
    # Get max scores
    temp = np.argpartition(-output_results.T[index], 5)
    max_result_args = pic_filepaths[temp[:5]]

    temp = np.partition(-output_results.T[index], 5)
    max_result = -temp[:5]

    # Get min scores
    temp = np.argpartition(output_results.T[index], 5)
    min_result_args = pic_filepaths[temp[:5]]
    temp = np.partition(output_results.T[index], 5)
    min_result = temp[:5]
    
    return max_result_args, max_result, min_result_args, min_result

def save_pic(output_results, pic_filepaths, label):
    max_result_args, max_result, min_result_args, min_result = get_max_min_results(label)
    max_filenames = list(max_result_args)
    min_filenames = list(min_result_args)

    cwd = os.getcwd()
    if not os.path.exists(cwd+'/topbot5/'):
        os.makedirs(cwd+'/topbot5/')
    fig,ax = plt.subplots(1,5)
    figsize=(15,8)
    dpi=150
    fig.set_size_inches(figsize)
    fig.set_dpi = dpi
    
    for i in range(len(max_filenames)):
        with open(max_filenames[i],'rb') as f:
            image=Image.open(f)
            ax[0][i].imshow(image)
            ax[0,i].axis('off')
    fig.suptitle(f'Top 5 pictures for {label}')
    fig.savefig(f'{cwd}/topbot5/{label}_top5.png', dpi=fig.dpi)

    for i in range(len(min_filenames)):
        with open(filenames[i],'rb') as f:
            image=Image.open(f)
            ax[0][i].imshow(image)
            ax[0,i].axis('off')
    fig.suptitle(f'Bottom 5 pictures for {label}')
    fig.savefig(f'{cwd}/topbot5/{label}_botom5.png', dpi=fig.dpi)

if __name__=='__main__':
    project_dir = os.getcwd() #"C:\\Users\\lohzy\\Desktop\\dl_project"

    # Set filepaths
    trainval_fp = os.path.join(project_dir,'VOCdevkit','VOC2012') # Location of trainval dataset
    test_fp = os.path.join(project_dir,'VOCdevkit','VOC2012_test','JPEGImages') # Location of test dataset

    # Set save destinations
    save_weights_fp = os.path.join(project_dir, f"model_weights/model_weights_{str(time.ctime()).replace(':','').replace('  ',' ').replace(' ','_')}.pth") # Save destination for model weights
    validation_results_fp = os.path.join(project_dir,f"predictions/validation_output_results_{str(time.ctime()).replace(':','').replace('  ',' ').replace(' ','_')}.npz")
    predictions_fp = os.path.join(project_dir, f"predictions/test_predictions_{str(time.ctime()).replace(':','').replace('  ',' ').replace(' ','_')}.npy") # Save destination for test set predictions

    # Set params, optimiser, loss and scheduler
    params = dict(
        train_model = True,
        predict_on_test = True,
        verbose = True,
        batch_size = 4,
        no_classes = 20,
        learning_rate = 0.005,
        num_epochs = 10,
        threshold = 0.5,
        criterion = torch.nn.BCELoss(),
        save_weights_fp = save_weights_fp,
        predictions_fp = predictions_fp,
        validation_results_fp = validation_results_fp
    )

    # Set transforms
    train_transforms_centcrop = transforms.Compose([transforms.Resize(280),
                                                    transforms.RandomCrop(224),
                                                    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                                    transforms.RandomRotation(20),
                                                    transforms.RandomHorizontalFlip(p=0.5),
                                                    transforms.RandomVerticalFlip(p=0.5),
                                                    transforms.RandomAffine(20),
                                                    transforms.RandomGrayscale(p=0.1),
                                                    transforms.ToTensor(), 
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    test_transforms_centcrop = transforms.Compose([transforms.Resize(280),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(), 
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    ##################
    # Load dataloaders
    if params['verbose']:
        print('Loading data...')
    train_dl, valid_dl, test_dl = prepare_data(batch_size = params['batch_size'], train_transforms = train_transforms_centcrop, test_transforms = test_transforms_centcrop, trainval_filepath = trainval_fp, test_filepath = test_fp)

    ##################
    # Initialise model
    #model_ft = models.resnet18(pretrained=True)
    #convo_output_num_features = model_ft.fc.in_features

    model_ft = models.densenet121(pretrained=True)
    convo_output_num_features = model_ft.classifier.in_features

    for param in model_ft.parameters():
        param.requires_grad = False

    model_ft.classifier = torch.nn.Sequential(
        torch.nn.Linear(convo_output_num_features, 20),
        torch.nn.Sigmoid()
    )

    params['model'] = model_ft
    params['optimizer'] = torch.optim.Adam(model_ft.parameters(),lr=params['learning_rate'])
    params['scheduler'] = torch.optim.lr_scheduler.MultiStepLR(params['optimizer'], milestones=[5], gamma=0.1)

    # Train model
    train_test_model(train_dataloader = train_dl, validation_dataloader = valid_dl, test_dataloader = test_dl, **params)

    # Get filepath
    validation_results = np.load(validation_results_fp, allow_pickle = True)
    output_results, pic_filepaths = validation_results.files
    pic_filepaths = validation_results[pic_filepaths]
    output_results = validation_results[output_results]
    pic_filepaths = np.hstack(pic_filepaths)

    for label in ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle']:
        save_pic(output_results, pic_filepaths, label)