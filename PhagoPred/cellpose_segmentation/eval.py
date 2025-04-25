from cellpose import models, metrics, core, io, plot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path

from PhagoPred import SETTINGS
from PhagoPred.utils import tools


def cellpose_eval(directory, model=SETTINGS.CELLPOSE_MODEL / 'models' / 'model', thresholds=np.arange(0.5, 1.0, 0.05)):
    use_GPU = core.use_gpu()
    io.logger_setup()
    model = models.CellposeModel(gpu=use_GPU, pretrained_model=str(model))
    #model = models.CellposeModel(gpu=use_GPU, pretrained_model=False, model_type='cyto3')
    im_names = [im.stem[:4] for im in directory.iterdir() if 'im' in im.stem]
    validation_ims = [io.imread(im) for im in directory.iterdir() if 'im' in im.name]
    channels = [0, 0]
    preds, flow, styles = model.eval(validation_ims, flow_threshold=0.5)
    for pred, im_name in zip(preds, im_names):
        Image.fromarray(pred.astype(np.uint16)).save(str(directory/f'{im_name}pred.png'))

        #plt.imsave(str(directory/f'{im_name}pred.png'), pred, cmap='gray')
    true_masks = [io.imread(im) for im in directory.iterdir() if 'mask' in im.name]
    # NOTE, this is not AP as defined elsewhere
    APs, TPs, FPs, FNs = metrics.average_precision(true_masks, preds, threshold=thresholds)
    precisions = TPs / (TPs+FPs)
    recalls = TPs / (TPs+FNs)
    F1s = TPs / (TPs + 0.5*(FPs+FNs))
    for i, im_name in enumerate(im_names):
        df = pd.DataFrame({'Precision': precisions[i],
                         'Recall': recalls[i],
                         'F1': F1s[i]},
                        index=thresholds)
        df.to_csv(str(directory / f'{im_name}_results.txt'), sep='\t')
        #view_frame.show_frame(str(directory / f'{im_name}im.png'), str(directory /f'{im_name}pred.png'), str(directory /f'{im_name}_view.png'))
        plt.imsave(str(directory /f'{im_name}_view.png'), tools.show_segmentation(np.array(validation_ims[i]), np.array(preds[i]).astype(np.int16), np.array(true_masks[i]).astype(np.int16)))


def cellpose_eval_from_ims(directory):
    image = [io.imread(str(directory/'Images'/'snap01.png'))]
    mask = [io.imread(str(directory/'Masks'/'01mask.png')).astype('int16')]
    pred = [io.imread(str(directory/'Masks'/'01pred.png')).astype('int16')]


    thresholds = np.arange(0.5, 1.0, 0.05)
    APs, TPs, FPs, FNs = metrics.average_precision(mask, pred, threshold=thresholds)
    precisions = TPs / (TPs + FPs)
    recalls = TPs / (TPs + FNs)
    F1s = TPs / (TPs + 0.5 * (FPs + FNs))

    df = pd.DataFrame({'Precision': precisions[0],
                       'Recall': recalls[0],
                       'F1': F1s[0]},
                      index=thresholds)
    df.to_csv(str(directory / f'01_results.txt'), sep='\t')
    # view_frame.show_frame(str(directory / '{im_name}im.png'), str(directory /f'{im_name}pred.png'), str(directory /f'{im_name}_view.png'))
    plt.imsave(str(directory / f'01_view.png'),
               tools.show_segmentation(np.array(image[0]), np.array(pred[0]).astype(np.int16),
                                       np.array(mask[0]).astype(np.int16)))


def plot_results(cellpose_results, rcnn_results):
    cellpose_results = pd.concat([pd.read_csv(file, sep='\t', index_col=0) for file in cellpose_results], axis=0)
    rcnn_results =  pd.concat([pd.read_csv(file, sep='\t', index_col=0) for file in rcnn_results], axis=0)
    cellpose_means, cellpose_stds = cellpose_results.groupby(level=0).mean(), cellpose_results.groupby(level=0).std()
    rcnn_means, rcnn_stds = rcnn_results.groupby(level=0).mean(), rcnn_results.groupby(level=0).std()
    metrics = cellpose_means.columns.values
    thresholds = cellpose_means.index.values
    print(thresholds)
    plt.rcParams["font.family"] = 'serif'
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    for ax, metric in zip(axs, metrics):
        ax.plot(thresholds, cellpose_means[metric], color='red', label='Cellpose')
        ax.fill_between(thresholds, cellpose_means[metric]-cellpose_stds[metric], cellpose_means[metric]+cellpose_stds[metric], color='red', alpha=0.5, edgecolor='none')
        ax.plot(thresholds, rcnn_means[metric], color='navy', label='Mask R-CNN')
        ax.fill_between(thresholds, rcnn_means[metric]-rcnn_stds[metric], rcnn_means[metric]+rcnn_stds[metric], color='navy', alpha=0.5, edgecolor='none')
        ax.set_xlabel('IOU Threshold')
        ax.set_ylabel(metric)
        ax.grid(True)
    plt.legend()
    plt.savefig('comparison_results.png')


def plot_train_dat_results(directory):
    results = [pd.read_csv(file, sep='\t', index_col=0) for file in directory.iterdir() if file.suffix=='.txt']
    metrics = results[0].columns.values
    thresholds = results[0].index.values
    num_training_data = range(1, len(results)+1)
    plt.rcParams["font.family"] = 'serif'
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric in zip(axs, metrics):
        for result, num_data in zip(results, num_training_data):
            ax.plot(thresholds, result[metric], label=num_data)
            ax.grid()
    plt.legend()
    plt.savefig('/home/ubuntu/Documents/detectron_and_track/cellpose_Models/test_training_data2/results.png')


def main():
    for name in ['cyto3_1', 'cyto3_2', 'cyto3_3', 'cyto3_4']:
        cellpose_eval(SETTINGS.CELLPOSE_MODEL / 'validate', name)

    plot_train_dat_results(Path('/home/ubuntu/Documents/detectron_and_track/cellpose_Models/test_training_data2/validate'))
    #cellpose_eval_from_ims(SETTINGS.MASK_RCNN_MODEL / 'Training_Data' / 'validate')
    # cellpose_results = ['/home/ubuntu/Documents/detectron_and_track/cellpose_Models/filters01/validate/02_results.txt',
    #                     '/home/ubuntu/Documents/detectron_and_track/cellpose_Models/filters02/validate/01_results.txt',
    #                     '/home/ubuntu/Documents/detectron_and_track/cellpose_Models/filters12/validate/00_results.txt']
    #
    # rcnn_results = ['/home/ubuntu/Documents/detectron_and_track/Models/filter01/Training_Data/validate/02_results.txt',
    #                 '/home/ubuntu/Documents/detectron_and_track/Models/filter02/Training_Data/validate/01_results.txt',
    #                 '/home/ubuntu/Documents/detectron_and_track/Models/filter12/Training_Data/validate/00_results.txt']
    # plot_results(cellpose_results, rcnn_results)
if __name__ == '__main__':
    main()