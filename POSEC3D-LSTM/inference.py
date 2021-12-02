import torch
import pickle
import os.path as osp
import os
import numpy as np
import pandas as pd
from mmcv import Config

from utils.models import modelFromConfig
from utils.mics import parse_args, PoseC3DTransform, createKNN, plot
from utils.metrics import hitAtK, precisionAtK, meanAveragePrecision

def main(args):
    cfg = Config.fromfile(args.cfg_path)
    
    # Read file contains path
    pathFile = pd.read_csv(args.txt_path, names=['filename', 'label', 'status'], header=None)
    pathFile = pathFile.replace(r'\\','/', regex=True)
    # Read keypoint annotation
    with open(args.kp_annotation, 'rb') as f:
        kp_annotation = pickle.load(f)

    # Embed dataset
    loader = PoseC3DTransform(cfg=cfg, 
                        sample_mode=args.sample_mode,
                        seed=255,
                        ram=True)
    
    checkpoint_paths = sorted(os.listdir(args.checkpoint))
    checkpoint_paths = [osp.join(args.checkpoint, checkpoint) for checkpoint in checkpoint_paths]
    
    metrics = []
    k = 1
    for checkpoint_path in checkpoint_paths:
        if not checkpoint_path.endswith(('.path', '.pth','.pth.tar')):
            continue
        # Build model
        model = modelFromConfig(cfg=cfg, checkpoint=checkpoint_path, device=args.device)
        model.eval()
        
        embed_path = osp.join('action-recognition/POSEC3D-LSTM/embeddings/uniform', "{}_{}".format(checkpoint_path.split('/')[-2],k))
        print(embed_path)
        k += 1
        if args.save and not os.path.exists(embed_path):
            pathList = pathFile.loc[pathFile['status'] != 'test', :]

            embeddings = np.zeros((len(pathList), model.cls_head.lstm.hidden_size))
            count = 0
            with torch.no_grad():
                for index, row in pathList.iterrows():
                    path = row['filename']

                    keypoint_dict = None
                    frameDirPath = osp.join('/content',path.split('.')[0])
                    for i in range(len(kp_annotation)):
                        if kp_annotation[i]['frame_dir'] == frameDirPath:
                            keypoint_dict = kp_annotation[i]
                            break
                    video = loader(keypoint_dict)
                    if args.device == 'cuda':
                        video = video.to(args.device)
                    pred = model(video, -1)
                    embeddings[count] = pred.to('cpu').numpy()
                    count += 1
            np.save(embed_path, embeddings)
        else:
            if not embed_path.endswith('.npy'):
                embed_path = embed_path + '.npy'
            embeddings = np.load(embed_path)

        knnModel = createKNN(embeddings, 30)
        pathList = pathFile.loc[pathFile['status'] == 'test', :].reset_index()

        with torch.no_grad():
            result = np.zeros((len(pathList), args.top))
            for index, row in pathList.iterrows():
                path = row['filename']
                keypoint_dict = None
                frameDirPath = osp.join('/content',path.split('.')[0])
                for i in range(len(kp_annotation)):
                    if kp_annotation[i]['frame_dir'] == frameDirPath:
                        keypoint_dict = kp_annotation[i]
                        break
                video = loader(keypoint_dict)
                if args.device == 'cuda':
                    video = video.to(args.device)
                pred = model(video, -1)
                dists, ids = knnModel.kneighbors(pred.to('cpu').numpy(), args.top)

                similarLabelList = pathFile.loc[pathFile['label']==row['label']].index
                isIn = np.isin(np.asarray(ids), np.asarray(similarLabelList))
                result[index,:] = isIn
            
            
            
            hak = hitAtK(result, args.top)
            pak = precisionAtK(result, args.top)
            mAP = meanAveragePrecision(result, args.top)

            metrics.append([hak, pak, mAP])
            print('Hit at {}: {:.2f}%'.format(args.top, hak*100))
            print('Precision at {}: {:.2f}%'.format(args.top, pak*100))
            print('Mean Average Precision at {}: {:.2f}'.format(args.top, mAP*100))
            print('\n')
                
    plot(metrics, ['hit at {}'.format(args.top), 'precision at {}'.format(args.top), 'mean average precision at {}'.format(args.top)], osp.join(args.checkpoint, 'metrics.csv'), osp.join(args.checkpoint, 'metrics.png'))
if __name__ == "__main__":
    args = parse_args()
    main(args)
