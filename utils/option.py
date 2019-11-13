import argparse

def getArgs():
    parser = argparse.ArgumentParser(description='Sub-pixel Detection')
    
    default_max_epoch=3
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_epoch', type=int, default=default_max_epoch)
    parser.add_argument('--step_size', type=int, default=5,help='step of updating learning rate')
    parser.add_argument('--scale_factor', type=int, default=2,help="accuracy of sub-pixel(scale factor of SR)")
    parser.add_argument('--feature_size', type=int, default=3,help="num of channels of feature map")
    parser.add_argument('--num_ResBlock', type=int, default=2,help="num of ResBlock in model")
    parser.add_argument('--max_corners', type=int, default=1000,help="max num of corners in GT and test output")
    parser.add_argument('--threshold', type=float, default=0.5,help="threshold of test output,point is omitted if gray(point)<max_gray*threshold")

    parser.add_argument("-m",'--model_name', type=str, default='SPResNet',help='which model to use')
    parser.add_argument("-n",'--experiment_name', type=str, default='exp0',help='name of exp, used to identify logs and pkls')
    parser.add_argument("-e",'--load_epoch_pkl', type=int, default=default_max_epoch,help='pkl of this epoch will be loaded')

    parser.add_argument("-tr",'--isTrain', type=int, default=1,help='whether to train, 1 for Yes, 0 for No')
    parser.add_argument("-ts",'--isTest', type=int, default=1,help='whether to test, 1 for Yes, 0 for No')
    parser.add_argument("-p",'--isPrepare', type=int, default=1,help='whether to do data preparation, 1 for Yes, 0 for No')
    parser.add_argument("-s",'--isSummary', type=int, default=1,help='whether to do model summary, 1 for Yes, 0 for No')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args=getArgs()
    print(args)

    