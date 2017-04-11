import argparse


parser = argparse.ArgumentParser()
"""
Dataset Arguments
"""
parser.add_argument("--images", default="/home/jason/Documents/CMPS-4720-6720/Dataset/ExpB128", help="path to folder containing images")
parser.add_argument("--scaling", type=int, default=286, help="scale images to this size before cropping to 256x256")
parser.add_argument("--batch", type=int, default=1, help="num images in batch")
parser.add_argument("--mode", default="train", choices=["train", "test", "export"]) #Add Required Later
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.add_argument("--results_dir", default = "/home/jason/Documents/CMPS-4720-6720/Results", help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default='/home/jason/Documents/CMPS-4720-6720/Checkpoints',
                    help="directory with checkpoint to resume training from or use for testing")
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument('--beta1', type=float, default=0.5,   help='Beta for Adam, default=0.5')
parser.add_argument('--lr_d', type=float, default=0.0002, help='Learning rate for Critic, default=0.0002')
parser.add_argument('--lr_g', type=float, default=0.002, help='Learning rate for Generator, default=0.002')
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--cropping", type=int, default=256, help="Size of image crop")
parser.add_argument('--l2_weight', type=float, default=0.999, help='Weight for l2 loss, default=0.999')


parser.set_defaults(flip=True)
args = vars(parser.parse_args())