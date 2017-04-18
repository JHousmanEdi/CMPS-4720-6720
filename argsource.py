import argparse


parser = argparse.ArgumentParser()
"""
Dataset Arguments
"""
parser.add_argument("--scaling", type=int, default=286, help="scale images to this size before cropping to 256x256")
parser.add_argument("--cropping", type=int, default=256, help="Size of image crop")
parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
#Image processing ends here
parser.add_argument("--batch", type=int, default=1, help="num images in batch")
parser.add_argument("--mode", default="export", choices=["train", "test", "export"]) #Add Required Later
parser.add_argument("--seed", type=int)
#Model processing ends here
parser.add_argument("--images", default="/home/jason/Documents/CMPS-4720-6720/Dataset/random_test", help="path to folder containing images")
parser.add_argument("--results_dir", default="/home/jason/Documents/CMPS-4720-6720/results/", help="where to put output files")
parser.add_argument("--checkpoint", default="/home/jason/Documents/CMPS-4720-6720/Training_Logs",
                    help="directory with checkpoint to resume training from or use for testing")
parser.add_argument("--log_dir", default = "/home/jason/Documents/CMPS-4720-6720/Training_Logs", help="Where log file(s) get saved")
parser.add_argument("--prog_dir", default="/home/jason/Documents/CMPS-4720-6720/results/", help="Where progress images are saved")
parser.add_argument("--initial_ckpt", default="/home/jason/Documents/CMPS-4720-6720/retouch_training", help= "where training is saved initially")
parser.add_argument("--personal_test", default="/home/jason/Documents/CMPS-4720-6720/Dataset/personal_images", help="where my personal images are")
parser.add_argument("--stand_alone", default="/home/jason/Documents/CMPS-4720-6720/Training_Logs/generator_only", help="Where generator model is")
#Directory processing ends here
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument('--beta1', type=float, default=0.5,   help='Beta for Adam, default=0.5')
parser.add_argument('--lr_d', type=float, default=0.0002, help='Learning rate for Critic, default=0.0002')
parser.add_argument('--lr_g', type=float, default=0.0002, help='Learning rate for Generator, default=0.0002')
parser.add_argument('--l2_weight', type=float, default=0.999, help='Weight for l2 loss, default=0.999')
#Model training parameters ends here

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=100, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=1000, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=2500, help="save model every save_freq steps, 0 to disable")

#Overarching training overview

parser.set_defaults(flip=True)
args = vars(parser.parse_args())
