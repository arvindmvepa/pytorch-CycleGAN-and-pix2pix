# Simple script to make sure basic usage
# such as training, testing, saving and loading
# runs without errors.
import os


def run(command):
    print(command)
    exit_status = os.system(command)
    if exit_status > 0:
        exit(1)


if __name__ == '__main__':
    # download mini datasets
    if not os.path.exists('./datasets/mini'):
        run('bash ./datasets/download_cyclegan_dataset.sh mini')

    # pretrained cyclegan model
    if not os.path.exists('./checkpoints/horse2zebra_pretrained/latest_net_G.pth'):
        run('bash ./scripts/download_cyclegan_model.sh horse2zebra')
    run('python test.py --model test --dataroot ./datasets/mini --name horse2zebra_pretrained --no_dropout --num_test 1 --no_dropout')

    # cyclegan train/test
    run('python train.py --model cycle_gan --name temp_cyclegan --dataroot ./datasets/mini --niter 1 --niter_decay 0 --save_latest_freq 10  --print_freq 1 --display_id -1')
    run('python test.py --model test --name temp_cyclegan --dataroot ./datasets/mini --num_test 1 --model_suffix "_A" --no_dropout')

    # template train/test
    run('python train.py --model template --name temp2 --dataroot ./datasets/mini_pix2pix --niter 1 --niter_decay 0 --save_latest_freq 10 --display_id -1')
    run('python test.py --model template --name temp2 --dataroot ./datasets/mini_pix2pix --num_test 1')