# if train_or_eval = True then 训练 else 测试
train_or_eval = False
# train_or_eval =

if train_or_eval is not True:
    # 测试的配置
    print('test......')
    task = 'denoising'
    # pathlistfile = r'las.txt'  # 测试的图片的具体路径
    dataset_dir = r'D:\Users\mxy\Data\J4R_old\frames_heavy_test_JPEG'  # 测试图片包括边缘图的路径
    dataset_gtc_dir = r'D:\Users\mxy\Data\J4R_old\frames_heavy_test_JPEG'
    # dataset_dir = r'E:\rainoutput\Data\Las\rain'  # 测试图片包括边缘图的路径
    # dataset_gtc_dir = r'E:\rainoutput\Las\rain'
    # dataset_dir = r'E:\rainoutput\Data\Dataset_Testing_Synthetic'  # 测试图片包括边缘图的路径
    # dataset_gtc_dir = r'E:\rainoutput\Data\Dataset_Testing_Synthetic'
    # # 相对应的gtc路径(使用了训练集中的gtc)，所有设置的路径都是00006/1这种文件夹的父文件夹才可以
    # pathlistfile = r'ntu_test.txt'  # 测试的图片的具体路径
    # dataset_dir = r'E:\rainoutput\Data\frames_syn_test_JPEG'  # 测试图片包括边缘图的路径
    # dataset_gtc_dir = r'E:\rainoutput\Data\frames_syn_test_JPEG'
    # # # # # 相对应的gtc路径(使用了训练集中的gtc)，所有设置的路径都是00006/1这种文件夹的父文件夹才可以
    pathlistfile = r'D:\Users\mxy\Data\J4R_old\test_light.txt'  # 测试的图片的具体路径
    # dataset_dir = r'D:\Users\mxy\Data\Ours\vimeo_test_clean\rained'  # 测试图片包括边缘图的路径
    # dataset_gtc_dir = r'D:\Users\mxy\Data\Ours\vimeo_test_clean\rained'
    # # # 相对应的gtc路径(使用了训练集中的gtc)，所有设置的路径都是00006/1这种文件夹的父文件夹才可以
    # pathlistfile = r'D:\Users\mxy\Data\Ours\vimeo_test_clean\sep_testlist_test.txt'  # 测试的图片的具体路径
    # dataset_dir = r'C:\Users\mxy\Desktop\RainStreakGen-master\out'
    # dataset_gtc_dir = r'E:\mxy\Data\youtube-dl_EXE\Harmonic Hong Kong 4K\images'
    # pathlistfile = './train/harmonic.txt'
    out_img_dir = './test3'  # 实验结果存放位置
    model_path = './ckpt_tip_thh/checkpoints_39epoch.ckpt'  # 1maxoper的新模型
    gpuID =0# map_location='cuda:1' 在evaluate.py里设置
    map_location = 'cuda:0'
    BATCH_SIZE = 1
    h = 500     # 500 报错
    w = 888     # 889 报错
    N = 3 # 5张图片

else:
    # 训练的配置
    print('train......')
    mode = 'train'
    task = 'denoising'
    edited_img_dir = r'D:\Users\mxy\Data\J4R_old\heavy_train/'  # 训练输入的图片的文件夹
    dataset_dir = r'D:\Users\mxy\Data\J4R_old\heavy_train/'
    # edited_img_dir = r'E:\G\Data\J4R_old/heavy_train'  # 训练输入的图片的文件夹
    # dataset_dir = r'E:\G\Data\J4R_old/heavy_train'  # gtc图片的文件夹，实际我们将对应的gtc与输入图片放在了一起，最好分开
    # pathlistfile = r'E:\G\Data\J4R_old/train_heavy.txt'  # 训练的图片的具体路径
    pathlistfile = r'D:\Users\mxy\Data\J4R_old\train_heavy.txt'
    # edited_img_dir = r'D:\Users\mxy\Data\Ours\vimeo_test_clean\rained'  # 训练输入的图片的文件夹
    # dataset_dir = r'D:\Users\mxy\Data\Ours\vimeo_test_clean\rained'
    # pathlistfile = r'D:\Users\mxy\Data\Ours\vimeo_test_clean\sep_testlist_train.txt'

    visualize_root = './ckpt_tip_thh/'  # 存放展示结果的文件夹
    visualize_pathlist = ['00001/120']  # 需要展示训练结果的训练图片所在的小文件夹
    checkpoints_root = './ckpt_tip_thh'  # 训练过程中产生的检查点的存放位置
    model_besaved_root = 'ckpt_tip_thh'  # best_model 和 final_model 的参数的保存位置
    model_best_name = '_best.ckpt'
    model_final_name = '_final.ckpt'
    gpuID = 2
    print('gpuID: ', gpuID)

    # Hyper Parameters
    if task == 'interp':
        LR = 3 * 1e-5
    elif task in ['denoise', 'denoising', 'sr', 'super-resolution']:
        # LR = 1 * 1e-5
        LR = 0.00009

    EPOCH = 60
    WEIGHT_DECAY = 5e-5
    BATCH_SIZE = 1
    LR_strategy = []
    h = 240
    w = 240  # 不起作用
    # h = 320
    # w = 320
    N = 5  # 输入7张图片   不起作用
    sample_frames = 5  # 一个视频采样的帧数
    scale_min = 0.4
    scale_max = 2
    crop_size = 240  # 起作用  裁剪大小，raft 最好是4的倍数
    size_multiplier = 2 ** 6    # ?
    geometry_aug = True
    order_aug = True

    ssim_weight = 1.1
    l1_loss_weight =0.75
    w_VGG = 0
    w_ST = 1
    w_LT = 0
    alpha = 50

    use_checkpoint = False  # 一开始不使用已有的检查点
    checkpoint_exited_path = 'toflow_models_mine/checkpoints_light.ckpt'  # 已有的检查点
    work_place = '.'
    model_name = task
    model_houzhui = '.pkl'
    Training_pic_path = 'ckpt_tip_th/Training_result_mine_maxoper.jpg'
    model_information_txt = model_name + '_information.txt'
