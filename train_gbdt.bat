python train.py --arch="mobilenet_1" ^
    --start-epoch=1 ^
    --gbdt=1 ^
    --alpha=0.2 ^
    --loss=wpdc ^
    --snapshot="snapshot/phase1_wpdc" ^
    --param-fp-train='D:/data/facealignment/train.configs/param_all_norm.pkl' ^
    --param-fp-val='D:/data/facealignment/train.configs/param_all_norm_val.pkl' ^
    --warmup=5 ^
    --resume=models/phase1_wpdc.pth.tar ^
    --opt-style=resample ^
    --resample-num=132 ^
    --batch-size=32 ^
    --base-lr=0.02 ^
    --epochs=1 ^
    --milestones=30,40 ^
    --print-freq=50 ^
    --devices-id=0 ^
    --workers=8 ^
    --filelists-train="D:/data/facealignment/train.configs/train_aug_120x120.list.train" ^
    --filelists-val="D:/data/facealignment/train.configs/train_aug_120x120.list.val" ^
    --root="D:\data\facealignment\train_aug_120x120" 