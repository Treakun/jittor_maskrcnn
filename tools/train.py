import os
import yaml
import jittor as jt
from jittor import nn, optim
from models import MaskRCNN 
from dataset import COCODataset
from transforms import Compose, RandomHorizontalFlip, ToTensor, Normalize, Resize
from utils import train_one_epoch, evaluate
import jittor.lr_scheduler as lr_scheduler
from models.backbone import resnet50_fpn_backbone
import datetime
import math
if __name__ == '__main__':
   
    # Load configuration from YAML
    cfg_path = os.path.join("configs", "mask_rcnn_r50_fpn_1x.yaml")
    with open(cfg_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)

    # Convert nested dict to object for attribute access
    class Config:
        def __init__(self, **entries):
            for k, v in entries.items():
                if isinstance(v, dict):
                    v = Config(**v)
                setattr(self, k, v)

    cfg = Config(**cfg_dict)

    def create_model(num_classes, load_pretrain_weights=False):
        backbone = resnet50_fpn_backbone(
            pretrain_path="resnet50.pth"
        )
        model = MaskRCNN(backbone, num_classes=num_classes)

        if load_pretrain_weights:
            weights_dict = jt.load("./maskrcnn_resnet50_fpn_coco.pth")

            keys_to_remove = []
            for k in list(weights_dict.keys()):
                if ("box_predictor" in k) or ("mask_fcn_logits" in k):
                    keys_to_remove.append(k)

            for k in keys_to_remove:
                del weights_dict[k]

            # 加载预训练参数（jittor）
            model.load_parameters(weights_dict)
            print("Loaded pretrained COCO weights, except box_predictor and mask_fcn_logits")

        return model

    def make_cosine_with_warmup(optimizer, warmup_iters, total_iters, min_lr_ratio=0.01):
        """返回一个 jt.optim.LambdaLR，用作 per-iteration 的 scheduler"""
        import math as _math
        start_warmup_factor = 1.0 / 100.0  # 起始 lr = base_lr * start_warmup_factor
        def lr_lambda(step):
            if step < warmup_iters:
                return start_warmup_factor + (1.0 - start_warmup_factor) * (float(step) / float(max(1, warmup_iters)))
            else:
                progress = float(step - warmup_iters) / float(max(1, total_iters - warmup_iters))
                cos_val = 0.5 * (1.0 + _math.cos(_math.pi * progress))
                return min_lr_ratio + (1.0 - min_lr_ratio) * cos_val
        return jt.optim.LambdaLR(optimizer, lr_lambda)

    def main():
        # 设置是否使用 CUDA
        jt.flags.use_cuda = 1 if jt.has_cuda else 0
        print("Use CUDA:", bool(jt.flags.use_cuda))

        # 日志目录
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(cfg.OUTPUT_DIR, f"logs_{current_time}")
        os.makedirs(log_dir, exist_ok=True)

        train_log_path = os.path.join(log_dir, "train_log.txt")
        val_log_path = os.path.join(log_dir, "val_log.txt")
        with open(val_log_path, "w") as f:
            f.write("epoch\tdet_mAP\tseg_mAP\tmean_train_loss\n")

        # transforms
        transforms_train = Compose([
            Resize(min_size=800, max_size=1333),
            ToTensor(),
            RandomHorizontalFlip(0.5),
            Normalize()
        ])
        transforms_val = Compose([
            Resize(min_size=800, max_size=1333),
            ToTensor(),
            Normalize()
        ])

        train_dataset = COCODataset(
            cfg.DATA.TRAIN_IMG_DIR,
            cfg.DATA.TRAIN_ANN,
            is_train=True,
            transforms=transforms_train,
            remove_images_without_annotations=True,
            sample_fraction=getattr(cfg.DATA, "SAMPLE_FRACTION", 0.02),
            min_objects=2
        )

        val_dataset = COCODataset(
            cfg.DATA.VAL_IMG_DIR,
            cfg.DATA.VAL_ANN,
            is_train=False,
            transforms=transforms_val,
            remove_images_without_annotations=True,
            sample_fraction=getattr(cfg.DATA, "SAMPLE_FRACTION", 0.02),
            min_objects=2
        )

        train_loader = train_dataset.set_attrs(
            batch_size=cfg.SOLVER.BATCH_SIZE,
            shuffle=True,
            num_workers=getattr(cfg.SOLVER, "NUM_WORKERS", 8),
            drop_last=True
        )

        val_loader = val_dataset.set_attrs(
            batch_size=cfg.SOLVER.BATCH_SIZE,
            shuffle=False,
            num_workers=max(1, getattr(cfg.SOLVER, "NUM_WORKERS", 4))
        )

        # 创建模型
        model = create_model(num_classes=cfg.MODEL.NUM_CLASSES+1,
                             load_pretrain_weights=getattr(cfg.MODEL, "load_pretrain_weights", False))

        
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.SGD(
            params,
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY
        )

        

        train_dataset_size = len(train_dataset)

        batch_size = cfg.SOLVER.BATCH_SIZE
        steps_per_epoch = int(math.ceil(float(train_dataset_size) / float(batch_size)))
        total_iters = steps_per_epoch * cfg.SOLVER.MAX_EPOCHS
        warmup_iters = min(500, max(50, int(steps_per_epoch * 0.2)))
        min_lr_ratio = getattr(cfg.SOLVER, "MIN_LR_RATIO", 0.01)

        
        scheduler = make_cosine_with_warmup(optimizer, warmup_iters, total_iters, min_lr_ratio)
        print(f"Scheduler created: warmup_iters={warmup_iters}, total_iters={total_iters}, min_lr_ratio={min_lr_ratio}")

        train_loss_history = []
        learning_rate_history = []
        val_map_history = []
        start_epoch = 0
        global_step = 0 

        
        if getattr(cfg.SOLVER, "RESUME", None):
            resume_path = cfg.SOLVER.RESUME
            if os.path.exists(resume_path):
                print(f"Resuming training from checkpoint: {resume_path}")
                checkpoint = jt.load(resume_path)

                # model 参数恢复
                if "model" in checkpoint:
                    try:
                        model.load_parameters(checkpoint["model"])
                    except Exception as e:
                        print("Warning: failed to load model parameters with load_parameters(), trying state_dict() style. Err:", e)
                        try:
                            model.load_state_dict(checkpoint["model"])
                        except Exception:
                            pass

                # optimizer 恢复
                if "optimizer" in checkpoint:
                    try:
                        optimizer.load_state_dict(checkpoint["optimizer"])
                    except Exception as e:
                        print("Warning: failed to load optimizer state:", e)

                
                sched_info = checkpoint.get("scheduler", None)
                if sched_info and isinstance(sched_info, dict):
                    sched_type = sched_info.get("type", "")
                    if sched_type == "CosineWithWarmup":
                        # 重建 scheduler（参数以 checkpoint 中为准，fallback 到当前计算值）
                        w_iters = int(sched_info.get("warmup_iters", warmup_iters))
                        t_iters = int(sched_info.get("total_iters", total_iters))
                        m_ratio = float(sched_info.get("min_lr_ratio", min_lr_ratio))
                        scheduler = make_cosine_with_warmup(optimizer, w_iters, t_iters, m_ratio)
                        last_step = int(sched_info.get("last_step", 0))
                        # advance scheduler
                        if last_step > 0:
                            print(f"Advancing scheduler by {last_step} steps to recover state...")
                            for _ in range(last_step):
                                try:
                                    scheduler.step()
                                except Exception:
                                    pass
                        global_step = last_step
                   
                    else:
                        pass
                else:
                   
                    pass

                start_epoch = int(checkpoint.get("epoch", -1)) + 1
                train_loss_history = checkpoint.get("train_loss", [])
                learning_rate_history = checkpoint.get("learning_rate", [])
                val_map_history = checkpoint.get("val_map", [])
                
                global_step = int(checkpoint.get("scheduler", {}).get("last_step", global_step))
                print(f"Resumed at epoch {start_epoch}, global_step {global_step}")
            else:
                print(f"Warning: resume path {resume_path} does not exist. Starting from scratch.")

        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        # 训练循环
        for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS):
            print(f"\n=== Epoch {epoch + 1}/{cfg.SOLVER.MAX_EPOCHS} ===")

            # train_one_epoch 现在返回 (loss, lr, updated_global_step)
            mean_loss, lr, global_step = train_one_epoch(
                model,
                optimizer,
                train_loader,
                epoch,
                log_file=train_log_path,
                print_freq=cfg.SOLVER.PRINT_FREQ,
                warmup=False,                # 用 global scheduler 管理 warmup
                global_scheduler=scheduler,
                start_global_step=global_step
            )

            train_loss_history.append(mean_loss)
            learning_rate_history.append(lr)

            # 保存 checkpoint（记录 scheduler 的 type 与 last_step）
            save_files = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": {
                    "type": "CosineWithWarmup",
                    "warmup_iters": warmup_iters,
                    "total_iters": total_iters,
                    "min_lr_ratio": min_lr_ratio,
                    "last_step": int(global_step)
                },
                "epoch": epoch,
                "train_loss": train_loss_history,
                "learning_rate": learning_rate_history,
                "val_map": val_map_history
            }

            checkpoint_path = os.path.join(cfg.OUTPUT_DIR, f"model_{epoch}.pkl")
            jt.save(save_files, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

            # 验证
            model.eval()
            det_info, seg_info = evaluate(model, val_loader)
            model.train()

            # 记录验证结果，使用 COCO stats[0] 作为主 mAP
            with open(val_log_path, "a") as f:
                if det_info and seg_info:
                    det_map = float(det_info[0])
                    seg_map = float(seg_info[0])
                    val_map_history.append((det_map, seg_map))
                    f.write(f"{epoch}\t{det_map:.4f}\t{seg_map:.4f}\t{mean_loss:.6f}\n")
                else:
                    f.write(f"{epoch}\t-\t-\t{mean_loss:.6f}\n")

        # 保存最终模型
        final_model_path = os.path.join(cfg.OUTPUT_DIR, 'maskrcnn_final.pkl')
        jt.save(model.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")

    main()