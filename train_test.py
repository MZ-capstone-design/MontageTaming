import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel
from taming.data.custom import CustomTrain, CustomTest
from main import DataModuleFromConfig

def main(config_path):
    # 설정 파일 로드
    config = OmegaConf.load(config_path)
    
    # 데이터 모듈 생성
    data_module = DataModuleFromConfig(**config.data.params)
    
    # 모델 초기화
    model = VQModel(**config.model.params)
    
    # 체크포인트 콜백 설정
    checkpoint_callback = ModelCheckpoint(
        monitor='val/rec_loss',
        mode='min',
        save_top_k=1,
        dirpath='checkpoints',
        filename='vqgan-{epoch:02d}-{val/rec_loss:.2f}'
    )
    
    # Trainer 설정
    trainer = Trainer(
        gpus=config.get('gpus', 0),
        callbacks=[checkpoint_callback],
        max_epochs=50,  # 원하는 epoch 수로 설정
        precision=16,  # Mixed precision 학습을 위한 설정 (선택 사항)
    )
    
    # 모델 학습
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train VQGAN Model")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()
    main(args.config)
