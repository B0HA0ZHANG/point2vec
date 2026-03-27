from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from point2vec.models.point2vec import Point2Vec
from point2vec.modules.cross_transformer import HybridCrossTransformerEncoder


class Point2VecPCPRefiner(Point2Vec):
    def __init__(
        self,
        tokenizer_num_groups: int = 64,
        tokenizer_group_size: int = 32,
        tokenizer_group_radius: float | None = None,
        d2v_masking_ratio: float = 0.65,
        d2v_masking_type: str = "rand",
        encoder_dim: int = 384,
        encoder_depth: int = 12,
        encoder_heads: int = 6,
        encoder_dropout: float = 0,
        encoder_attention_dropout: float = 0.05,
        encoder_drop_path_rate: float = 0.25,
        encoder_add_pos_at_every_layer: bool = True,
        decoder: bool = False,
        decoder_depth: int = 4,
        decoder_dropout: float = 0,
        decoder_attention_dropout: float = 0.05,
        decoder_drop_path_rate: float = 0.25,
        decoder_add_pos_at_every_layer: bool = True,
        d2v_target_layers: List[int] = [6, 7, 8, 9, 10, 11],
        d2v_target_layer_part: str = "final",
        d2v_target_layer_norm: Optional[str] = "layer",
        d2v_target_norm: Optional[str] = "layer",
        d2v_ema_tau_max: Optional[float] = 0.9998,
        d2v_ema_tau_min: Optional[float] = 0.99999,
        d2v_ema_tau_epochs: int = 200,
        loss: str = "smooth_l1",
        learning_rate: float = 1e-3,
        cls_ce_lambda: float = 0.5,
        cls_ce_temperature_s: float = 0.1,
        cls_ce_temperature_t: float = 0.04,
        optimizer_adamw_weight_decay: float = 0.05,
        lr_scheduler_linear_warmup_epochs: int = 80,
        lr_scheduler_linear_warmup_start_lr: float = 1e-6,
        lr_scheduler_cosine_eta_min: float = 1e-6,
        train_transformations: List[str] = [
            "subsample",
            "scale",
            "center",
            "unit_sphere",
            "rotate",
        ],
        val_transformations: List[str] = ["subsample", "center", "unit_sphere"],
        transformation_subsample_points: int = 1024,
        transformation_scale_min: float = 0.8,
        transformation_scale_max: float = 1.2,
        transformation_scale_symmetries: Tuple[int, int, int] = (1, 0, 1),
        transformation_rotate_dims: List[int] = [1],
        transformation_rotate_degs: Optional[int] = None,
        transformation_translate: float = 0.2,
        transformation_height_normalize_dim: int = 1,
        svm_validation: Dict[str, pl.LightningDataModule] = {},
        svm_validation_C=0.012,
        fix_estimated_stepping_batches: Optional[int] = None,
        pcp_pos_prediction_lambda: float = 1.0,
        pcp_pos_prediction_loss: str = "mse",
        pcp_detach_predicted_pos: bool = True,
        pcp_refinement_depth: int = 2,
        pcp_refinement_dropout: float = 0.0,
        pcp_refinement_attention_dropout: float = 0.05,
        pcp_refinement_drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__(
            tokenizer_num_groups=tokenizer_num_groups,
            tokenizer_group_size=tokenizer_group_size,
            tokenizer_group_radius=tokenizer_group_radius,
            d2v_masking_ratio=d2v_masking_ratio,
            d2v_masking_type=d2v_masking_type,
            encoder_dim=encoder_dim,
            encoder_depth=encoder_depth,
            encoder_heads=encoder_heads,
            encoder_dropout=encoder_dropout,
            encoder_attention_dropout=encoder_attention_dropout,
            encoder_drop_path_rate=encoder_drop_path_rate,
            encoder_add_pos_at_every_layer=encoder_add_pos_at_every_layer,
            decoder=False,
            decoder_depth=decoder_depth,
            decoder_dropout=decoder_dropout,
            decoder_attention_dropout=decoder_attention_dropout,
            decoder_drop_path_rate=decoder_drop_path_rate,
            decoder_add_pos_at_every_layer=decoder_add_pos_at_every_layer,
            d2v_target_layers=d2v_target_layers,
            d2v_target_layer_part=d2v_target_layer_part,
            d2v_target_layer_norm=d2v_target_layer_norm,
            d2v_target_norm=d2v_target_norm,
            d2v_ema_tau_max=d2v_ema_tau_max,
            d2v_ema_tau_min=d2v_ema_tau_min,
            d2v_ema_tau_epochs=d2v_ema_tau_epochs,
            loss=loss,
            learning_rate=learning_rate,
            cls_ce_lambda=cls_ce_lambda,
            cls_ce_temperature_s=cls_ce_temperature_s,
            cls_ce_temperature_t=cls_ce_temperature_t,
            optimizer_adamw_weight_decay=optimizer_adamw_weight_decay,
            lr_scheduler_linear_warmup_epochs=lr_scheduler_linear_warmup_epochs,
            lr_scheduler_linear_warmup_start_lr=lr_scheduler_linear_warmup_start_lr,
            lr_scheduler_cosine_eta_min=lr_scheduler_cosine_eta_min,
            train_transformations=train_transformations,
            val_transformations=val_transformations,
            transformation_subsample_points=transformation_subsample_points,
            transformation_scale_min=transformation_scale_min,
            transformation_scale_max=transformation_scale_max,
            transformation_scale_symmetries=transformation_scale_symmetries,
            transformation_rotate_dims=transformation_rotate_dims,
            transformation_rotate_degs=transformation_rotate_degs,
            transformation_translate=transformation_translate,
            transformation_height_normalize_dim=transformation_height_normalize_dim,
            svm_validation=svm_validation,
            svm_validation_C=svm_validation_C,
            fix_estimated_stepping_batches=fix_estimated_stepping_batches,
        )
        self.save_hyperparameters()

        if decoder:
            raise ValueError(
                "Point2VecPCPRefiner is encoder-only. Leave decoder=False and use pcp_refinement_depth instead."
            )

        init_std = 0.02
        self.mask_query_token = nn.Parameter(torch.zeros(encoder_dim))
        self.mask_pos_token = nn.Parameter(torch.zeros(encoder_dim))
        nn.init.trunc_normal_(
            self.mask_query_token, mean=0, std=init_std, a=-init_std, b=init_std
        )
        nn.init.trunc_normal_(
            self.mask_pos_token, mean=0, std=init_std, a=-init_std, b=init_std
        )

        dpr = [
            x.item() for x in torch.linspace(0, encoder_drop_path_rate, encoder_depth)
        ]
        self.student = HybridCrossTransformerEncoder(
            embed_dim=encoder_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            qkv_bias=True,
            drop_rate=encoder_dropout,
            attn_drop_rate=encoder_attention_dropout,
            drop_path_rate=dpr,
            add_pos_at_every_layer=encoder_add_pos_at_every_layer,
            cross_depth=pcp_refinement_depth,
        )
        self.pred_pos_proj = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.LayerNorm(encoder_dim),
            nn.GELU(),
            nn.Linear(encoder_dim, encoder_dim),
        )
        self.regressor = nn.Linear(encoder_dim, encoder_dim)

    def forward(
        self,
        embeddings: torch.Tensor,
        centers: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, _, C = embeddings.shape
        pos = self.positional_encoding(centers)

        w = mask.unsqueeze(-1).type_as(embeddings)
        corrupted_embeddings = (1 - w) * embeddings + w * self.mask_token

        visible_embeddings = corrupted_embeddings[~mask].reshape(B, -1, C)
        visible_pos = pos[~mask].reshape(B, -1, C)
        masked_pos = pos[mask].reshape(B, -1, C)
        num_masked = masked_pos.shape[1]

        masked_queries = self.mask_query_token.reshape(1, 1, C).expand(
            B, num_masked, -1
        )
        masked_query_pos = self.mask_pos_token.reshape(1, 1, C).expand(B, num_masked, -1)

        student_tokens, student_pos = self.append_cls_token(
            visible_embeddings,
            visible_pos,
        )
        student_output = self.student(
            student_tokens,
            student_pos,
            masked_queries,
            masked_query_pos,
        )
        student_encoded = student_output.last_hidden_state
        student_cls = student_output.cls_hidden_state
        assert student_cls is not None

        visible_count = student_tokens.shape[1]
        masked_query_embeddings = student_encoded[:, visible_count:]
        predicted_masked_pos = self.pred_pos_proj(masked_query_embeddings)

        prediction_pos = predicted_masked_pos.detach()
        if not self.hparams.pcp_detach_predicted_pos:  # type: ignore
            prediction_pos = predicted_masked_pos
        predictions = self.regressor((masked_query_embeddings + prediction_pos).reshape(-1, C))

        targets_full, teacher_cls = self.generate_targets(embeddings, pos)
        targets = targets_full[mask]
        cls_ce_loss = self.cls_distillation_loss(student_cls, teacher_cls)

        match self.hparams.pcp_pos_prediction_loss:  # type: ignore
            case "mse":
                pos_loss = F.mse_loss(predicted_masked_pos, targets.detach().reshape(B, num_masked, C))
            case "smooth_l1":
                pos_loss = F.smooth_l1_loss(
                    predicted_masked_pos,
                    targets.detach().reshape(B, num_masked, C),
                )
            case _:
                raise ValueError(
                    f"Unknown position prediction loss: {self.hparams.pcp_pos_prediction_loss}"  # type: ignore
                )

        return predictions, targets, cls_ce_loss, pos_loss

    def _perform_step(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, centers = self.tokenizer(inputs)
        mask = self.masking(centers)
        return self.forward(tokens, centers, mask)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        points, _ = batch
        points = self.train_transformations(points)

        x, y, cls_ce_loss, pos_loss = self._perform_step(points)
        ssl_loss = self.loss_func(x, y)
        loss = (
            ssl_loss
            + self.cls_ce_lambda * cls_ce_loss
            + self.hparams.pcp_pos_prediction_lambda * pos_loss  # type: ignore
        )

        self.log("train/ssl_loss", ssl_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train/cls_ce_loss",
            cls_ce_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/pos_loss", pos_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/pred_std", self.token_std(x))
        self.log("train/target_std", self.token_std(y))
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        points, _ = batch
        points = self.val_transformations(points)

        x, y, cls_ce_loss, pos_loss = self._perform_step(points)
        ssl_loss = self.loss_func(x, y)
        loss = (
            ssl_loss
            + self.cls_ce_lambda * cls_ce_loss
            + self.hparams.pcp_pos_prediction_lambda * pos_loss  # type: ignore
        )

        self.log("val/ssl_loss", ssl_loss, on_epoch=True, prog_bar=True)
        self.log("val/cls_ce_loss", cls_ce_loss, on_epoch=True, prog_bar=True)
        self.log("val/pos_loss", pos_loss, on_epoch=True, prog_bar=True)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/pred_std", self.token_std(x))
        self.log("val/target_std", self.token_std(y))

    def svm_validation(self, datamodule: pl.LightningDataModule) -> Tuple[float, float]:
        assert not self.training
        assert not torch.is_grad_enabled()

        def xy(
            dataloader: DataLoader,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            x_list = []
            label_list = []
            for (points, label) in iter(dataloader):
                points = points.cuda()
                label = label.cuda()
                points = self.val_transformations(points)
                embeddings, centers = self.tokenizer(points)
                pos = self.positional_encoding(centers)
                tokens, pos = self.append_cls_token(embeddings, pos)
                output = self.student(tokens, pos)
                encoded = output.last_hidden_state
                cls_feature = output.cls_hidden_state
                assert cls_feature is not None
                patch_tokens = encoded[:, 1:]
                max_feature = patch_tokens.max(dim=1).values
                mean_feature = patch_tokens.mean(dim=1)
                x = torch.cat([cls_feature, max_feature, mean_feature], dim=-1)
                x_list.append(x.cpu())
                label_list.append(label.cpu())

            return torch.cat(x_list, dim=0), torch.cat(label_list, dim=0)

        x_train, y_train = xy(datamodule.train_dataloader())  # type: ignore
        x_val, y_val = xy(datamodule.val_dataloader())  # type: ignore

        svm_C: float = self.hparams.svm_validation_C  # type: ignore
        svm = self._build_svm(svm_C)
        svm.fit(x_train, y_train)  # type: ignore
        train_acc: float = svm.score(x_train, y_train)  # type: ignore
        val_acc: float = svm.score(x_val, y_val)  # type: ignore
        return train_acc, val_acc

    @staticmethod
    def _build_svm(C: float):
        from sklearn.svm import SVC

        return SVC(C=C, kernel="linear")
