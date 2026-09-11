"""Orchestrates the full pipeline: `series_encoder -> cross_series_attention
-> study_encoder -> pathology_attention -> task_heads`, plus an optional
auxiliary report branch, all built from one config's `model.*` flags.

M1 through M7 are the SAME class with different flags -- never a different
model file per experiment. See `configs/base.yaml` for the flag schema and
`README.md`'s M1..M7 table.

**Inference contract**: `forward(batch, report_texts=None)` -- passing
`None` (the default) means the report branch is never invoked, even on a
checkpoint trained with `report_weak_supervision`/`distillation` enabled.
This is what guarantees the model is fully MRI+metadata-driven at test time;
see `tests/test_models_shapes.py::test_report_teacher_never_called_when_report_texts_is_none`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.cross_series_attention import CrossSeriesTransformer
from src.models.metadata_encoder import SeriesMetadataEncoder
from src.models.pathology_attention import PathologyAttentionModule
from src.models.report_teacher import ReportTeacher
from src.models.series_encoder import SeriesEncoder
from src.models.slice_attention import MaskedAttentionPool
from src.models.slice_encoder import SliceEncoder
from src.models.study_encoder import StudyEncoder
from src.models.task_heads import TaskHeads
from src.models.text_encoder import ReportTextEncoder


class KneeModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        embedding_dim = m["embedding_dim"]

        self.slice_encoder = SliceEncoder(
            backbone=m["backbone"], embedding_dim=embedding_dim,
            pretrained=m["pretrained"], in_channels=m["in_channels"],
        )
        self.slice_attention = MaskedAttentionPool(embedding_dim, enabled=m["slice_attention"]["enabled"])

        metadata_cfg = m["metadata"]
        self.metadata_encoder = (
            SeriesMetadataEncoder(embedding_dim=metadata_cfg["embedding_dim"], output_dim=embedding_dim)
            if metadata_cfg["enabled"] else None
        )
        self.series_encoder = SeriesEncoder(self.slice_encoder, self.slice_attention, self.metadata_encoder, embedding_dim)

        csa_cfg = m["cross_series_attention"]
        self.cross_series_attention = CrossSeriesTransformer(
            embedding_dim, enabled=csa_cfg["enabled"], num_heads=csa_cfg["num_heads"],
            num_layers=csa_cfg["num_layers"], dropout=csa_cfg["dropout"],
        )

        self.study_encoder = StudyEncoder(embedding_dim, pooling=m["study_pooling"])

        path_cfg = m["pathology_attention"]
        self.pathology_attention = PathologyAttentionModule(
            embedding_dim, enabled=path_cfg["enabled"], bottleneck_ratio=path_cfg["bottleneck_ratio"],
        )

        self.task_heads = TaskHeads(embedding_dim)

        needs_text = m["report_weak_supervision"]["enabled"] or m["distillation"]["enabled"]
        self.report_teacher: ReportTeacher | None = None
        if needs_text:
            text_cfg = m["text_encoder"]
            text_encoder = ReportTextEncoder(
                model_name=text_cfg["model_name"], embedding_dim=embedding_dim,
                freeze_backbone=text_cfg["freeze_backbone"], max_length=text_cfg["max_length"],
            )
            self.report_teacher = ReportTeacher(text_encoder, embedding_dim)

    def forward(self, batch: dict, report_texts: list[str] | None = None) -> dict:
        series_emb = self.series_encoder(
            batch["pixel_values"], batch["slice_mask"], batch["plane_id"], batch["fluid_sensitive"], batch["fat_suppression"],
        )
        series_emb = self.cross_series_attention(series_emb, batch["series_mask"])
        study_emb = self.study_encoder(series_emb, batch["series_mask"])
        group_embeddings = self.pathology_attention(study_emb)
        logits = self.task_heads(group_embeddings)

        teacher_logits = None
        if self.report_teacher is not None and report_texts is not None:
            teacher_logits = self.report_teacher(report_texts)

        return {"logits": logits, "teacher_logits": teacher_logits, "study_embedding": study_emb}
