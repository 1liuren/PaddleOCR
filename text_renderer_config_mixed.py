
import os
from pathlib import Path
from text_renderer.config import (
    GeneratorCfg,
    RenderCfg,
)
from text_renderer.corpus import EnumCorpus, EnumCorpusCfg
from text_renderer.effect import Effects, Line, DropoutRand, DropoutVertical, OneOf

def mixed_data():
    return GeneratorCfg(
        num_image=5,
        save_dir=Path(r"E:/gitlab_open/PaddleOCR/synthetic_test_output"),
        render_cfg=RenderCfg(
            bg_dir=Path(r"E:/gitlab_open/PaddleOCR/text_renderer/example_data/bg"),
            corpus=EnumCorpus(
                EnumCorpusCfg(
                    text_paths=[Path(r"E:/gitlab_open/PaddleOCR/mixed_corpus.txt")],
                    filter_by_chars=False,
                    font_dir=Path(r"C:/Windows/Fonts"),
                    font_list_file=Path(r"E:/gitlab_open/PaddleOCR/font_list_windows.txt"),
                    font_size=(32, 48),
                    char_spacing=(-0.1, 0.1),
                ),
            ),
            corpus_effects=Effects(
                [
                    Line(0.5),
                    # OneOf([DropoutRand(), DropoutVertical()]),
                ]
            ),
            gray=False,
        ),
    )

configs = [mixed_data()]
