import inspect
import os
from pathlib import Path

from text_renderer.config import (
    FixedTextColorCfg,
    GeneratorCfg,
    NormPerspectiveTransformCfg,
    RenderCfg,
)
from text_renderer.corpus import CharCorpus, CharCorpusCfg, WordCorpus, WordCorpusCfg
from text_renderer.effect import (
    Effects, Line, DropoutRand, DropoutVertical, Padding, NoEffects, OneOf,
    Noise, GaussianBlur, SaltPepperNoise, Curve, Emboss, MotionBlur
)
from text_renderer.layout.same_line import SameLineLayout

CURRENT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
OUT_DIR = CURRENT_DIR / "output"
DATA_DIR = CURRENT_DIR
BG_DIR = DATA_DIR / "bg"
CHAR_DIR = DATA_DIR / "char"
FONT_DIR = DATA_DIR / "font"
FONT_LIST_DIR = DATA_DIR / "font_list"
TEXT_DIR = DATA_DIR / "text"

# Default Chinese/English font config
chn_font_cfg = dict(
    font_dir=FONT_DIR,
    font_list_file=FONT_LIST_DIR / "font_list.txt",
    font_size=(30, 31),
)

# Tibetan font config
tibetan_font_cfg = dict(
    font_dir=FONT_DIR,
    font_list_file=FONT_LIST_DIR / "tibetan_font_list.txt",
    font_size=(30, 31),
)

perspective_transform = NormPerspectiveTransformCfg(20, 20, 1.1)

def base_cfg(
    name: str, corpus, corpus_effects=None, layout_effects=None, layout=None, gray=True, num_image=1000
):
    return GeneratorCfg(
        num_image=num_image,
        save_dir=OUT_DIR / name,
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            perspective_transform=perspective_transform,
            gray=gray,
            layout_effects=layout_effects,
            layout=layout,
            corpus=corpus,
            corpus_effects=corpus_effects,
        ),
    )

def chn_data():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=CharCorpus(
            CharCorpusCfg(
                text_paths=[TEXT_DIR / "chn_text.txt"],
                filter_by_chars=True,
                chars_file=CHAR_DIR / "chn.txt",
                length=(5, 10),
                char_spacing=(-0.3, 1.3),
                **chn_font_cfg
            ),
        ),
        corpus_effects=Effects(
            [
                Line(0.5, color_cfg=FixedTextColorCfg()),
                OneOf([DropoutRand(), DropoutVertical()]),
            ]
        ),
    )

def eng_data():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=WordCorpus(
            WordCorpusCfg(
                text_paths=[TEXT_DIR / "eng_text.txt"],
                filter_by_chars=True,
                chars_file=CHAR_DIR / "eng.txt",
                **chn_font_cfg
            ),
        ),
    )

def tibetan_data():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=CharCorpus(
            CharCorpusCfg(
                text_paths=[TEXT_DIR / "tibetan_real_text.txt"],
                # Assuming tibetan.txt exists or we disable filter_by_chars
                filter_by_chars=True, 
                chars_file=CHAR_DIR / "tibetan.txt",
                length=(5, 10),
                char_spacing=(-0.3, 1.3),
                **tibetan_font_cfg
            ),
        ),
    )

# 1. 藏汉混排
def tibetan_chn_mixed():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        layout=SameLineLayout(),
        gray=False,
        num_image=10000,
        corpus=[
            CharCorpus(
                CharCorpusCfg(
                    text_paths=[TEXT_DIR / "chn_rich.txt"],
                    filter_by_chars=True,
                    chars_file=CHAR_DIR / "chn_rich_chars.txt",
                    length=(45, 55),
                    **chn_font_cfg
                ),
            ),
            CharCorpus(
                CharCorpusCfg(
                    text_paths=[TEXT_DIR / "tibetan_real_text.txt"],
                    filter_by_chars=True,
                    chars_file=CHAR_DIR / "tibetan.txt",
                    length=(45, 55),
                    **tibetan_font_cfg
                ),
            ),
        ],
        corpus_effects=[Effects([Padding()]), Effects([Padding()])],
        layout_effects=Effects(Line(p=0.2)),
    )

# 2. 藏汉英混排 (原 mixed_line_data)
def tibetan_chn_eng_mixed():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        layout=SameLineLayout(),
        gray=False,
        num_image=10000,
        corpus=[
            CharCorpus(
                CharCorpusCfg(
                    text_paths=[TEXT_DIR / "chn_rich.txt"],
                    filter_by_chars=True,
                    chars_file=CHAR_DIR / "chn_rich_chars.txt",
                    length=(30, 35),
                    **chn_font_cfg
                ),
            ),
            CharCorpus(
                CharCorpusCfg(
                    text_paths=[TEXT_DIR / "tibetan_real_text.txt"],
                    filter_by_chars=True,
                    chars_file=CHAR_DIR / "tibetan.txt",
                    length=(30, 35),
                    **tibetan_font_cfg
                ),
            ),
            WordCorpus(
                WordCorpusCfg(
                    text_paths=[TEXT_DIR / "eng_rich.txt"],
                    filter_by_chars=True,
                    chars_file=CHAR_DIR / "eng_rich_chars.txt",
                    num_word=(5, 8),
                    **chn_font_cfg
                ),
            ),
        ],
        corpus_effects=[Effects([Padding()]), Effects([Padding()]), NoEffects()],
        layout_effects=Effects(Line(p=0.2)),
    )

# 3. 噪声很强的合成藏汉英脏数据
def tibetan_chn_eng_dirty():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        layout=SameLineLayout(),
        gray=False,
        corpus=[
            CharCorpus(
                CharCorpusCfg(
                    text_paths=[TEXT_DIR / "chn_rich.txt"],
                    filter_by_chars=True,
                    chars_file=CHAR_DIR / "chn_rich_chars.txt",
                    length=(30, 35),
                    **chn_font_cfg
                ),
            ),
            CharCorpus(
                CharCorpusCfg(
                    text_paths=[TEXT_DIR / "tibetan_real_text.txt"],
                    filter_by_chars=True,
                    chars_file=CHAR_DIR / "tibetan.txt",
                    length=(30, 35),
                    **tibetan_font_cfg
                ),
            ),
            WordCorpus(
                WordCorpusCfg(
                    text_paths=[TEXT_DIR / "eng_rich.txt"],
                    filter_by_chars=True,
                    chars_file=CHAR_DIR / "eng_rich_chars.txt",
                    num_word=(5, 8),
                    **chn_font_cfg
                ),
            ),
        ],
        # 对每个语料应用更强的 Corpus Effect
        corpus_effects=[
            Effects([Padding(), OneOf([DropoutRand(p=0.5), DropoutVertical(p=0.5)])]), 
            Effects([Padding(), OneOf([DropoutRand(p=0.5), DropoutVertical(p=0.5)])]), 
            Effects([OneOf([DropoutRand(p=0.5), DropoutVertical(p=0.5)])])
        ],
        # 布局层应用强噪声和模糊
        layout_effects=Effects([
            Line(p=0.5),
            OneOf([
                GaussianBlur(blur_limit=(3, 5)),
                MotionBlur(blur_limit=(3, 5)),
                Noise(p=0.5),
                SaltPepperNoise(p=0.3),
            ]),
            OneOf([
                Curve(period=20, amplitude=(3, 8)),
                Emboss(p=0.3),
            ])
        ]),
        num_image=10000  # 脏数据生成 10000 张
    )

# List of configurations to run
configs = [
    tibetan_chn_mixed(),
    tibetan_chn_eng_mixed(),
    tibetan_chn_eng_dirty(),
]
