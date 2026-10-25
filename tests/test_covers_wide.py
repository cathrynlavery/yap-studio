"""YouTube thumbnail contracts on the finished picture clock."""
from __future__ import annotations

import hashlib
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from yap_studio import ffmpeg
from yap_studio.covers.layout import check_geometry, plan_blocks
from yap_studio.covers.pick import TimelineSnapshot, extract_frame, pick_frame
from yap_studio.covers.render import CoverAssets, prepare_assets, render_cover
from yap_studio.errors import ValidationError
from yap_studio.faces.model import Box, FaceBox, FacesData, VideoIdentity
from yap_studio.gallery.still import FaceLandmarks
from yap_studio.style import resolve

COPY = {
    'photo': {'headline': 'Make *room* for better work'},
    'number': {'number': '5', 'label': 'better habits'},
    'before-after': {'before': 'Busy days', 'after': 'Good work'},
}


@pytest.mark.parametrize('look', COPY)
@pytest.mark.parametrize('side', ['left', 'right'])
def test_wide_looks_clear_face_safe_area_and_badge(look, side):
    face = (170, 220, 280, 330) if side == 'left' else (830, 220, 280, 330)
    plan = plan_blocks(look, COPY[look], resolve('preset:editorial_serif').data,
                       face_box=face, shape='horizontal')
    assert plan.speaker_side == side
    assert plan.speaker_shift_px == 0
    for block in plan.blocks:
        x, y, w, h = (block[key] for key in ('x', 'y', 'width', 'height'))
        assert 48 <= x < x + w <= 1232
        assert 48 <= y < y + h <= 672
        assert x + w <= 1080 or y + h <= 620
        assert x >= face[0] + face[2] + 24 or x + w <= face[0] - 24
        assert len(block['text'].split()) >= 2 or block['role'] == 'number'
    headlines = [b for b in plan.blocks if b['role'] == 'headline']
    if headlines:
        from yap_studio.render.text_metrics import text_width_em
        widths = [text_width_em(b['text'], b['font'], 400) for b in headlines]
        assert min(widths) / max(widths) > .55
        assert any(b.get('accent_start', -1) >= 0 for b in headlines)


def test_wide_refuses_long_copy_instead_of_shrinking_to_illegibility():
    with pytest.raises(ValidationError, match='shorten|orphan'):
        plan_blocks('photo', {'headline': 'WWWWWWWWW WWWWWWWWW WWWWWWWWW WWWWWWWWW'},
                    resolve('preset:editorial_serif').data, face_box=(170, 220, 280, 330),
                    shape='horizontal')


def test_wide_kicker_balances_copy_and_inset_clears_head_and_text():
    from yap_studio.covers.wide import check_box, check_face
    face = (170, 220, 280, 200)
    plan = plan_blocks('photo', {'headline': 'Make *room* for work',
                                'kicker': 'Small changes', 'inset': 'other.png'},
                       resolve('preset:editorial_serif').data, face_box=face,
                       shape='horizontal')
    inset = Box(*plan.inset_box)
    check_box(inset)
    check_face([{'x': inset.x, 'y': inset.y, 'width': inset.w, 'height': inset.h}], face)
    for block in plan.blocks:
        box = Box(*(block[k] for k in ('x', 'y', 'width', 'height')))
        assert inset.x1 <= box.x or box.x1 <= inset.x or inset.y1 <= box.y or box.y1 <= inset.y
    assert all(len(b['text'].split()) >= 2 for b in plan.blocks)


@pytest.mark.parametrize('face', [None, (500, 100, 280, 330)])
def test_wide_refuses_missing_face_or_no_headline_space(face):
    with pytest.raises(ValidationError, match='face|room'):
        plan_blocks('photo', COPY['photo'], resolve('preset:editorial_serif').data,
                    face_box=face, shape='horizontal')


def test_wide_geometry_badge_and_small_glyph_check():
    from yap_studio.covers.wide import check_legibility
    mask = np.zeros((720, 1280), dtype=np.uint8)
    mask[630:650, 1120:1200] = 255
    with pytest.raises(ValidationError, match='badge'):
        check_geometry(mask, shape='horizontal')
    text = np.zeros((720, 1280, 4), dtype=np.uint8)
    text[100:180, 660:1000] = 255
    backdrop = np.zeros_like(text)
    with pytest.raises(ValidationError, match='168x94'):
        check_legibility([{'role': 'headline', 'measured_box': [660, 100, 1000, 135]}],
                          text, backdrop)
    result = check_legibility([{'role': 'headline', 'measured_box': [660, 100, 1000, 180],
                                'box': [660, 100, 340, 80]}], text, backdrop)
    assert result['passed'] and result['preview'] == [168, 94]
    with pytest.raises(ValidationError, match='168x94'):
        check_legibility([{'role': 'label', 'measured_box': [660, 100, 1000, 135]}],
                          text, backdrop, look='number')


def test_wide_pick_prefers_open_lit_sharp_face_with_headline_room():
    frames = []
    for index in range(60):
        x = 810 if index == 0 else 280
        frames.append((FaceBox(Box(x, 240, 350, 400), .99, 'detected'),))
    faces = FacesData('measured', None, VideoIdentity('cut.mp4', 'hash', 60,
                      Fraction(30), 1920, 1080), {}, {}, tuple(frames), ())
    def image(index):
        pixels = np.indices((720, 1280)).sum(axis=0) % 2 * 70 + 100
        return (pixels if index != 15 else pixels // 8).astype(np.uint8)
    class Detector:
        def measure(self, images):
            return [[FaceLandmarks(190, 160, 233, 267, .01 if '00030' in p.name else .4,
                                  .01 if '00030' in p.name else .4, 0, 0, 1)] for p in images]
    picked = pick_frame(None, faces, TimelineSnapshot(60, 0, ()), shape='horizontal',
                        image_loader=image, landmark_detector=Detector())
    assert picked.frame == 45
    rows = {r['frame']: r for r in picked.candidates}
    assert rows[45]['headline_room'] > rows[0]['headline_room']
    assert rows[15]['face_luma'] < rows[45]['face_luma']
    assert rows[30]['eyes_open'] is False


def test_wide_pick_checks_later_batches(monkeypatch):
    frames = tuple((FaceBox(Box(800 if i < 360 else 250, 240, 350, 400), .9,
                                 'detected'),) for i in range(375))
    faces = FacesData('measured', None, VideoIdentity('cut.mp4', 'hash', 375,
                      Fraction(30), 1920, 1080), {}, {}, frames, ())
    monkeypatch.setattr('yap_studio.gallery.still._sharpness', lambda *_: 100.)
    picked = pick_frame(None, faces, TimelineSnapshot(375, 0, ()), shape='horizontal',
                        image_loader=lambda _: np.full((720, 1280), 130, np.uint8))
    assert picked.frame == 360 and len(picked.candidates) == 25


def test_long_wide_pick_samples_the_whole_edit_with_bounded_decodes(monkeypatch):
    centre = (FaceBox(Box(800, 240, 350, 400), .9, 'detected'),)
    left = (FaceBox(Box(250, 240, 350, 400), .9, 'detected'),)
    frames = (centre,) * 5880 + (left,) * 120
    faces = FacesData('measured', None, VideoIdentity('cut.mp4', 'hash', 6000,
                      Fraction(30), 1920, 1080), {}, {}, frames, ())
    monkeypatch.setattr('yap_studio.gallery.still._sharpness', lambda *_: 100.)
    picked = pick_frame(None, faces, TimelineSnapshot(6000, 0, ()), shape='horizontal',
                        image_loader=lambda _: np.broadcast_to(np.uint8(130), (720, 1280)))
    assert picked.frame >= 5880
    assert len(picked.candidates) <= 120


@pytest.mark.render
def test_wide_extract_and_assets_keep_landscape(tmp_path):
    clip = tmp_path / 'cut.mp4'
    ffmpeg.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                'testsrc2=size=1920x1080:rate=30', '-frames:v', '64', str(clip)], timeout_s=60)
    picture = extract_frame(clip, 31, tmp_path / 'wide.png', shape='horizontal')
    info = ffmpeg.probe(picture)
    assert (info.width, info.height) == (1280, 720)
    expected = tmp_path / 'full-decode.png'
    ffmpeg.run(['ffmpeg', '-v', 'error', '-i', str(clip), '-vf',
                'select=eq(n\\,31),scale=1280:720:force_original_aspect_ratio=increase,'
                'crop=1280:720', '-frames:v', '1', '-pix_fmt', 'rgb24', str(expected)],
               timeout_s=60)
    assert picture.read_bytes() == expected.read_bytes()
    assets = prepare_assets(clip, 0, tmp_path / 'assets', 'number', shape='horizontal',
                            face_box=(250, 240, 350, 400),
                            matte_provider=lambda *_: pytest.fail('wide needs no matte'))
    assert assets.face_box == pytest.approx((250 * 2/3, 160, 350 * 2/3, 400 * 2/3))
    assert assets.cutout is None and assets.shape == 'horizontal'


@pytest.mark.render
@pytest.mark.parametrize(('look', 'values'), [
    *COPY.items(), ('number', {'number': '12', 'label': 'build better daily habits'}),
])
@pytest.mark.parametrize('side', ['left', 'right'])
def test_wide_renders_checked_photo_at_1280x720(tmp_path, look, values, side):
    frame = tmp_path / 'frame.png'
    ffmpeg.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                'color=c=gray:s=1280x720:r=30', '-frames:v', '1', str(frame)], timeout_s=60)
    face = (170, 220, 280, 330) if side == 'left' else (830, 220, 280, 330)
    assets = CoverAssets(frame, face_box=face, shape='horizontal')
    result = render_cover(tmp_path, assets, resolve('preset:editorial_serif'), look, values)
    info = ffmpeg.probe(result.png)
    assert (info.width, info.height) == (1280, 720)
    assert result.checks['geometry']['passed']
    assert result.checks['contrast']['passed']
    assert result.checks['legibility']['passed']
    assert result.checks['face_clearance']['passed']
    if look == 'number':
        number, *labels = [b['measured_box'] for b in result.checks['blocks']]
        cap_height = number[3] - number[1]
        gap = labels[0][1] - number[3]
        assert 1 <= len(labels) <= 2
        assert gap == pytest.approx(cap_height / 4, abs=8)
        for label in labels:
            assert (number[0] + number[2]) / 2 == pytest.approx(
                (label[0] + label[2]) / 2, abs=8)


@pytest.mark.render
@pytest.mark.parametrize(('look', 'digest'), [
    ('photo', '2ce059a3028a1d3fbbcde7ca0451c21ff14a8f5f0b7ba44cd8d4c8a4694d2cef'),
    ('number', '35533732261fe72e07253d53c1a2b716b1d6f5be8c871bb97cc3f24fd519f960'),
    ('hook', '16226e8d0e1997f1eb06693c5eac9ebe83067a7afe15992979def31fdec75684'),
    ('before-after', '52aed414cdeea4fae2f7127c005049ea6fb1f33ea7a088bf35343488bbc4ef84'),
])
def test_vertical_cover_fixture_bytes_unchanged(tmp_path: Path, look, digest):
    import synth
    values = {'photo': {'headline': 'A *better* night'},
              'number': {'number': '5', 'label': 'better nights'},
              'hook': {'headline': 'A *better* night', 'label': 'A real result'},
              'before-after': {'before': '8 hours', 'after': '2 hours'}}
    assets = synth.cover_assets(tmp_path)
    result = render_cover(tmp_path, assets, resolve('preset:editorial_serif'), look, values[look])
    assert hashlib.sha256(result.png.read_bytes()).hexdigest() == digest


@pytest.mark.parametrize('headline', ['Fuck this', '*Fuck* this'])
def test_wide_bleep_preserves_accent_span(headline):
    style = resolve('preset:editorial_serif').data
    style['music']['bleep_swears'] = True
    plan = plan_blocks('photo', {'headline': headline}, style,
                       face_box=(170, 220, 280, 330), shape='horizontal')
    assert ' '.join(b['text'] for b in plan.blocks) == 'F*** this'
    if headline.startswith('*'):
        assert plan.blocks[0]['accent_start'] == 0
        assert plan.blocks[0]['accent_end'] == 4


def test_wide_pick_tries_requested_layout_on_ranked_candidates(monkeypatch):
    boxes = [Box(600, 360, 720, 600)] * 15 + [Box(375, 375, 225, 360)] * 15
    faces = FacesData('measured', None, VideoIdentity('cut.mp4', 'hash', 30,
                      Fraction(30), 1920, 1080), {}, {},
                      tuple((FaceBox(box, .99, 'detected'),) for box in boxes), ())
    monkeypatch.setattr('yap_studio.gallery.still._sharpness', lambda *_: 100.)
    def layout(face):
        return plan_blocks('photo', {'headline': 'Good work'},
                           resolve('preset:editorial_serif').data,
                           face_box=face, shape='horizontal')
    picked = pick_frame(None, faces, TimelineSnapshot(30, 0, ()), shape='horizontal',
                        image_loader=lambda _: np.full((720, 1280), 130, np.uint8),
                        layout=layout)
    assert picked.frame == 15
    assert picked.candidates[0]['placement_error']
    with pytest.raises(ValidationError, match='room'):
        pick_frame(None, faces, TimelineSnapshot(30, 0, ()), frame=0,
                   shape='horizontal', layout=layout)


@pytest.mark.parametrize('stem_width', [1, 16])
def test_wide_preview_checks_surviving_strokes_and_contrast(stem_width):
    from yap_studio.covers.wide import _preview, check_legibility
    text = np.zeros((720, 1280, 4), np.uint8)
    text[200:300, 700:700 + stem_width] = 255
    backdrop = np.zeros_like(text)
    blocks = [{'role': 'headline', 'box': [680, 180, 100, 140],
               'measured_box': [700, 200, 700 + stem_width, 300]}]
    assert (300 - 200) * 94 / 720 > 8  # The old height-only check accepted either stem.
    if stem_width == 1:
        assert _preview(text[:, :, 3:4] / 255).max() == pytest.approx(.125)
        with pytest.raises(ValidationError, match='168x94'):
            check_legibility(blocks, text, backdrop)
    else:
        assert check_legibility(blocks, text, backdrop)['passed']
        backdrop[:, :, :3] = 255
        with pytest.raises(ValidationError, match='168x94'):
            check_legibility(blocks, text, backdrop)


@pytest.mark.parametrize('face', [(0, 200, 450, 400), (170, 100, 280, 330),
                                  (1200, 220, 200, 300), (170, 500, 280, 300)])
def test_wide_refuses_clipped_face_or_head(face):
    from yap_studio.covers.wide import check_face
    with pytest.raises(ValidationError, match='canvas|clipped'):
        check_face([], face)


def test_wide_pick_retains_ranked_interior_face():
    interior = FaceBox(Box(250, 375, 225, 360), .99, 'detected')
    clipped = FaceBox(Box(0, 300, 675, 600), .99, 'detected')
    faces = FacesData('measured', None, VideoIdentity('cut.mp4', 'hash', 1,
                      Fraction(30), 1920, 1080), {}, {}, ((interior, clipped),), ())
    picked = pick_frame(None, faces, TimelineSnapshot(1, 0, ()), shape='horizontal',
                        image_loader=lambda _: np.full((720, 1280), 130, np.uint8))
    assert picked.face_box == (250, 375, 225, 360)


def test_wide_accent_phrase_stays_on_one_line_or_refuses_copy():
    with pytest.raises(ValidationError, match='shorten') as error:
        plan_blocks('photo', {'headline': 'Make *more room* for better work'},
                    resolve('preset:editorial_serif').data,
                    face_box=(170, 220, 280, 330), shape='horizontal')
    assert error.value.suggested_action == 'Shorten the headline or mark a shorter accent phrase.'
    plan = plan_blocks('photo', {'headline': '*more room*'},
                       resolve('preset:editorial_serif').data,
                       face_box=(170, 220, 280, 330), shape='horizontal')
    accented = [b for b in plan.blocks if 'accent_start' in b]
    assert len(accented) == 1
    block = accented[0]
    assert block['text'][block['accent_start']:block['accent_end']] == 'more room'
    with pytest.raises(ValidationError, match='shorten'):
        plan_blocks('photo', {'headline': 'Make *WWWWWWWWW WWWWWWWWW* work'},
                    resolve('preset:editorial_serif').data,
                    face_box=(170, 220, 280, 330), shape='horizontal')


def test_wide_plain_headline_does_not_resolve_unused_accent():
    style = resolve('preset:editorial_serif').data
    style['brand'] = {'accent_color': '#101010'}
    for section, field in [('captions', 'highlight_color'), ('hook', 'accent_color'),
                           ('graphics', 'accent_color')]:
        style[section][field] = '#101010'
    plan = plan_blocks('photo', {'headline': 'Good work'}, style,
                       face_box=(170, 220, 280, 330), shape='horizontal')
    assert all(b['color'] == style['captions']['text_color'] for b in plan.blocks)
    with pytest.raises(ValidationError, match='accent'):
        plan_blocks('photo', {'headline': 'Good *work*'}, style,
                    face_box=(170, 220, 280, 330), shape='horizontal')


def test_wide_sampling_seeds_short_speaker_window():
    face = (FaceBox(Box(375, 375, 225, 360), .99, 'detected'),)
    faces = FacesData('measured', None, VideoIdentity('cut.mp4', 'hash', 1801,
                      Fraction(30), 1920, 1080), {}, {}, (face,) * 1801, ())
    picked = pick_frame(None, faces, TimelineSnapshot(1801, 0, (
        ('insert', 0, 15), ('insert', 30, 1801))), shape='horizontal',
        image_loader=lambda _: np.full((720, 1280), 130, np.uint8))
    assert picked.frame == 15
