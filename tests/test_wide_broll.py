"""Offline wide B-roll contracts and real synthetic deliveries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import synth
import synth_broll
from test_assemble_picture import insert, lane, project, write_document
from test_landscape import _context
from test_pictures_sheet import ask
from test_stock_sheet import search

from yap_studio import ffmpeg, paths, receipt
from yap_studio.canvas import canvas, require_supported
from yap_studio.errors import YapError
from yap_studio.media.catalog import add_paths
from yap_studio.media.sheet import make_sheet
from yap_studio.pictures import wavespeed
from yap_studio.project.model import Anchor, Element, TimelineLane
from yap_studio.stages import assemble_picture as picture
from yap_studio.stages.broll_match import _geometry
from yap_studio.stages.timeline import ResolvedElement, resolve_timeline


def test_wide_guard_accepts_broll_and_refuses_split(tmp_path):
    context = _context(tmp_path)
    for kind in ('insert', 'overlay'):
        item = Element('broll-sample', kind, Anchor(time=0), fields={'media': {'fit': 'cover'},
                       'placement': 'box'})
        require_supported(context, {'broll': TimelineLane((item,), lane='broll')})
    split = Element('broll-split', 'insert', Anchor(time=0), fields={'media': {'fit': 'split'}})
    with pytest.raises(YapError) as error:
        require_supported(context, {'broll': TimelineLane((split,), lane='broll')})
    assert 'split' in str(error.value).lower()
    assert 'cover' in error.value.suggested_action


@pytest.mark.parametrize('size', ['180x320', '240x240', '320x180'])
def test_wide_insert_filter_uses_delivery_frame(tmp_path, size):
    source = picture._source(synth_broll.png(tmp_path / 'still.png', size=size), [], [],
                             owner='broll-still')
    item = ResolvedElement('broll-still', 'insert', 'broll', 0, 15, 'front',
                           {'media': {'fit': 'cover'}})
    graph = picture._Graph()
    picture._insert(graph, source, item, size=(1920, 1080))
    filters = ';'.join(graph.filters)
    assert '1920:1080' in filters
    if size != '320x180':
        assert 'force_original_aspect_ratio=decrease' in filters
        assert 'boxblur=24:2' in filters and 'brightness=-0.12' in filters
    else:
        assert 'boxblur' not in filters


def test_wide_colour_measurement_follows_visible_foreground():
    item = ResolvedElement('broll-still', 'insert', 'broll', 0, 15, 'front',
                           {'media': {'fit': 'contain'}})
    expression, width, height = _geometry(item, frame=canvas('horizontal'))
    assert (width, height) == (192, 108)
    assert 'scale=192:108:force_original_aspect_ratio=decrease' in expression


def test_horizontal_stock_search_and_pick_sheet(tmp_path, fake_pexels):
    from test_stock_provider import calls

    write_document(tmp_path / 'project.json', {
        'schema': 'yap-studio/project@1', 'name': 'wide', 'shape': 'horizontal',
        'takes': [{'id': 't1', 'path': 'take.mov'}], 'broll_sources': ['stock'],
    })
    data = search(tmp_path)
    searches = [args for args in calls(fake_pexels) if '--orientation' in args]
    assert searches and searches[0][searches[0].index('--orientation') + 1] == 'landscape'
    assert data['shape'] == 'horizontal'
    html = (tmp_path / 'stock/sheets/s1/pick-sheet.html').read_text()
    assert 'aspect-ratio:16/9' in html


def test_horizontal_generated_size_reaches_quote_and_submission(tmp_path, fake_wavespeed):
    from test_pictures_provider import calls

    write_document(tmp_path / 'project.json', {
        'schema': 'yap-studio/project@1', 'name': 'wide', 'shape': 'horizontal',
        'takes': [{'id': 't1', 'path': 'take.mov'}], 'broll_sources': ['folder'],
    })
    data = ask(tmp_path)
    assert data['size'] == '1920*1080'
    wavespeed.generate('A synthetic park', tmp_path / 'generated', dry_run=True, size=data['size'])
    requests = [args for args in calls(fake_wavespeed) if args[0] in {'price', 'run'}]
    assert requests and all('size=1920*1080' in args for args in requests)


@pytest.mark.parametrize('kind', ['video', 'image'])
def test_horizontal_media_contact_sheet(tmp_path, kind, capsys):
    source = (synth_broll.clip(tmp_path / 'clip.mov', size='180x320', duration=.3)
              if kind == 'video' else synth_broll.png(tmp_path / 'photo.png', size='180x320'))
    added = add_paths([source])
    result = make_sheet(added['added'][0], frames=6, asr='none', shape='horizontal')
    info = ffmpeg.probe(Path(result['sheet']['path']), count_frames=False)
    assert info.width * 9 == info.height * 16
    assert make_sheet(added['added'][0], frames=6, asr='none', shape='horizontal')['noop']
    from yap_studio import cli

    assert cli.main(['media', 'sheet', added['added'][0], '--asr', 'none',
                     '--shape', 'horizontal', '--json']) == 0
    echoed = json.loads(capsys.readouterr().out)
    assert echoed['noop'] and echoed['sheet']['path'] == result['sheet']['path']


@pytest.mark.render
@pytest.mark.parametrize('kind,size', [('clip', '320x180'), ('portrait', '180x320'),
                                      ('square', '240x240'), ('still', '320x180'),
                                      ('overlay', '320x180')])
def test_wide_broll_real_render(tmp_path: Path, monkeypatch, kind, size):
    pictures = {
        'cut': synth.picture_source(tmp_path / 'cut.mov', 'color=c=0x55606a:s=1920x1080:r=30', 45),
        'take': synth.picture_source(tmp_path / 'take.mov', 'color=c=0x55606a:s=320x180:r=30', 45),
    }
    root = project(tmp_path / 'project', pictures, frames=45, hold_frames=0)
    data = json.loads((root / 'project.json').read_bytes())
    data.update(shape='horizontal')
    write_document(root / 'project.json', data)
    # Rebuild only the fixture receipt identity after changing its delivery shape.
    from yap_studio.project.receipts import StageRun, read_receipt
    from yap_studio.project.status import ProjectContext, cut_inputs

    saved = read_receipt(root / 'cut')
    with StageRun(root, 'cut') as run:
        for name, digest in cut_inputs(ProjectContext(root)).items():
            run.record_inputs(name, digest)
        run.outputs.update({item['path']: item for item in saved['outputs']})
        run.finish(extra={'plan': saved['plan']})
    source = (synth_broll.clip(root / 'clip.mov', size=size, colour='0x8ca0b9', duration=1)
              if kind == 'clip' else synth_broll.png(root / 'photo.png', size=size))
    shot = insert(root, source, 6, 30, identifier='broll-wide')
    if kind == 'overlay':
        shot.update(kind='overlay', placement='box', box=[1300, 300, 600, 400],
                    fit='cover', motion='none')
    shot['color_match'] = True
    lane(root, [shot], name='broll')
    timeline = resolve_timeline(root).require_valid()
    if kind == 'overlay':
        lane(root, [{**shot, 'box': [1300, 900, 600, 400]}], name='broll')
        assert any(f.rule == 'overlay_box' and f.severity == 'violation'
                   for f in resolve_timeline(root).findings)
        lane(root, [shot], name='broll')
    with paths.operation():
        result = picture.render_picture(root, root / 'wide.mp4', timeline=timeline,
            title_layers=[], caption_layer=None, work_dir=root / 'work', frame=canvas('horizontal'))
    info = ffmpeg.probe(root / 'wide.mp4')
    assert (info.width, info.height, info.video_frames) == (1920, 1080, 45)
    assert result.broll_match and result.broll_match[0]['id'] == 'broll-wide'
    pixels = synth.picture_pixels(root / 'wide.mp4', 15)
    assert pixels.shape == (1080, 1920, 3)
    if kind == 'overlay':
        assert np.linalg.norm(pixels[500, 1600].astype(float)
                              - pixels[500, 1000].astype(float)) > 30
    if kind in {'portrait', 'square'}:
        # Foreground remains whole; fill is visible and darker on both sides.
        center = pixels[540, 960].mean()
        assert pixels[540, 20].mean() > 5
        assert center > pixels[540, 20].mean() + 10
        np.testing.assert_allclose(pixels[540, 20], pixels[540, -20], atol=3)
    assert receipt.sha256_file(root / 'wide.mp4')


@pytest.mark.parametrize('source_kind', ['image', 'clip'])
@pytest.mark.parametrize('boxes,expected', [([(900, 10, 200, 180)], 'crop'),
    ([(200, 10, 200, 180), (200, 700, 200, 180)], 'contain')])
def test_wide_crop_keeps_existing_face_measurements(tmp_path, monkeypatch, source_kind,
                                                    boxes, expected):
    from fractions import Fraction
    from types import SimpleNamespace

    from yap_studio.faces import project as faces_project
    from yap_studio.faces.model import Box
    from yap_studio.gallery import still
    from yap_studio.stages.broll_framing import wide_insert

    source = SimpleNamespace(info=SimpleNamespace(width=1200, height=900),
                             path=tmp_path / 'synthetic', image=source_kind == 'image')
    measured = [SimpleNamespace(x=x, y=y, w=w, h=h) for x, y, w, h in boxes]
    monkeypatch.setattr(still, 'default_landmark_detector',
                         lambda: SimpleNamespace(measure=lambda _: [measured]))
    monkeypatch.setattr(faces_project, 'cut_faces', lambda *_: SimpleNamespace(
        video=SimpleNamespace(fps=Fraction(30)),
        frames=tuple(tuple(SimpleNamespace(box=Box(*box)) for box in boxes) for _ in range(30))))
    item = ResolvedElement('broll-face', 'insert', 'broll', 0, 30, 'front',
                            {'media': {'fit': 'cover'}})
    result = wide_insert(item, source, tmp_path, (1920, 1080))
    assert result.payload['media']['fit'] == expected
    if expected == 'crop':
        x, y, w, h = result.payload['media']['crop']
        for left, top, width, height in boxes:
            assert x <= left and y <= top
            assert x + w >= left + width and y + h >= top + height
    assert item.payload['media'] == {'fit': 'cover'}


def test_wide_picture_make_uses_approved_size(tmp_path, fake_wavespeed):
    from test_pictures_make import make
    from test_stock_provider import calls
    from test_stock_sheet import words

    root = tmp_path / 'video'
    write_document(root / 'project.json', {
        'schema': 'yap-studio/project@1', 'name': 'wide', 'shape': 'horizontal',
        'style': 'style.json', 'takes': [{'id': 't1', 'path': 'own.mov'}],
        'broll_sources': ['folder'], 'asr': {'engine': 'none'},
    })
    write_document(root / 'style.json', {'schema_version': 1,
                                       'broll': {'generated_pictures': 'ask'}})
    write_document(root / 'cut/cut-words.json', words().to_dict())
    ask(root)
    assert make(root, dry_run=True)['status'] == 'dry_run'
    result = make(root)
    assert result['status'] == 'ok' and not result['failed']
    from yap_studio.media.catalog import list_items

    assert list_items()[0]['generated']['size'] == '1920*1080'
    requests = [args for args in calls(fake_wavespeed) if args[0] in {'price', 'run'}]
    assert len(requests) >= 4 and all('size=1920*1080' in args for args in requests)


def test_wide_planner_defaults_to_landscape_cover(tmp_path):
    from test_planner_broll import catalog_item, phrase_words, requests

    from yap_studio.planner.broll import plan_broll

    item = catalog_item(tmp_path, kind='image')
    item.update(width=1920, height=1080, orientation='horizontal')
    write_document(tmp_path / 'project.json', {'shape': 'horizontal'})
    plan = plan_broll(requests(kind='insert'), phrase_words(), [item], {},
                      project_dir=tmp_path, frames=90)
    assert plan.lane()['elements'][0]['media']['fit'] == 'cover'


def test_wide_contain_keeps_the_whole_photo_even_with_default_push(tmp_path):
    source = picture._source(synth_broll.png(tmp_path / 'portrait.png', size='180x320'), [], [],
                             owner='broll-portrait')
    item = ResolvedElement('broll-portrait', 'insert', 'broll', 0, 15, 'front',
        {'media': {'fit': 'cover'}, 'motion': 'push', 'push_scale': 1.08, 'push_direction': 'in'})
    graph = picture._Graph()
    picture._insert(graph, source, item, size=(1920, 1080))
    filters = ';'.join(graph.filters)
    assert 'force_original_aspect_ratio=decrease' in filters
    assert 'eval=frame' not in filters


def test_wide_unknown_faces_use_contain(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from yap_studio.gallery import still
    from yap_studio.stages.broll_framing import wide_insert

    monkeypatch.setattr(still, 'default_landmark_detector', lambda: None)
    source = SimpleNamespace(info=SimpleNamespace(width=1200, height=900),
                             path=tmp_path / 'photo.png', image=True)
    item = ResolvedElement('broll-photo', 'insert', 'broll', 0, 30, 'front', {'media': {}})
    assert wide_insert(item, source, tmp_path, (1920, 1080)).payload['media']['fit'] == 'contain'


@pytest.mark.parametrize('kind,fields,fix', [
    ('insert', {'letters': {}, 'media': {}}, 'Remove letters'),
    ('overlay', {'placement': 'headroom'}, 'placement to box'),
])
def test_wide_vertical_only_broll_variants_have_fix_lines(tmp_path, kind, fields, fix):
    item = Element('broll-variant', kind, Anchor(time=0), fields=fields)
    with pytest.raises(YapError) as error:
        require_supported(_context(tmp_path), {'broll': TimelineLane((item,), lane='broll')})
    assert error.value.code == 'horizontal_unsupported'
    assert fix in error.value.suggested_action


def test_wide_stock_accepts_landscape_delivery_file():
    from types import SimpleNamespace

    from yap_studio.stock.provider import StockFile, best_file

    file = StockFile('wide', 'https://videos.pexels.com/video-files/1/wide.mp4',
                     1920, 1080, 30, 'hd')
    clip = SimpleNamespace(files=(file,))
    assert best_file(clip) is None
    assert best_file(clip, shape='horizontal') == file


def test_wide_photo_defaults_plan_full_frame(tmp_path):
    from test_planner_broll import catalog_item, phrase_words, requests

    from yap_studio.planner.broll import plan_broll, suggest_broll
    from yap_studio.style import resolve

    item = catalog_item(tmp_path, kind='image', orientation='horizontal')
    item.update(width=1920, height=1080)
    style = resolve('preset:clean_minimal').data
    write_document(tmp_path / 'project.json', {'shape': 'horizontal'})
    plan = plan_broll(requests(), phrase_words(), [item], style,
                      project_dir=tmp_path, frames=90)
    assert plan.lane()['elements'][0]['kind'] == 'insert'
    boxed = plan_broll(requests(box=[1300, 300, 600, 400]), phrase_words(), [item], style,
                       project_dir=tmp_path, frames=90)
    assert boxed.lane()['elements'][0]['kind'] == 'overlay'
    suggested = suggest_broll(phrase_words(), [item], style, shape='horizontal')
    candidate = suggested['phrases'][0]['candidates'][0]
    assert (candidate['kind'], candidate['placement'], candidate['box']) == (
        'insert', 'full_frame', None)


def test_wide_still_push_does_not_crop_a_measured_face(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from yap_studio.gallery import still
    from yap_studio.stages.broll_framing import wide_insert

    source = SimpleNamespace(info=SimpleNamespace(width=1920, height=1080),
                             path=tmp_path / 'photo.png', image=True)
    face = SimpleNamespace(x=50, y=10, w=100, h=100)
    monkeypatch.setattr(still, 'default_landmark_detector',
                         lambda: SimpleNamespace(measure=lambda _: [[face]]))
    item = ResolvedElement('broll-photo', 'insert', 'broll', 0, 30, 'front',
        {'media': {}, 'motion': 'push', 'push_scale': 1.08, 'push_direction': 'in'})
    result = wide_insert(item, source, tmp_path, (1920, 1080))
    assert result.payload['motion'] == 'still'
    assert item.payload['motion'] == 'push'
